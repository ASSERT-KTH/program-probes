import pytest
from src.probes.will_be_correct import WillBeCorrectProbe
from src.probes.currently_correct import CurrentlyCorrectProbe
from src.probes.currently_compiles import CurrentlyCompilesProbe
from src.probes.currently_reduces_failing import CurrentlyReducesFailingProbe
from src.probes.currently_has_regressions import CurrentlyHasRegressionsProbe
from src.probes.base import TrajectoryContext, EditEvent
from tests.conftest import MockTaskAdapter
from src.configs import TaskConfig


@pytest.fixture
def task_adapter():
    return MockTaskAdapter()


def test_will_be_correct_true(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    sample = {"task_id": "task_0", "idx": 0}  # even => correct
    ctx = TrajectoryContext(sample=sample, generated_text="any")
    assert probe.compute_label(ctx) is True


def test_will_be_correct_false(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    sample = {"task_id": "task_1", "idx": 1}  # odd => incorrect
    ctx = TrajectoryContext(sample=sample, generated_text="any")
    assert probe.compute_label(ctx) is False


def test_will_be_correct_is_static(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    assert probe.is_dynamic is False


def test_dynamic_probes_return_empty_list_without_edit_history():
    probes = [
        CurrentlyCorrectProbe(),
        CurrentlyCompilesProbe(),
        CurrentlyReducesFailingProbe(),
        CurrentlyHasRegressionsProbe(),
    ]
    ctx = TrajectoryContext(
        sample={"task_id": "t0", "idx": 0},
        generated_text="x",
        edit_history=[],
    )
    for probe in probes:
        result = probe.compute_label(ctx)
        assert result == []


def test_dynamic_probes_are_dynamic():
    probes = [
        CurrentlyCorrectProbe(),
        CurrentlyCompilesProbe(),
        CurrentlyReducesFailingProbe(),
        CurrentlyHasRegressionsProbe(),
    ]
    for probe in probes:
        assert probe.is_dynamic is True


def test_currently_correct_per_edit():
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            EditEvent(step_idx=2, code="x", test_results={"passed": [], "failed": ["t1"]}),
        ],
    )
    probe = CurrentlyCorrectProbe()
    result = probe.compute_label(ctx)
    assert isinstance(result, list)
    assert len(result) == len(ctx.edit_history)
    assert result == [False]


def test_will_be_correct_scalar_return(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    sample = {"task_id": "task_4", "idx": 4}
    ctx = TrajectoryContext(sample=sample, generated_text="")
    label = probe.compute_label(ctx)
    assert isinstance(label, bool)


def test_currently_correct_uses_resolved_when_present():
    """When test_results has a 'resolved' key it takes precedence over raw failed count."""
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            # resolved=True even though failed list is non-empty
            EditEvent(step_idx=1, code="x", test_results={"passed": [], "failed": ["t1"], "resolved": True}),
        ],
    )
    probe = CurrentlyCorrectProbe()
    result = probe.compute_label(ctx)
    assert result == [True]


def test_currently_correct_falls_back_on_failed_count():
    """Without 'resolved', fall back to len(failed)==0."""
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            EditEvent(step_idx=0, code="x", test_results={"passed": ["t1"], "failed": []}),
            EditEvent(step_idx=1, code="y", test_results={"passed": [], "failed": ["t1"]}),
        ],
    )
    probe = CurrentlyCorrectProbe()
    result = probe.compute_label(ctx)
    assert result == [True, False]


def test_currently_compiles_per_edit():
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            EditEvent(step_idx=0, code="a", test_results=None, compiles=True),
            EditEvent(step_idx=2, code="b", test_results=None, compiles=None),
            EditEvent(step_idx=4, code="c", test_results=None, compiles=False),
        ],
    )
    probe = CurrentlyCompilesProbe()
    result = probe.compute_label(ctx)
    assert result == [True, None, False]


def test_currently_reduces_failing_per_edit():
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            # Baseline (first edit): 3 failing, no prior state → False
            EditEvent(step_idx=0, code="a", test_results={"passed": ["t1"], "failed": ["t2", "t3", "t4"]}),
            # Second edit: 2 failing < 3 (baseline) → True
            EditEvent(step_idx=2, code="b", test_results={"passed": ["t1", "t2"], "failed": ["t3", "t4"]}),
            # Third edit: still 2 failing < 3 (baseline) → True (baseline comparison, not prev-to-prev)
            EditEvent(step_idx=4, code="c", test_results={"passed": ["t1", "t2"], "failed": ["t3", "t4"]}),
        ],
    )
    probe = CurrentlyReducesFailingProbe()
    result = probe.compute_label(ctx)
    assert result == [False, True, True]


def test_currently_has_regressions_per_edit():
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            # First edit: no baseline → False
            EditEvent(step_idx=0, code="a", test_results={"passed": ["t1", "t2"], "failed": ["t3"]}),
            # Second edit: t1 (previously passed) now fails → True
            EditEvent(step_idx=2, code="b", test_results={"passed": ["t2"], "failed": ["t1", "t3"]}),
            # Third edit: no regression (t4 is new, not previously passed) → False
            EditEvent(step_idx=4, code="c", test_results={"passed": ["t1", "t2"], "failed": ["t4"]}),
        ],
    )
    probe = CurrentlyHasRegressionsProbe()
    result = probe.compute_label(ctx)
    assert result == [False, True, False]


def test_will_be_correct_uses_edit_history_when_present(task_adapter):
    """When edit_history is populated, use the last edit's resolved status."""
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        edit_history=[
            EditEvent(step_idx=0, code="a", test_results={"passed": [], "failed": ["t1"], "resolved": False}),
            EditEvent(step_idx=1, code="b", test_results={"passed": ["t1"], "failed": [], "resolved": True}),
        ],
    )
    probe = WillBeCorrectProbe(task_adapter)
    assert probe.compute_label(ctx) is True


def test_will_be_correct_falls_back_to_task_adapter(task_adapter):
    """Without edit_history, fall back to task_adapter.check_correct()."""
    sample = {"task_id": "task_1", "idx": 1}  # odd → False in MockTaskAdapter
    ctx = TrajectoryContext(sample=sample, generated_text="x", edit_history=[])
    probe = WillBeCorrectProbe(task_adapter)
    assert probe.compute_label(ctx) is False


def test_newly_added_probe_per_edit_names():
    """Ensure all probes report correct names and dynamic/static classifiers."""
    assert CurrentlyCorrectProbe().name == "currently_correct"
    assert CurrentlyCompilesProbe().name == "currently_compiles"
    assert CurrentlyReducesFailingProbe().name == "currently_reduces_failing"
    assert CurrentlyHasRegressionsProbe().name == "currently_has_regressions"
