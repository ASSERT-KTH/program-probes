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
    ctx = TrajectoryContext(sample=sample, generated_text="any", n_captured_steps=5)
    assert probe.compute_label(ctx) is True


def test_will_be_correct_false(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    sample = {"task_id": "task_1", "idx": 1}  # odd => incorrect
    ctx = TrajectoryContext(sample=sample, generated_text="any", n_captured_steps=5)
    assert probe.compute_label(ctx) is False


def test_will_be_correct_is_static(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    assert probe.is_dynamic is False


def test_dynamic_probes_raise_without_edit_history():
    probes = [
        CurrentlyCorrectProbe(),
        CurrentlyCompilesProbe(),
        CurrentlyReducesFailingProbe(),
        CurrentlyHasRegressionsProbe(),
    ]
    ctx = TrajectoryContext(
        sample={"task_id": "t0", "idx": 0},
        generated_text="x",
        n_captured_steps=3,
        edit_history=[],
    )
    for probe in probes:
        with pytest.raises(NotImplementedError):
            probe.compute_label(ctx)


def test_dynamic_probes_are_dynamic():
    probes = [
        CurrentlyCorrectProbe(),
        CurrentlyCompilesProbe(),
        CurrentlyReducesFailingProbe(),
        CurrentlyHasRegressionsProbe(),
    ]
    for probe in probes:
        assert probe.is_dynamic is True


def test_dynamic_stub_returns_list_of_correct_length_when_edits_present():
    ctx = TrajectoryContext(
        sample={},
        generated_text="x",
        n_captured_steps=5,
        edit_history=[
            EditEvent(step_idx=2, code="x", test_results={"passed": [], "failed": ["t1"]}),
        ],
    )
    # Stubs return [None] * n_captured_steps when edit_history is non-empty
    probe = CurrentlyCorrectProbe()
    result = probe.compute_label(ctx)
    assert isinstance(result, list)
    assert len(result) == ctx.n_captured_steps
    assert all(v is None for v in result)


def test_will_be_correct_scalar_return(task_adapter):
    probe = WillBeCorrectProbe(task_adapter)
    sample = {"task_id": "task_4", "idx": 4}
    ctx = TrajectoryContext(sample=sample, generated_text="", n_captured_steps=3)
    label = probe.compute_label(ctx)
    assert isinstance(label, bool)
