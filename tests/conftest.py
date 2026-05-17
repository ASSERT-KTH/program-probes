import pytest
from src.configs import ModelConfig, TaskConfig, GenerationConfig
from src.probes.base import TrajectoryContext, EditEvent
from tests.helpers import MockModelAdapter, MockTaskAdapter, make_synthetic_pt


@pytest.fixture
def mock_model_adapter():
    return MockModelAdapter()


@pytest.fixture
def mock_task_adapter():
    return MockTaskAdapter()


@pytest.fixture
def model_config():
    return ModelConfig(model_id="mock/model", probe_layers=[0, 1, 2], adapter="mock")


@pytest.fixture
def task_config():
    return TaskConfig(dataset="mock", adapter="mock", execution_timeout=5)


@pytest.fixture
def generation_config():
    return GenerationConfig(max_new_tokens=20, n_generations=2, temperature=0.7, stride=3, seed=42)


@pytest.fixture
def tmp_output_dir(tmp_path):
    output_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"
    output_dir.mkdir()
    cache_dir.mkdir()
    return tmp_path


@pytest.fixture
def simple_trajectory_ctx(mock_task_adapter):
    sample = {"task_id": "task_0", "idx": 0}
    return TrajectoryContext(
        sample=sample,
        generated_text=MockModelAdapter.FIXED_RAW,
        n_captured_steps=5,
        edit_history=[],
    )


@pytest.fixture
def trajectory_ctx_with_edits():
    sample = {"task_id": "task_0", "idx": 0}
    edits = [
        EditEvent(step_idx=1, code="x = 1", test_results={"passed": ["t1"], "failed": ["t2"]}),
        EditEvent(step_idx=3, code="x = 2", test_results={"passed": ["t1", "t2"], "failed": []}),
    ]
    return TrajectoryContext(
        sample=sample,
        generated_text="def f(x): return x",
        n_captured_steps=5,
        edit_history=edits,
    )
