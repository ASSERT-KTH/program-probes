import pytest
from src.configs import GenerationConfig
from src.models.base import ModelAdapter, GenerationResult
from src.tasks.base import TaskAdapter, ChatPrompt
from tests.helpers import MockModelAdapter, MockTaskAdapter


def test_mock_model_adapter_is_model_adapter(mock_model_adapter):
    assert isinstance(mock_model_adapter, ModelAdapter)


def test_mock_task_adapter_is_task_adapter(mock_task_adapter):
    assert isinstance(mock_task_adapter, TaskAdapter)


def test_model_adapter_load_for_extraction(mock_model_adapter, model_config, generation_config):
    mock_model_adapter.load_for_extraction(model_config, generation_config)
    assert mock_model_adapter.HIDDEN_DIM == 64


def test_model_adapter_build_prompt(mock_model_adapter, model_config, generation_config):
    mock_model_adapter.load_for_extraction(model_config, generation_config)
    token_ids = mock_model_adapter.build_prompt(ChatPrompt(user_content="hello"))
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0


def test_model_adapter_generate(mock_model_adapter, model_config, generation_config):
    mock_model_adapter.load_for_generation(model_config, generation_config)
    results = mock_model_adapter.generate([[0, 1, 2]], generation_config)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, GenerationResult)
    assert isinstance(r.raw_text, str)
    assert isinstance(r.generated_token_ids, list)


def test_model_adapter_extract_hidden_states(mock_model_adapter, model_config, generation_config):
    mock_model_adapter.load_for_extraction(model_config, generation_config)
    sequences = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    prompt_lengths = [5]
    layer_indices = [0, 1, 2]
    results = mock_model_adapter.extract_hidden_states(sequences, prompt_lengths, layer_indices, stride=3)
    assert len(results) == 1
    for li in layer_indices:
        assert li in results[0]
        assert results[0][li].ndim == 2
        assert results[0][li].shape[1] == MockModelAdapter.HIDDEN_DIM


def test_task_adapter_load_dataset(mock_task_adapter, task_config):
    samples = mock_task_adapter.load_dataset(task_config)
    assert len(samples) == 20


def test_task_adapter_format_prompt(mock_task_adapter):
    sample = {"task_id": "task_0", "idx": 0}
    prompt = mock_task_adapter.format_prompt(sample)
    assert isinstance(prompt, ChatPrompt)
    assert isinstance(prompt.user_content, str)


def test_task_adapter_check_correct_even(mock_task_adapter):
    sample = {"task_id": "task_0", "idx": 0}
    assert mock_task_adapter.check_correct("any", sample) is True


def test_task_adapter_check_correct_odd(mock_task_adapter):
    sample = {"task_id": "task_1", "idx": 1}
    assert mock_task_adapter.check_correct("any", sample) is False


def test_task_adapter_group_id(mock_task_adapter):
    sample = {"task_id": "task_7", "idx": 7}
    assert mock_task_adapter.group_id(sample) == "group_2"
