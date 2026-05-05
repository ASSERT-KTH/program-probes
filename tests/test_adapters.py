import pytest
from src.models.base import ModelAdapter
from src.tasks.base import TaskAdapter
from tests.helpers import MockModelAdapter, MockTaskAdapter


def test_mock_model_adapter_is_model_adapter(mock_model_adapter):
    assert isinstance(mock_model_adapter, ModelAdapter)


def test_mock_task_adapter_is_task_adapter(mock_task_adapter):
    assert isinstance(mock_task_adapter, TaskAdapter)


def test_model_adapter_load(mock_model_adapter, model_config, hardware_config):
    mock_model_adapter.load(model_config, hardware_config)
    assert mock_model_adapter.get_hidden_dim() == 64


def test_model_adapter_get_layer_modules(mock_model_adapter, model_config, hardware_config):
    mock_model_adapter.load(model_config, hardware_config)
    layers = mock_model_adapter.get_layer_modules()
    assert len(layers) > 0


def test_model_adapter_tokenize(mock_model_adapter, model_config, hardware_config):
    mock_model_adapter.load(model_config, hardware_config)
    result = mock_model_adapter.tokenize("hello")
    assert "input_ids" in result


def test_model_adapter_generate(mock_model_adapter, model_config, hardware_config):
    mock_model_adapter.load(model_config, hardware_config)
    inputs = mock_model_adapter.tokenize("hello")
    text, tokens = mock_model_adapter.generate(inputs, max_new_tokens=10, temperature=0.7)
    assert isinstance(text, str)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_task_adapter_load_dataset(mock_task_adapter, task_config):
    samples = mock_task_adapter.load_dataset(task_config)
    assert len(samples) == 20


def test_task_adapter_format_prompt(mock_task_adapter):
    from src.tasks.base import ChatPrompt
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
