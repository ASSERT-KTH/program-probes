import pytest
from src.tasks.cruxeval_fix import CruxEvalFixAdapter, _extract_code
from src.tasks.base import ChatPrompt
from src.configs import TaskConfig


@pytest.fixture
def adapter():
    return CruxEvalFixAdapter()


def _make_sample():
    return {
        "sample_id": "bug__sample_0__off_by_one_minus",
        "pair_id": "sample_0__off_by_one_minus",
        "original_id": "sample_0",
        "buggy_code": "def f(nums):\n    output = []\n    for n in nums:\n        output.append((nums.count(n), n))\n    output.sort(reverse=0)\n    return output",
        "input_str": "[1, 1, 3, 1, 3, 1]",
        "correct_output": "[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]",
        "wrong_output": "[(2, 3), (2, 3), (4, 1), (4, 1), (4, 1), (4, 1)]",
        "mutation_type": "off_by_one_minus",
    }


def test_format_prompt_returns_chat_prompt(adapter):
    prompt = adapter.format_prompt(_make_sample())
    assert isinstance(prompt, ChatPrompt)
    assert isinstance(prompt.user_content, str)
    assert isinstance(prompt.system_content, str)


def test_format_prompt_includes_buggy_code(adapter):
    sample = _make_sample()
    prompt = adapter.format_prompt(sample)
    assert "reverse=0" in prompt.user_content


def test_format_prompt_includes_io(adapter):
    sample = _make_sample()
    prompt = adapter.format_prompt(sample)
    assert sample["input_str"] in prompt.user_content
    assert sample["wrong_output"] in prompt.user_content
    assert sample["correct_output"] in prompt.user_content


def test_format_prompt_system_mentions_think(adapter):
    prompt = adapter.format_prompt(_make_sample())
    assert "<think>" in prompt.system_content


def test_check_correct_with_correct_code(adapter):
    sample = _make_sample()
    correct_code = (
        "def f(nums):\n"
        "    output = []\n"
        "    for n in nums:\n"
        "        output.append((nums.count(n), n))\n"
        "    output.sort(reverse=True)\n"
        "    return output"
    )
    assert adapter.check_correct(correct_code, sample) is True


def test_check_correct_with_buggy_code(adapter):
    sample = _make_sample()
    assert adapter.check_correct(sample["buggy_code"], sample) is False


def test_check_correct_strips_fence(adapter):
    sample = _make_sample()
    fenced = (
        "```python\n"
        "def f(nums):\n"
        "    output = []\n"
        "    for n in nums:\n"
        "        output.append((nums.count(n), n))\n"
        "    output.sort(reverse=True)\n"
        "    return output\n"
        "```"
    )
    assert adapter.check_correct(fenced, sample) is True


def test_check_correct_strips_thinking_block(adapter):
    sample = _make_sample()
    with_think = (
        "<think>\nThe bug is reverse=0 instead of True.\n</think>\n"
        "```python\n"
        "def f(nums):\n"
        "    output = []\n"
        "    for n in nums:\n"
        "        output.append((nums.count(n), n))\n"
        "    output.sort(reverse=True)\n"
        "    return output\n"
        "```"
    )
    assert adapter.check_correct(with_think, sample) is True


def test_sample_id(adapter):
    assert adapter.sample_id(_make_sample()) == "bug__sample_0__off_by_one_minus"


def test_group_id(adapter):
    assert adapter.group_id(_make_sample()) == "sample_0"


def test_extract_code_strips_thinking():
    text = "<think>\nsome reasoning\n</think>\ndef f(x):\n    return x + 1"
    assert _extract_code(text) == "def f(x):\n    return x + 1"


def test_extract_code_python_fence():
    text = "```python\ndef f(x):\n    return x\n```"
    assert _extract_code(text) == "def f(x):\n    return x"


def test_extract_code_picks_last_block():
    text = (
        "Partial:\n```python\nx = 1\n```\n"
        "Full:\n```python\ndef f(x):\n    return x\n```"
    )
    assert _extract_code(text) == "def f(x):\n    return x"


def test_load_dataset_requires_pairs_path(adapter):
    config = TaskConfig(dataset="cruxeval_fix", adapter="cruxeval_fix", pairs_path=None)
    with pytest.raises(ValueError, match="pairs_path"):
        adapter.load_dataset(config)


def test_load_dataset_from_file(adapter, tmp_path):
    pairs = [
        {
            "pair_id": "sample_0__off_by_one_minus",
            "original_id": "sample_0",
            "original_code": "def f(x): return x",
            "input_str": "1",
            "correct_output": "1",
            "buggy_code": "def f(x): return x - 1",
            "wrong_output": "0",
            "mutation_type": "off_by_one_minus",
        }
    ]
    p = tmp_path / "pairs.json"
    p.write_text(__import__("json").dumps(pairs))
    config = TaskConfig(dataset="cruxeval_fix", adapter="cruxeval_fix", pairs_path=str(p))
    samples = adapter.load_dataset(config)
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "bug__sample_0__off_by_one_minus"
    assert samples[0]["original_id"] == "sample_0"
