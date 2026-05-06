import pytest
from src.tasks.humaneval_fix import HumanEvalFixAdapter, _extract_code
from src.tasks.base import ChatPrompt


@pytest.fixture
def adapter():
    return HumanEvalFixAdapter()


def _make_sample():
    return {
        "task_id": "HumanEval/0",
        "entry_point": "has_close_elements",
        "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
        "buggy_solution": "    return any(abs(a - b) > threshold for a, b in zip(numbers, numbers[1:]))\n",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0], 0.3) == True\n",
    }


def test_format_prompt_returns_chat_prompt(adapter):
    sample = _make_sample()
    prompt = adapter.format_prompt(sample)
    assert isinstance(prompt, ChatPrompt)
    assert isinstance(prompt.user_content, str)
    assert isinstance(prompt.assistant_prefill, str)


def test_format_prompt_includes_buggy_code(adapter):
    sample = _make_sample()
    prompt = adapter.format_prompt(sample)
    assert "has_close_elements" in prompt.user_content
    assert sample["buggy_solution"].strip() in prompt.user_content


def test_format_prompt_has_fix_instruction(adapter):
    sample = _make_sample()
    prompt = adapter.format_prompt(sample)
    assert "fix" in prompt.user_content.lower()


def test_check_correct_with_correct_code(adapter):
    sample = _make_sample()
    correct_code = (
        "from typing import List\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n"
    )
    assert adapter.check_correct(correct_code, sample) is True


def test_check_correct_with_wrong_code(adapter):
    sample = _make_sample()
    wrong_code = (
        "from typing import List\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    return False\n"
    )
    assert adapter.check_correct(wrong_code, sample) is False


def test_check_correct_strips_fence(adapter):
    sample = _make_sample()
    correct_code = (
        "from typing import List\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n"
        "```\n"
        "This is an explanation that should be ignored.\n"
    )
    assert adapter.check_correct(correct_code, sample) is True


def test_sample_id(adapter):
    sample = _make_sample()
    assert adapter.sample_id(sample) == "HumanEval/0"


def test_group_id(adapter):
    sample = _make_sample()
    assert adapter.group_id(sample) == "HumanEval/0"


def test_extract_code_strips_fence():
    text = "def foo():\n    return 1\n```\n### Explanation\nSome text."
    assert _extract_code(text) == "def foo():\n    return 1"


def test_extract_code_no_fence():
    text = "def foo():\n    return 1\n"
    assert _extract_code(text) == "def foo():\n    return 1"
