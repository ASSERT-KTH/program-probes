import subprocess
import sys
from src.tasks.base import TaskAdapter, ChatPrompt
from src.configs import TaskConfig

_INSTRUCTION_PREFIX = (
    "Please fix the buggy Python function below and provide the corrected version "
    "in a markdown code block. Do not include any explanation or text outside the code block."
)
_RESPONSE_PREFIX = "Below is the fixed Python script:\n```python"


def _extract_code(generated: str) -> str:
    if "```" in generated:
        return generated[:generated.index("```")].strip()
    return generated.strip()


class HumanEvalFixAdapter(TaskAdapter):
    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        from datasets import load_dataset
        data = load_dataset("bigcode/humanevalpack", "python", trust_remote_code=True)
        return list(data["test"])

    def format_prompt(self, sample: dict) -> ChatPrompt:
        buggy_code = sample["prompt"] + sample["buggy_solution"]
        user_content = f"{_INSTRUCTION_PREFIX}\n```python\n{buggy_code.strip()}\n```\n"
        return ChatPrompt(user_content=user_content, assistant_prefill=_RESPONSE_PREFIX)

    def check_correct(self, generated: str, sample: dict) -> bool:
        code = _extract_code(generated)
        test_code = code + "\n" + sample["test"] + "\ncheck(" + sample["entry_point"] + ")"
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                timeout=10,
                capture_output=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def group_id(self, sample: dict) -> str:
        return sample["task_id"]

    def sample_id(self, sample: dict) -> str:
        return sample["task_id"]
