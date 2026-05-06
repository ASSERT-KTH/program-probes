import subprocess
import sys
from src.tasks.base import TaskAdapter, ChatPrompt
from src.configs import TaskConfig

_FIX_SUFFIX = "Provide the complete fixed function in a final markdown code block at the end of your response."


def _extract_code(generated: str) -> str:
    import re
    blocks = re.findall(r"```(?:python)?\n(.*?)```", generated, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return generated.strip()


class HumanEvalFixAdapter(TaskAdapter):
    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        from datasets import load_dataset
        data = load_dataset("bigcode/humanevalpack", "python", trust_remote_code=True)
        return list(data["test"])

    def format_prompt(self, sample: dict) -> ChatPrompt:
        buggy_code = sample["prompt"] + sample["buggy_solution"]
        user_content = (
            f"Fix bugs in {sample['entry_point']}.\n\n"
            f"```python\n{buggy_code.strip()}\n```\n\n"
            f"{_FIX_SUFFIX}"
        )
        return ChatPrompt(user_content=user_content, assistant_prefill=None)

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
