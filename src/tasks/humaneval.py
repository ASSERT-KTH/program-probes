import subprocess
import sys
from src.tasks.base import TaskAdapter, ChatPrompt
from src.configs import TaskConfig

_INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
_RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python"


class HumanEvalAdapter(TaskAdapter):
    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        from evalplus.data import get_human_eval_plus
        data = get_human_eval_plus()
        return [{"task_id": k, **v} for k, v in data.items()]

    def format_prompt(self, sample: dict) -> ChatPrompt:
        user_content = f"{_INSTRUCTION_PREFIX}\n```\n{sample['prompt'].strip()}\n```\n"
        return ChatPrompt(user_content=user_content, assistant_prefill=_RESPONSE_PREFIX)

    def check_correct(self, generated: str, sample: dict) -> bool:
        # Chat model outputs a full self-contained script; also try concatenation as fallback.
        candidates = [generated, sample["prompt"] + generated]
        for code in candidates:
            test_code = code + "\n" + sample["test"] + "\ncheck(" + sample["entry_point"] + ")"
            try:
                result = subprocess.run(
                    [sys.executable, "-c", test_code],
                    timeout=10,
                    capture_output=True,
                )
                if result.returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                pass
        return False

    def group_id(self, sample: dict) -> str:
        return sample["task_id"]

    def sample_id(self, sample: dict) -> str:
        return sample["task_id"]
