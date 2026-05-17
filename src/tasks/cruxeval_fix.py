import json
import re
import subprocess
import sys
from pathlib import Path
from src.tasks.base import TaskAdapter, ChatPrompt
from src.configs import TaskConfig

_SYSTEM_PROMPT = (
    "You are an expert Python programmer and debugger. "
    "You always reason carefully before responding, using the following format:\n\n"
    "<think>\nyour step-by-step analysis of the bug\n</think>\n"
    "your corrected Python function (only the function, no explanation)"
)

_FIX_SUFFIX = "Identify the bug and provide the corrected function."


def _extract_code(generated: str) -> str:
    # Strip thinking block first so fallback doesn't capture it
    text = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text


class CruxEvalFixAdapter(TaskAdapter):
    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        if task_config.pairs_path is None:
            raise ValueError("cruxeval_fix requires pairs_path in task config")
        pairs = json.loads(Path(task_config.pairs_path).read_text())
        samples = []
        for p in pairs:
            samples.append({
                "sample_id": f"bug__{p['pair_id']}",
                "pair_id": p["pair_id"],
                "original_id": p["original_id"],
                "buggy_code": p["buggy_code"],
                "input_str": p["input_str"],
                "correct_output": p["correct_output"],
                "wrong_output": p["wrong_output"],
                "mutation_type": p["mutation_type"],
            })
        return samples

    def format_prompt(self, sample: dict) -> ChatPrompt:
        user_content = (
            f"The following Python function has a bug.\n\n"
            f"```python\n{sample['buggy_code']}\n```\n\n"
            f"When called as `f({sample['input_str']})`, it returns `{sample['wrong_output']}` "
            f"but the correct output should be `{sample['correct_output']}`.\n\n"
            f"{_FIX_SUFFIX}"
        )
        return ChatPrompt(user_content=user_content, system_content=_SYSTEM_PROMPT)

    def check_correct(self, generated: str, sample: dict) -> bool:
        code = _extract_code(generated)
        test_script = f"{code}\nassert {sample['correct_output']} == f({sample['input_str']})"
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                timeout=sample.get("execution_timeout", 10),
                capture_output=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def group_id(self, sample: dict) -> str:
        return sample["original_id"]

    def sample_id(self, sample: dict) -> str:
        return sample["sample_id"]
