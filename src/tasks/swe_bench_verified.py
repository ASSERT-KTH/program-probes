from __future__ import annotations

from src.tasks.base import TaskAdapter, ChatPrompt
from src.configs import TaskConfig

_SYSTEM_TEMPLATE = (
    "You are a software engineer working in a Linux container. "
    "The repository is checked out at /testbed and all dependencies are installed. "
    "Fix the reported issue by modifying the source files. "
    "Every response must contain exactly one bash command in this format:\n\n"
    "```mswea_bash_command\n"
    "your_command_here\n"
    "```\n\n"
    "When you are done, submit by running a command whose first output line is:\n"
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
)

_INSTANCE_TEMPLATE = (
    "Please solve this task: {{task}}\n\n"
    "The repository is at /testbed. Use bash commands to explore, edit files, and run tests.\n"
    "When you are confident the issue is fixed, submit by running:\n"
    "```mswea_bash_command\n"
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cd /testbed && git diff HEAD\n"
    "```"
)


def _make_eval_script(instance: dict) -> str:
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
        return make_test_spec(instance).eval_script
    except Exception:
        return ""


class SWEBenchVerifiedAdapter(TaskAdapter):
    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        from datasets import load_dataset as hf_load_dataset
        dataset_name = task_config.dataset or "princeton-nlp/SWE-bench_Verified"
        data = hf_load_dataset(dataset_name, split="test")
        instances = list(data)
        print(f"[swe-bench] computing eval scripts for {len(instances)} instances...", flush=True)
        for inst in instances:
            if "eval_script" not in inst or not inst["eval_script"]:
                inst["eval_script"] = _make_eval_script(inst)
        return instances

    def format_prompt(self, sample: dict) -> ChatPrompt:
        problem = sample["problem_statement"]
        repo = sample.get("repo", "the repository")
        user_content = (
            f"Repository: {repo}\n\n"
            f"Issue:\n{problem}\n\n"
            "The repository is checked out at /testbed. "
            "Fix the issue by modifying the source files. "
            "When done, submit by running a command whose first output line is:\n"
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        )
        return ChatPrompt(user_content=user_content, system_content=_SYSTEM_TEMPLATE)

    def check_correct(self, generated: str, sample: dict) -> bool:
        # Correctness for SWE-bench is evaluated inside the sandbox via evaluate().
        return False

    def group_id(self, sample: dict) -> str:
        return sample["instance_id"]

    def sample_id(self, sample: dict) -> str:
        return sample["instance_id"]
