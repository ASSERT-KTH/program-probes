from __future__ import annotations

from src.tasks.base import TaskAdapter, ChatPrompt
from src.configs import TaskConfig

_SYSTEM_TEMPLATE = (
    "You are a software engineer working in a Linux container. "
    "The repository is checked out at /app and all dependencies are installed. "
    "Fix the reported issue by modifying the source files. "
    "Every response must contain exactly one bash command in this format:\n\n"
    "```mswea_bash_command\n"
    "your_command_here\n"
    "```\n\n"
    "When you are done, submit by running a command whose first output line is:\n"
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
)


class SWEBenchProAdapter(TaskAdapter):
    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        from datasets import load_dataset as hf_load_dataset
        dataset_name = task_config.dataset or "ScaleAI/SWE-bench_Pro"
        data = hf_load_dataset(dataset_name, split="test")
        return list(data)

    def format_prompt(self, sample: dict) -> ChatPrompt:
        problem = sample["problem_statement"]
        repo = sample.get("repo", "the repository")
        user_content = (
            f"Repository: {repo}\n\n"
            f"Issue:\n{problem}\n\n"
            "The repository is checked out at /app. "
            "Fix the issue by modifying the source files. "
            "When done, submit by running a command whose first output line is:\n"
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        )
        return ChatPrompt(user_content=user_content, system_content=_SYSTEM_TEMPLATE)

    def check_correct(self, generated: str, sample: dict) -> bool:
        return False

    def group_id(self, sample: dict) -> str:
        return sample["instance_id"]

    def sample_id(self, sample: dict) -> str:
        return sample["instance_id"]
