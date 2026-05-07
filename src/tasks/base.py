from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.configs import TaskConfig


@dataclass
class ChatPrompt:
    user_content: str


class TaskAdapter(ABC):
    @abstractmethod
    def load_dataset(self, task_config: TaskConfig) -> list[dict]: ...

    @abstractmethod
    def format_prompt(self, sample: dict) -> ChatPrompt: ...

    @abstractmethod
    def check_correct(self, generated: str, sample: dict) -> bool: ...

    @abstractmethod
    def group_id(self, sample: dict) -> str: ...

    @abstractmethod
    def sample_id(self, sample: dict) -> str: ...
