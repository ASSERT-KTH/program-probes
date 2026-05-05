from abc import ABC, abstractmethod
import torch.nn as nn
from src.configs import ModelConfig, HardwareConfig
from src.tasks.base import ChatPrompt


class ModelAdapter(ABC):
    @abstractmethod
    def load(self, model_config: ModelConfig, hardware_config: HardwareConfig) -> None: ...

    @abstractmethod
    def get_layer_modules(self) -> list[nn.Module]: ...

    @abstractmethod
    def get_hidden_dim(self) -> int: ...

    @abstractmethod
    def tokenize(self, prompt: ChatPrompt) -> dict: ...

    @abstractmethod
    def generate(
        self, inputs: dict, max_new_tokens: int, temperature: float,
        top_p: float | None = None, top_k: int | None = None, min_p: float | None = None,
    ) -> tuple[str, list[int], str]: ...
