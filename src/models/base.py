from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from src.configs import ModelConfig, GenerationConfig
from src.tasks.base import ChatPrompt


@dataclass
class GenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    raw_text: str


class ModelAdapter(ABC):
    @abstractmethod
    def load_tokenizer(self, model_config: ModelConfig) -> None: ...

    @abstractmethod
    def load_for_generation(self, model_config: ModelConfig, gen_config: GenerationConfig, max_model_len: int) -> None: ...

    @abstractmethod
    def load_for_extraction(self, model_config: ModelConfig, gen_config: GenerationConfig) -> None: ...

    @abstractmethod
    def build_prompt(self, prompt: ChatPrompt) -> list[int]: ...

    @abstractmethod
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        gen_config: GenerationConfig,
    ) -> list[GenerationResult]: ...

    @abstractmethod
    def extract_hidden_states(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        layer_indices: list[int],
        stride: int,
    ) -> list[dict[int, torch.Tensor]]:
        """Return one dict per sequence: {layer_idx: Tensor[n_steps, hidden_dim]}."""
        ...
