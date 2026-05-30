from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from src.configs import ModelConfig, GenerationConfig
from src.tasks.base import ChatPrompt


def hf_extract_hidden_states_chunked(
    model,
    token_ids: list[int],
    layer_indices: list[int],
    mask: list[int],
    stride: int,
    chunk_size: int,
    device,
) -> dict[int, torch.Tensor]:
    """Extract hidden states for a single sequence using KV-cache chunking.

    Equivalent to a full-sequence forward pass with output_hidden_states=True,
    but GPU memory is bounded by chunk_size rather than the full sequence length.
    Correctness relies on causal attention: token i attends only to tokens 0..i,
    so KV-cache carries exact context across chunk boundaries.
    """
    seq_len = len(token_ids)
    positions = [i for i, m in enumerate(mask) if m][::stride]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    collected: dict[int, dict[int, torch.Tensor]] = {li: {} for li in layer_indices}
    past_kv = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_input = input_ids[:, start:end]
        # Attention mask must cover past (KV-cached) + current chunk tokens
        chunk_attn = torch.ones(1, end, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                input_ids=chunk_input,
                attention_mask=chunk_attn,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )

        for pos in positions:
            if start <= pos < end:
                local_pos = pos - start
                for li in layer_indices:
                    collected[li][pos] = out.hidden_states[li + 1][0, local_pos].cpu()

        past_kv = out.past_key_values
        del out

    result = {}
    for li in layer_indices:
        tensors = [collected[li][pos] for pos in positions if pos in collected[li]]
        result[li] = torch.stack(tensors) if tensors else torch.zeros(0)
    return result


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
        extraction_masks: list[list[int]] | None = None,
        chunk_size: int | None = None,
    ) -> list[dict[int, torch.Tensor]]:
        """Return one dict per sequence: {layer_idx: Tensor[n_steps, hidden_dim]}.

        If extraction_masks is provided (one binary vector per sequence, same length),
        hidden states are extracted only at positions where mask == 1, then strided.
        When None, falls back to striding over the generated portion (prompt_lengths).
        If chunk_size is set, sequences longer than chunk_size are processed via
        KV-cache chunking to bound peak GPU memory.
        """
        ...
