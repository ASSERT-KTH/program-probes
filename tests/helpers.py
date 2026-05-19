import torch
from pathlib import Path
from src.configs import ModelConfig, GenerationConfig, TaskConfig
from src.models.base import ModelAdapter, GenerationResult
from src.tasks.base import TaskAdapter, ChatPrompt


class MockTokenizer:
    """Minimal tokenizer mock — only needs decode()."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "mock decoded text"


class MockModelAdapter(ModelAdapter):
    HIDDEN_DIM = 64
    FIXED_RAW = "```python\ndef f(x): return x\n```"
    _N_TOKENS = 10

    def load_tokenizer(self, model_config: ModelConfig) -> None:
        self._tokenizer = MockTokenizer()

    def load_for_generation(self, model_config: ModelConfig, gen_config: GenerationConfig, max_model_len: int = 4096) -> None:
        pass

    def load_for_extraction(self, model_config: ModelConfig, gen_config: GenerationConfig) -> None:
        self._hidden_dim = self.HIDDEN_DIM
        self._tokenizer = MockTokenizer()

    def build_prompt(self, prompt: ChatPrompt) -> list[int]:
        return list(range(5))

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        gen_config: GenerationConfig,
    ) -> list[GenerationResult]:
        return [
            GenerationResult(
                prompt_token_ids=ids,
                generated_token_ids=list(range(self._N_TOKENS)),
                raw_text=self.FIXED_RAW,
            )
            for ids in prompt_token_ids
        ]

    def extract_hidden_states(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        layer_indices: list[int],
        stride: int,
    ) -> list[dict[int, torch.Tensor]]:
        n_steps = max(1, (self._N_TOKENS) // stride)
        return [
            {li: torch.randn(n_steps, self.HIDDEN_DIM, dtype=torch.float16) for li in layer_indices}
            for _ in sequences
        ]


class MockTaskAdapter(TaskAdapter):
    N_SAMPLES = 20

    def load_dataset(self, task_config: TaskConfig) -> list[dict]:
        return [{"task_id": f"task_{i}", "idx": i} for i in range(self.N_SAMPLES)]

    def format_prompt(self, sample: dict) -> ChatPrompt:
        return ChatPrompt(user_content=f"prompt_{sample['idx']}")

    def check_correct(self, generated: str, sample: dict) -> bool:
        return sample["idx"] % 2 == 0

    def group_id(self, sample: dict) -> str:
        return f"group_{sample['idx'] % 5}"

    def sample_id(self, sample: dict) -> str:
        return sample["task_id"]


def make_synthetic_pt(
    tmp_path: Path,
    run_id: str,
    sample_id: str,
    gen_idx: int,
    probe_layers: list[int],
    hidden_dim: int,
    n_steps: int,
    labels: dict,
) -> Path:
    out_dir = tmp_path / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    activations = {li: torch.randn(n_steps, hidden_dim, dtype=torch.float16) for li in probe_layers}
    data = {
        "activations": activations,
        "labels": labels,
        "sample_id": sample_id,
        "group_id": f"group_{sample_id}",
        "generation_idx": gen_idx,
        "metadata": {
            "prompt_token_ids": [0, 1, 2],
            "raw_text": "```python\ndef f(x): return x\n```",
            "task_sample": {"task_id": sample_id},
        },
    }
    fname = out_dir / f"{sample_id}_gen{gen_idx}.pt"
    torch.save(data, fname)
    return fname
