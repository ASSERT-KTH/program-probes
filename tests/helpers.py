import torch
import torch.nn as nn
from pathlib import Path
from src.configs import ModelConfig, HardwareConfig, TaskConfig
from src.models.base import ModelAdapter
from src.tasks.base import TaskAdapter, ChatPrompt


class MockModelAdapter(ModelAdapter):
    HIDDEN_DIM = 64
    FIXED_OUTPUT = "def f(x): return x"
    _N_TOKENS = 10

    def load(self, model_config: ModelConfig, hardware_config: HardwareConfig) -> None:
        self._model_config = model_config
        self._layers = nn.ModuleList([nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM) for _ in range(40)])

    def get_layer_modules(self) -> list:
        return list(self._layers)

    def get_hidden_dim(self) -> int:
        return self.HIDDEN_DIM

    def tokenize(self, prompt: ChatPrompt) -> dict:
        return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    def generate(self, inputs: dict, max_new_tokens: int, temperature: float, top_p: float | None = None, top_k: int | None = None, min_p: float | None = None) -> tuple[str, list[int], str]:
        token_ids = list(range(self._N_TOKENS))
        return self.FIXED_OUTPUT, token_ids, self.FIXED_OUTPUT


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
        "n_captured_steps": n_steps,
        "metadata": {
            "prompt": "prompt",
            "generated_text": "generated",
            "task_sample": {"task_id": sample_id},
        },
    }
    fname = out_dir / f"{sample_id}_gen{gen_idx}.pt"
    torch.save(data, fname)
    return fname
