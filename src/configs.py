from pydantic import BaseModel, field_validator, model_validator
from typing import Literal


class GenerationConfig(BaseModel):
    max_new_tokens: int = 1024
    n_generations: int = 10
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    stride: int = 5
    seed: int = 42


class HardwareConfig(BaseModel):
    device_map: str = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class ModelConfig(BaseModel):
    model_id: str
    probe_layers: list[int]
    adapter: str

    @field_validator("probe_layers")
    @classmethod
    def probe_layers_nonempty(cls, v):
        if not v:
            raise ValueError("probe_layers must not be empty")
        return v


class TaskConfig(BaseModel):
    dataset: str
    adapter: str
    execution_timeout: int = 10


def load_config(path: str, model: type[BaseModel]) -> BaseModel:
    import yaml
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))
