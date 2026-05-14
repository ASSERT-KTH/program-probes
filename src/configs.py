from pydantic import BaseModel, Field, field_validator
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


class VllmServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str | None = None
    api_key: str = "EMPTY"
    startup_timeout_s: int = 900
    healthcheck_interval_s: float = 2.0
    extra_args: list[str] = Field(default_factory=list)
    log_path: str | None = None

    @field_validator("port")
    @classmethod
    def valid_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class MiniSweAgentConfig(BaseModel):
    step_limit: int = 250
    cost_limit: float = 0.0
    cwd: str = "/testbed"
    timeout: int = 60
    environment_class: Literal["local", "docker", "modal"] = "local"
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    system_template: str = (
        "You are a helpful assistant that can interact with a computer.\n\n"
        "Every response must contain exactly one bash command in this format:\n\n"
        "```mswea_bash_command\n"
        "your_command_here\n"
        "```"
    )
    instance_template: str = (
        "Please solve this task: {{task}}\n\n"
        "Use bash commands to execute actions.\n"
        "When the task is complete, submit by running a bash command whose first output line is:\n"
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n\n"
        "Example:\n"
        "```mswea_bash_command\n"
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo final answer here\n"
        "```"
    )
    extra_model_kwargs: dict = Field(default_factory=dict)


def load_config(path: str, model: type[BaseModel]) -> BaseModel:
    import yaml
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))
