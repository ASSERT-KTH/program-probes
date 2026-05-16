from pydantic import BaseModel, Field, field_validator
from typing import Literal


class GenerationConfig(BaseModel):
    # Sampling
    max_new_tokens: int = 1024
    n_generations: int = 10
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    stride: int = 5
    seed: int = 42
    # Compute
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    num_gpus: int = 1
    extraction_batch_size: int = 8
    max_model_len: int | None = None


class ModelConfig(BaseModel):
    model_id: str
    probe_layers: list[int] = []
    adapter: str = ""


class TaskConfig(BaseModel):
    dataset: str
    adapter: str
    execution_timeout: int = 10
    pairs_path: str | None = None


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


class SWEBenchRunConfig(BaseModel):
    """Unified config for a SWE-bench Verified agent run.

    Covers everything needed to launch the vLLM server and run mini-swe-agent
    on SWE-bench instances in Modal sandboxes.
    """

    # --- Model / vLLM ---
    model_id: str
    served_model_name: str | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    gpu_memory_utilization: float = 0.9
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 18000
    vllm_startup_timeout_s: int = 900
    vllm_log_path: str | None = "logs/vllm.log"
    vllm_extra_args: list[str] = Field(default_factory=list)

    # --- Generation ---
    temperature: float = 0.7
    max_new_tokens: int | None = None  # None = no per-call token cap (recommended for thinking models)
    top_p: float | None = None

    # --- Agent ---
    step_limit: int = 30
    command_timeout: int = 180  # seconds per bash command inside sandbox

    # --- Execution ---
    n_runs: int = 1             # times to run the agent per instance (pass@k)
    n_instances: int = -1       # instances to process per shard (-1 = all)
    n_workers: int = 4          # parallel agent threads sharing the vLLM server
    modal_app_name: str = "program-probes-swebench"
    modal_timeout: int = 300    # seconds for the Modal sandbox lifecycle
    output_dir: str = "generations/swebench"

    def to_model_config(self) -> "ModelConfig":
        return ModelConfig(
            model_id=self.model_id,
            probe_layers=[],
            adapter="",
        )

    def to_vllm_server_config(self) -> "VllmServerConfig":
        extra_args = [
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            *self.vllm_extra_args,
        ]
        return VllmServerConfig(
            host=self.vllm_host,
            port=self.vllm_port,
            served_model_name=self.served_model_name or self.model_id,
            startup_timeout_s=self.vllm_startup_timeout_s,
            extra_args=extra_args,
            log_path=self.vllm_log_path,
        )

    def to_generation_config(self) -> "GenerationConfig":
        kwargs: dict = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
        )
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        return GenerationConfig(**kwargs)

    def to_agent_config(self) -> "MiniSweAgentConfig":
        import yaml as _yaml
        from pathlib import Path as _Path
        _swe_cfg_path = _Path(__file__).parent.parent / "configs/agents/mini_swe_swebench.yaml"
        if _swe_cfg_path.exists():
            base = _yaml.safe_load(_swe_cfg_path.read_text())
        else:
            base = {}
        base["step_limit"] = self.step_limit
        base["timeout"] = self.command_timeout
        return MiniSweAgentConfig.model_validate(base)


def load_config(path: str, model: type[BaseModel]) -> BaseModel:
    import yaml
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))
