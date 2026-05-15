from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.agents.base import AgentAdapter
from src.agents.trajectory import save_agent_trajectory
from src.configs import GenerationConfig, MiniSweAgentConfig
from src.probes.base import TrajectoryContext
from src.tasks.base import ChatPrompt


class LoggingEnvironment:
    """Small proxy that prints and records every bash action."""

    def __init__(self, env: Any, label: str = "bash"):
        self.env = env
        self.label = label
        self.command_history: list[dict[str, Any]] = []

    def execute(self, action: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        command = action.get("command", "")
        if command:
            print(f"[{self.label}] {command}", flush=True)
        entry = {"command": command, "returncode": None, "output": "", "status": "started"}
        self.command_history.append(entry)
        try:
            result = self.env.execute(action, *args, **kwargs)
        except Exception as exc:
            entry["status"] = type(exc).__name__
            raise
        entry.update(
            {
                "returncode": result.get("returncode"),
                "output": result.get("output", ""),
                "diff": result.get("diff", ""),
                "status": "completed",
            }
        )
        return result

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.env.get_template_vars(**kwargs)

    def serialize(self) -> dict[str, Any]:
        return self.env.serialize()


def build_litellm_vllm_model_config(
    *,
    model_name: str,
    base_url: str,
    api_key: str = "EMPTY",
    generation_config: GenerationConfig | None = None,
    extra_model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the mini-SWE-agent/LiteLLM model block for a vLLM server."""
    model_kwargs: dict[str, Any] = {
        "custom_llm_provider": "openai",
        "api_base": base_url.rstrip("/"),
        "api_key": api_key,
        "drop_params": True,
    }
    if generation_config is not None:
        model_kwargs["temperature"] = generation_config.temperature
        model_kwargs["max_tokens"] = generation_config.max_new_tokens
        if generation_config.top_p is not None:
            model_kwargs["top_p"] = generation_config.top_p
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    return {
        "model_name": model_name,
        "model_class": "litellm",
        "model_kwargs": model_kwargs,
    }


def build_mini_swe_config(
    *,
    task: str,
    model_name: str,
    base_url: str,
    api_key: str = "EMPTY",
    agent_config: MiniSweAgentConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Build a serializable mini-SWE-agent config connected to vLLM."""
    agent_config = agent_config or MiniSweAgentConfig()

    agent_block: dict[str, Any] = {
        "step_limit": agent_config.step_limit,
        "cost_limit": agent_config.cost_limit,
        "system_template": agent_config.system_template,
        "instance_template": agent_config.instance_template,
    }

    return {
        "agent": agent_block,
        "environment": {
            "environment_class": agent_config.environment_class,
            "cwd": agent_config.cwd,
            "timeout": agent_config.timeout,
        },
        "model": {
            **build_litellm_vllm_model_config(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                generation_config=generation_config,
                extra_model_kwargs=agent_config.extra_model_kwargs,
            ),
            "action_regex": agent_config.action_regex,
        },
        "run": {"task": task},
    }


def write_mini_swe_config(config: dict[str, Any], path: str | Path) -> Path:
    """Write a mini-SWE-agent YAML config for inspection or CLI use."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return out_path


class MiniSweAgentAdapter(AgentAdapter):
    """AgentAdapter implementation that drives mini-SWE-agent through vLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        agent_config: MiniSweAgentConfig | None = None,
        generation_config: GenerationConfig | None = None,
        environment: Any | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.agent_config = agent_config or MiniSweAgentConfig()
        self.generation_config = generation_config
        self.environment = environment
        self.command_history: list[dict[str, Any]] = []
        self.last_task: str | None = None
        self.last_result: Any | None = None
        self.last_messages: list[dict[str, Any]] = []

    def run_task(self, task: str) -> Any:
        """Run a raw mini-SWE-agent task string."""
        try:
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.environments.local import LocalEnvironment
            from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
        except ImportError as exc:
            raise ImportError(
                "mini-swe-agent is required to run agent episodes. Install it with `uv sync`."
            ) from exc

        model_block = build_litellm_vllm_model_config(
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            generation_config=self.generation_config,
            extra_model_kwargs=self.agent_config.extra_model_kwargs,
        )
        model = LitellmTextbasedModel(
            model_name=model_block["model_name"],
            model_kwargs=model_block["model_kwargs"],
            action_regex=self.agent_config.action_regex,
            cost_tracking="ignore_errors",
        )
        base_env = self.environment or LocalEnvironment(
            cwd=self.agent_config.cwd,
            timeout=self.agent_config.timeout,
        )
        env = LoggingEnvironment(base_env, label=self.agent_config.environment_class)
        agent = DefaultAgent(
            model,
            env,
            system_template=self.agent_config.system_template,
            instance_template=self.agent_config.instance_template,
            step_limit=self.agent_config.step_limit,
            cost_limit=self.agent_config.cost_limit,
        )
        result = agent.run(task)
        self.command_history = list(env.command_history)
        self.last_task = task
        self.last_result = result
        self.last_messages = list(agent.messages)
        return result

    def save_trajectory(
        self,
        path: str | Path,
        *,
        tokenizer_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if self.last_task is None:
            raise RuntimeError("No completed task is available to save. Call run_task() first.")
        return save_agent_trajectory(
            path,
            task=self.last_task,
            result=self.last_result,
            messages=self.last_messages,
            command_history=self.command_history,
            model_name=self.model_name,
            base_url=self.base_url,
            tokenizer_name=tokenizer_name,
            metadata=metadata,
        )

    def run_episode(
        self,
        sample: dict,
        model_adapter,
        task_adapter,
    ) -> TrajectoryContext:
        prompt = task_adapter.format_prompt(sample)
        task = _prompt_to_task(prompt)
        result = self.run_task(task)
        generated_text = _result_to_text(result)
        return TrajectoryContext(
            sample=sample,
            generated_text=generated_text,
            n_captured_steps=0,
            edit_history=[],
        )


def _prompt_to_task(prompt: ChatPrompt) -> str:
    if prompt.assistant_prefill is None:
        return prompt.user_content
    return f"{prompt.user_content}\n\nAssistant prefill:\n{prompt.assistant_prefill}"


def _result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("content", "output", "submission"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    return str(result)
