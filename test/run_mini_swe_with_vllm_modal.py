"""Start vLLM and run mini-SWE-agent with bash commands in Modal.

Run inside a Berzelius GPU allocation after repository setup and Modal auth:

    export MODAL_TOKEN_ID="..."
    export MODAL_TOKEN_SECRET="..."

    uv run --python 3.12 python test/run_mini_swe_with_vllm_modal.py \
      --model-config configs/models/qwen3_8b.yaml \
      --generation-config configs/generation.yaml \
      --vllm-config configs/agents/vllm_launch.yaml \
      --mini-swe-config configs/agents/mini_swe_local.yaml

The agent controller and vLLM server run on Berzelius. Only bash commands are
sent to the Modal sandbox through ModalSandboxEnvironment.
"""

import argparse

from src.agents.mini_swe import MiniSweAgentAdapter
from src.agents.modal_environment import ModalSandboxEnvironment
from src.agents.vllm_server import VllmServer
from src.configs import (
    GenerationConfig,
    MiniSweAgentConfig,
    ModelConfig,
    VllmServerConfig,
    load_config,
)


DEFAULT_TASK = """
Use exactly one bash command to submit the final output.
The command should print COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on the first line.
Then print this string on the second line:

hello from mini-swe-agent via vllm and modal
""".strip()


def _load_modal_image(image_ref: str | None):
    if image_ref is None:
        return None

    try:
        import modal
    except ImportError as exc:
        raise ImportError("modal is required when --modal-image is provided.") from exc

    return modal.Image.from_registry(image_ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3_8b.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--vllm-config", default="configs/agents/vllm_launch.yaml")
    parser.add_argument("--mini-swe-config", default="configs/agents/mini_swe_local.yaml")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--modal-app-name", default="program-probes-agent")
    parser.add_argument("--modal-timeout", type=int, default=60)
    parser.add_argument("--trajectory-output", default="outputs/agent_trajectories/modal_smoke.json")
    parser.add_argument(
        "--modal-image",
        default=None,
        help="Optional registry image, e.g. python:3.12-slim.",
    )
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    gen_cfg = load_config(args.generation_config, GenerationConfig)
    vllm_cfg = load_config(args.vllm_config, VllmServerConfig)
    mini_cfg = load_config(args.mini_swe_config, MiniSweAgentConfig)

    modal_env = ModalSandboxEnvironment(
        app_name=args.modal_app_name,
        image=_load_modal_image(args.modal_image),
        timeout=args.modal_timeout,
    )

    print("Starting vLLM server...")
    with VllmServer(model_cfg, vllm_cfg) as server, modal_env:
        print(f"vLLM ready at {server.base_url} with served model {server.model_name!r}")

        print("Running direct Modal sandbox smoke command...")
        smoke = modal_env.execute(
            {"command": "echo modal sandbox ready && pwd && uname -a"},
        )
        print(smoke)

        agent = MiniSweAgentAdapter(
            model_name=server.model_name,
            base_url=server.base_url,
            api_key=vllm_cfg.api_key,
            agent_config=mini_cfg,
            generation_config=gen_cfg,
            environment=modal_env,
        )

        print("Running mini-SWE-agent task with Modal-backed bash execution...")
        result = agent.run_task(args.task)
        print("mini-SWE-agent result:")
        print(result)
        print("Modal bash command history:")
        for idx, item in enumerate(modal_env.command_history, start=1):
            print(f"{idx}. returncode={item['returncode']} command={item['command']!r}")
        trajectory_path = agent.save_trajectory(
            args.trajectory_output,
            tokenizer_name=model_cfg.model_id,
            metadata={
                "model_config": args.model_config,
                "generation_config": args.generation_config,
                "vllm_config": args.vllm_config,
                "mini_swe_config": args.mini_swe_config,
                "modal_app_name": args.modal_app_name,
                "modal_image": args.modal_image,
            },
        )
        print(f"Saved trajectory to {trajectory_path}")

    print("Modal sandbox closed and vLLM server stopped.")


if __name__ == "__main__":
    main()
