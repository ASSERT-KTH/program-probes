"""Start vLLM, connect mini-SWE-agent to it, and run a small task.

Run inside a Berzelius GPU allocation after repository setup:

    uv run --python 3.12 python test/run_mini_swe_with_vllm.py \
      --model-config configs/models/qwen3_8b.yaml \
      --generation-config configs/generation.yaml \
      --vllm-config configs/agents/vllm_launch.yaml \
      --mini-swe-config configs/agents/mini_swe_local.yaml

The script owns the vLLM subprocess. When the script exits, it terminates vLLM.
"""

import argparse

from src.agents.mini_swe import MiniSweAgentAdapter
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

hello from mini-swe-agent via vllm
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3_8b.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--vllm-config", default="configs/agents/vllm_launch.yaml")
    parser.add_argument("--mini-swe-config", default="configs/agents/mini_swe_local.yaml")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--trajectory-output", default="outputs/agent_trajectories/local_smoke.json")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    gen_cfg = load_config(args.generation_config, GenerationConfig)
    vllm_cfg = load_config(args.vllm_config, VllmServerConfig)
    mini_cfg = load_config(args.mini_swe_config, MiniSweAgentConfig)

    print("Starting vLLM server...")
    with VllmServer(model_cfg, vllm_cfg) as server:
        print(f"vLLM ready at {server.base_url} with served model {server.model_name!r}")

        agent = MiniSweAgentAdapter(
            model_name=server.model_name,
            base_url=server.base_url,
            api_key=vllm_cfg.api_key,
            agent_config=mini_cfg,
            generation_config=gen_cfg,
        )

        print("Running mini-SWE-agent task...")
        result = agent.run_task(args.task)
        print("mini-SWE-agent result:")
        print(result)
        print("Bash command history:")
        for idx, item in enumerate(agent.command_history, start=1):
            print(f"{idx}. returncode={item['returncode']} command={item['command']!r}")
        trajectory_path = agent.save_trajectory(
            args.trajectory_output,
            tokenizer_name=model_cfg.model_id,
            metadata={
                "model_config": args.model_config,
                "generation_config": args.generation_config,
                "vllm_config": args.vllm_config,
                "mini_swe_config": args.mini_swe_config,
            },
        )
        print(f"Saved trajectory to {trajectory_path}")

    print("vLLM server stopped.")


if __name__ == "__main__":
    main()
