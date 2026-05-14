import argparse
import time
import urllib.request

from src.agents.vllm_server import VllmServer
from src.configs import ModelConfig, VllmServerConfig, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--vllm-config", required=True)
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="Keep the server alive after the /v1/models check.",
    )
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    vllm_cfg = load_config(args.vllm_config, VllmServerConfig)

    with VllmServer(model_cfg, vllm_cfg) as server:
        models_url = f"{server.base_url}/models"
        print(f"vLLM server is ready: {server.base_url}")
        with urllib.request.urlopen(models_url, timeout=10) as response:
            print(response.read().decode())

        if args.hold_seconds > 0:
            print(f"Holding server for {args.hold_seconds}s before shutdown...")
            time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
