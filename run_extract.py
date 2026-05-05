import argparse
from src.configs import ModelConfig, TaskConfig, HardwareConfig, GenerationConfig, load_config
from src.extract import run_extraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    task_cfg = load_config(args.task_config, TaskConfig)
    hw_cfg = load_config(args.hardware_config, HardwareConfig)
    gen_cfg = load_config(args.generation_config, GenerationConfig)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        hardware_config=hw_cfg,
        generation_config=gen_cfg,
        probe_names=args.probe,
        run_id=args.run_id,
        output_dir=args.output_dir,
        shard_rank=args.shard_rank,
        num_shards=args.num_shards,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
