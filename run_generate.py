import argparse
from src.configs import ModelConfig, TaskConfig, GenerationConfig, load_config
from src.generate import run_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generations-dir", default="generations")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_generation(
        model_config=load_config(args.model_config, ModelConfig),
        task_config=load_config(args.task_config, TaskConfig),
        gen_config=load_config(args.generation_config, GenerationConfig),
        run_id=args.run_id,
        generations_dir=args.generations_dir,
        shard_rank=args.shard_rank,
        num_shards=args.num_shards,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
