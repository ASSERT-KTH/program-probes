import argparse
from src.configs import ModelConfig, TaskConfig, GenerationConfig, load_config
from src.extract import run_extraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generations-dir", default="generations")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    run_extraction(
        model_config=load_config(args.model_config, ModelConfig),
        task_config=load_config(args.task_config, TaskConfig),
        gen_config=load_config(args.generation_config, GenerationConfig),
        probe_names=args.probe,
        run_id=args.run_id,
        generations_dir=args.generations_dir,
        output_dir=args.output_dir,
        shard_rank=args.shard_rank,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
