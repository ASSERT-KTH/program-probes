import argparse
from src.build_cache import build_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    build_cache(
        run_id=args.run_id,
        probe_names=args.probe,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
