import argparse
from src.build_cache import build_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--cache-run-id", default=None,
                        help="Override the cache output directory name (default: same as --run-id).")
    parser.add_argument("--label-shift", type=int, default=0,
                        help="Shift labels forward by this many assistant turns. "
                             "Token at turn s gets the label of turn s+k. "
                             "Tokens in the last k turns are dropped.")
    args = parser.parse_args()

    build_cache(
        run_id=args.run_id,
        probe_names=args.probe,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        label_shift=args.label_shift,
        cache_run_id=args.cache_run_id,
    )


if __name__ == "__main__":
    main()
