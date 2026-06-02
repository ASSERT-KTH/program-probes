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
                             "Token at turn s gets the label of turn s+k.")
    parser.add_argument("--max-label-shift", type=int, default=None,
                        help="Fix the evaluation window to turns where turn + max_label_shift < n_turns. "
                             "Use this to ensure all k values are evaluated on the same token set. "
                             "Must be >= --label-shift.")
    parser.add_argument("--group-by", choices=["instance", "project"], default="project",
                        help="Grouping unit for train/val/test splits. "
                             "'instance' (default) splits by individual trajectory; "
                             "'project' groups all instances from the same repo together.")
    args = parser.parse_args()

    build_cache(
        run_id=args.run_id,
        probe_names=args.probe,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        label_shift=args.label_shift,
        max_label_shift=args.max_label_shift,
        cache_run_id=args.cache_run_id,
        group_by=args.group_by,
    )


if __name__ == "__main__":
    main()
