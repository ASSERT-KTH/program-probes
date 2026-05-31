import argparse
from src.configs import ModelConfig, load_config
from src.probe import run_sweep, run_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--cache-run-id", default=None,
                        help="Run ID to use for cache lookup (defaults to --run-id)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-eval-bins", type=int, default=None,
                        help="Evaluate at this bin granularity; requires --n-bins 1 or --n-eval-bins equal to --n-bins")
    parser.add_argument("--eval-bin-axis", choices=["position", "step_relative", "step_absolute"],
                        default="position",
                        help="Axis to bin on during evaluation: token position, relative step, or exact step number")
    parser.add_argument("--probe-arch", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Randomly permute labels within each split before training (sanity-check baseline).")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    sweep_p = subparsers.add_parser("sweep")
    sweep_p.add_argument("--sweep-id", default=None, help="Join an existing W&B sweep instead of creating a new one")
    sweep_p.add_argument("--count", type=int, default=None, help="Max number of runs this agent will execute")

    final_p = subparsers.add_parser("final")
    final_p.add_argument("--lr", type=float, required=True)
    final_p.add_argument("--weight-decay", type=float, required=True)
    final_p.add_argument("--batch-size", type=int, required=True)
    final_p.add_argument("--patience", type=int, required=True)

    args = parser.parse_args()
    if args.n_eval_bins is not None and args.n_eval_bins != args.n_bins and args.n_bins != 1:
        parser.error("--n-eval-bins requires --n-bins 1 or --n-eval-bins equal to --n-bins")
    model_cfg = load_config(args.model_config, ModelConfig)
    cache_run_id = args.cache_run_id or args.run_id

    if args.mode == "sweep":
        run_sweep(
            run_id=args.run_id,
            probe_name=args.probe,
            probe_layers=model_cfg.probe_layers,
            seed=args.seed,
            sweep_id=args.sweep_id,
            count=args.count,
            cache_dir=args.cache_dir,
            cache_run_id=cache_run_id,
            n_bins=args.n_bins,
            probe_arch=args.probe_arch,
            n_eval_bins=args.n_eval_bins,
            eval_bin_axis=args.eval_bin_axis,
            shuffle_labels=args.shuffle_labels,
        )
    else:
        run_final(
            run_id=args.run_id,
            probe_name=args.probe,
            probe_layers=model_cfg.probe_layers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
            cache_dir=args.cache_dir,
            cache_run_id=cache_run_id,
            results_dir=args.results_dir,
            n_bins=args.n_bins,
            probe_arch=args.probe_arch,
            n_eval_bins=args.n_eval_bins,
            eval_bin_axis=args.eval_bin_axis,
            shuffle_labels=args.shuffle_labels,
        )


if __name__ == "__main__":
    main()
