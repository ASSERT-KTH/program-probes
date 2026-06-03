import argparse
import json
from src.configs import ModelConfig, load_config
from src.probe import run_sweep, run_final, run_eval, create_sweep


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

    create_p = subparsers.add_parser("create-sweep", help="Create a W&B sweep and print its ID (no training)")
    create_p.add_argument("--fixed", type=json.loads, default=None,
                          help='JSON dict of fixed HP values')

    sweep_p = subparsers.add_parser("sweep")
    sweep_p.add_argument("--sweep-id", default=None, help="Join an existing W&B sweep instead of creating a new one")
    sweep_p.add_argument("--count", type=int, default=None, help="Max number of runs this agent will execute")
    sweep_p.add_argument("--then-final", action="store_true",
                        help="After sweep, automatically train on all layers with best HPs")
    sweep_p.add_argument("--fixed", type=json.loads, default=None,
                        help='JSON dict of fixed HP values, e.g. \'{"lr": 0.001, "loss": "weighted_cross_entropy"}\'')

    final_p = subparsers.add_parser("final")
    final_p.add_argument("--lr", type=float, default=None)
    final_p.add_argument("--weight-decay", type=float, default=None)
    final_p.add_argument("--batch-size", type=int, default=None)
    final_p.add_argument("--patience", type=int, default=None)
    final_p.add_argument("--from-sweep", default=None,
                        help="Sweep ID to load best hyperparameters from")
    final_p.add_argument("--loss", choices=["cross_entropy", "weighted_cross_entropy"],
                        default="cross_entropy")
    final_p.add_argument("--pos-weight", type=float, default=1.0,
                        help="Positive class weight multiplier (only used with --loss weighted_cross_entropy)")

    eval_p = subparsers.add_parser("eval", help="Evaluate pre-trained weights on n_eval_bins bins (no retraining)")
    eval_p.add_argument("--weights-run-id", default=None,
                        help="Run ID whose weights.pt to load (defaults to --run-id)")
    eval_p.add_argument("--output-run-id", required=True,
                        help="Run ID under which to save results.pt")

    args = parser.parse_args()
    if args.n_eval_bins is not None and args.n_eval_bins != args.n_bins and args.n_bins != 1:
        parser.error("--n-eval-bins requires --n-bins 1 or --n-eval-bins equal to --n-bins")
    model_cfg = load_config(args.model_config, ModelConfig)
    cache_run_id = args.cache_run_id or args.run_id

    if args.mode == "create-sweep":
        create_sweep(
            run_id=args.run_id,
            probe_name=args.probe,
            fixed_params=args.fixed,
        )
    elif args.mode == "sweep":
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
            then_final=args.then_final,
            results_dir=args.results_dir,
            fixed_params=args.fixed,
            shuffle_labels=args.shuffle_labels,
        )
    elif args.mode == "eval":
        run_eval(
            weights_run_id=args.weights_run_id or args.run_id,
            output_run_id=args.output_run_id,
            probe_name=args.probe,
            probe_layers=model_cfg.probe_layers,
            seed=args.seed,
            cache_dir=args.cache_dir,
            cache_run_id=cache_run_id,
            results_dir=args.results_dir,
            probe_arch=args.probe_arch,
            n_eval_bins=args.n_eval_bins or 10,
            eval_bin_axis=args.eval_bin_axis,
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
            sweep_id=args.from_sweep,
            cache_dir=args.cache_dir,
            cache_run_id=cache_run_id,
            results_dir=args.results_dir,
            n_bins=args.n_bins,
            probe_arch=args.probe_arch,
            loss=args.loss,
            pos_weight=args.pos_weight,
            n_eval_bins=args.n_eval_bins,
            eval_bin_axis=args.eval_bin_axis,
            shuffle_labels=args.shuffle_labels,
        )


if __name__ == "__main__":
    main()
