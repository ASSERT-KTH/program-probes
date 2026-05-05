import argparse
from src.configs import ModelConfig, load_config
from src.probe import run_sweep, run_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=10)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    sweep_p = subparsers.add_parser("sweep")
    sweep_p.add_argument("--sweep-id", default=None)

    final_p = subparsers.add_parser("final")
    final_p.add_argument("--lr", type=float, required=True)
    final_p.add_argument("--weight-decay", type=float, required=True)
    final_p.add_argument("--batch-size", type=int, required=True)
    final_p.add_argument("--patience", type=int, required=True)

    args = parser.parse_args()
    model_cfg = load_config(args.model_config, ModelConfig)

    if args.mode == "sweep":
        run_sweep(
            run_id=args.run_id,
            probe_name=args.probe,
            probe_layers=model_cfg.probe_layers,
            seed=args.seed,
            cache_dir=args.cache_dir,
            n_bins=args.n_bins,
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
            results_dir=args.results_dir,
            n_bins=args.n_bins,
        )


if __name__ == "__main__":
    main()
