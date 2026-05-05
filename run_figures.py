import argparse
from src.configs import ModelConfig, load_config
from src.figures import plot_probe_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)

    for probe_name in args.probe:
        plot_probe_heatmap(
            run_id=args.run_id,
            probe_name=probe_name,
            probe_layers=model_cfg.probe_layers,
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
            n_bins=args.n_bins,
        )


if __name__ == "__main__":
    main()
