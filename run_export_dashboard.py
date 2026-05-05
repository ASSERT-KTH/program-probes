import argparse
from src.configs import ModelConfig, TaskConfig, load_config
from src.export_dashboard import export_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dashboard-dir", default="dashboard")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    task_cfg = load_config(args.task_config, TaskConfig)

    export_dashboard(
        run_id=args.run_id,
        probe_names=args.probe,
        probe_layers=model_cfg.probe_layers,
        task_name=task_cfg.dataset,
        model_name=model_cfg.model_id,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        dashboard_dir=args.dashboard_dir,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
