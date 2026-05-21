"""Export SWE-bench probe results to the dashboard data directory.

Usage:
    uv run python run_export_swebench_dashboard.py \\
        --run-id qwen36_27b_full_labeled \\
        --probe will_resolve currently_compiles_swe currently_correct_swe \\
        --model-config configs/models/qwen36_27b.yaml \\
        --output-dir outputs/swebench \\
        --results-dir results/swebench \\
        --cache-dir cache/swebench
"""
import argparse
from src.configs import ModelConfig, load_config
from src.export_swebench_dashboard import export_swebench_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", default="outputs/swebench")
    parser.add_argument("--results-dir", default="results/swebench")
    parser.add_argument("--cache-dir", default="cache/swebench")
    parser.add_argument("--dashboard-dir", default="dashboard")
    parser.add_argument("--traj-dir", default=None,
                        help="Directory of trajectory JSONs; if provided, loads messages and accurate token/turn counts")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)

    export_swebench_dashboard(
        run_id=args.run_id,
        probe_names=args.probe,
        probe_layers=model_cfg.probe_layers,
        model_name=model_cfg.model_id,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        dashboard_dir=args.dashboard_dir,
        traj_dir=args.traj_dir,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
