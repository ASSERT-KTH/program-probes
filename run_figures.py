import argparse
from src.configs import ModelConfig, load_config
from src.figures import plot_probe_heatmap, plot_lookahead_horizon, plot_tool_nll_correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--n-bins", type=int, default=10)
    # Lookahead horizon plot: pass shift run-ids and corresponding k values
    parser.add_argument("--lookahead-shift-run-ids", nargs="+", default=None,
                        help="Run IDs for label-shifted probes (k=1, 2, 3, ...). "
                             "Must match --lookahead-k-values in order.")
    parser.add_argument("--lookahead-k-values", nargs="+", type=int, default=None,
                        help="k values corresponding to --lookahead-shift-run-ids.")
    parser.add_argument("--lookahead-probes", nargs="+", default=None,
                        help="Subset of --probe to generate lookahead figures for. "
                             "Defaults to all probes. Exclude static-label probes (e.g. will_resolve).")
    parser.add_argument("--lookahead-filename-suffix", default="",
                        help="Suffix appended to the output filename, e.g. 'max50' → probe_lookahead_max50.png")
    parser.add_argument("--tool-nll-run-id", default=None,
                        help="Run ID whose nll_corr.pt to load for the tool NLL correlation figure.")
    parser.add_argument("--tool-nll-filename-suffix", default="",
                        help="Suffix for the tool NLL figure filename.")
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

        lookahead_probes = args.lookahead_probes or args.probe
        if args.tool_nll_run_id:
            plot_tool_nll_correlation(
                run_id=args.tool_nll_run_id,
                probe_name=probe_name,
                probe_layers=model_cfg.probe_layers,
                results_dir=args.results_dir,
                figures_dir=args.figures_dir,
                filename_suffix=args.tool_nll_filename_suffix,
            )

        if args.lookahead_shift_run_ids and args.lookahead_k_values and probe_name in lookahead_probes:
            if len(args.lookahead_shift_run_ids) != len(args.lookahead_k_values):
                raise ValueError("--lookahead-shift-run-ids and --lookahead-k-values must have the same length")
            plot_lookahead_horizon(
                base_run_id=args.run_id,
                shift_run_ids=args.lookahead_shift_run_ids,
                k_values=args.lookahead_k_values,
                probe_name=probe_name,
                probe_layers=model_cfg.probe_layers,
                results_dir=args.results_dir,
                figures_dir=args.figures_dir,
                filename_suffix=args.lookahead_filename_suffix,
            )


if __name__ == "__main__":
    main()
