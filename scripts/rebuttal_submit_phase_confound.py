"""Submit the E2/E16 (lookahead) and E17 (k0) jobs of scripts/rebuttal_phase_confound.py.

Lookahead jobs depend (afterok) on the trajectory-shuffle training jobs listed in
reports/rerun/traj_shuffle_lookahead_jobs.json for the same (run, probe, window).

Usage:
    uv run python scripts/rebuttal_submit_phase_confound.py [--modes lookahead k0] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

RUNS = [  # bench, pooled run id, shift-cache prefix, k=0 cache dir
    ("swebench", "laguna_xs2_full_pooled", "laguna_xs2_full", "laguna_xs2_full"),
    ("swebench", "qwen36_35b_a3b_full_pooled", "qwen36_35b_a3b_full", "qwen36_35b_a3b_full"),
    ("swebench_pro", "laguna_xs2_pro_full_pooled", "laguna_xs2_pro_full", "laguna_xs2_full"),
    ("swebench_pro", "qwen36_35b_a3b_pro_full_pooled", "qwen36_35b_a3b_pro_full", "qwen36_35b_a3b_full"),
]
LOOKAHEAD_PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
K0_PROBES = ["currently_compiles"] + LOOKAHEAD_PROBES
OUT = Path("reports/rerun/phase")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["lookahead", "k0"])
    ap.add_argument("--account", default="berzelius-2026-259")
    ap.add_argument("--manifest", default="reports/rerun/traj_shuffle_lookahead_jobs.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text()) if "lookahead" in args.modes else []
    base = ["sbatch", "--parsable", "-A", args.account, "--exclude=node226"]
    for bench, pooled, prefix, k0dir in RUNS:
        if "k0" in args.modes:
            for probe in K0_PROBES:
                out = OUT / f"k0_{bench}_{pooled}_{probe}.json"
                cmd = base + ["slurm/phase_confound.sh", "k0", "--bench", bench, "--pooled", pooled,
                              "--cache-prefix", k0dir, "--probe", probe, "--output", str(out)]
                jid = "dry" if args.dry_run else subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
                print(jid, out, flush=True)
        if "lookahead" in args.modes:
            for probe in LOOKAHEAD_PROBES:
                for K in (15, 50):
                    deps = [m["job_id"] for m in manifest
                            if m["bench"] == bench and m["probe"] == probe and "job_id" in m
                            and m["run_id"].startswith(f"{pooled}_shift") and m["run_id"].endswith(f"_max{K}_shuffled_traj")]
                    out = OUT / f"lookahead_{bench}_{pooled}_{probe}_max{K}.json"
                    dep = ["--dependency=afterok:" + ":".join(deps)] if deps else []
                    cmd = base + dep + ["slurm/phase_confound.sh", "lookahead", "--bench", bench, "--pooled", pooled,
                                        "--cache-prefix", prefix, "--probe", probe, "--max-k", str(K), "--output", str(out)]
                    jid = "dry" if args.dry_run else subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
                    print(jid, out, f"deps={len(deps)}", flush=True)


if __name__ == "__main__":
    main()
