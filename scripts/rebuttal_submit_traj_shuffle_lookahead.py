"""Submit trajectory-shuffle lookahead probes (rebuttal E16), one per (run, probe, k).

For each real lookahead probe <pooled>_shift{k}_max{K}, take the layer with the best
VALIDATION AUC, find the submit line of the job that trained that (run, probe, layer)
on the task split (latest COMPLETED `probe_final.sh ... final` line in sacct, submitted
after the split change), and resubmit it unchanged except for
  --run-id <pooled>_shift{k}_max{K}_shuffled_traj
  --shuffle-labels --shuffle-unit trajectory
so hyperparameters, cache and flags mirror the real probe.

Usage (login node: only reads results.pt and sacct output, then calls sbatch):
    sacct -u $USER -S 2026-09-20 --format=JobID%20,State%12,Submit%20,SubmitLine%800 -n -X -P \
        | grep probe_final.sh | grep ' final ' > tmp/final_submitlines.txt
    uv run python scripts/rebuttal_submit_traj_shuffle_lookahead.py --submitlines tmp/final_submitlines.txt [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import torch

COMBOS = [
    ("swebench", "laguna_xs2_full_pooled"),
    ("swebench", "qwen36_35b_a3b_full_pooled"),
    ("swebench_pro", "laguna_xs2_pro_full_pooled"),
    ("swebench_pro", "qwen36_35b_a3b_pro_full_pooled"),
]
PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
WINDOWS = {15: [1, 2, 3, 5, 10, 15], 50: [1, 2, 3, 5, 10, 15, 25, 50]}
SPLIT_CUTOFF = "2026-09-29T10:47:21"  # ff6b081: always split by task


def val_best_layer(path: Path) -> int:
    res = torch.load(path, weights_only=False)
    return max((L for L in res if res[L]), key=lambda L: res[L][0].val_auc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submitlines", required=True)
    ap.add_argument("--account", default="berzelius-2026-259")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="reports/rerun/traj_shuffle_lookahead_jobs.json")
    args = ap.parse_args()

    lines: dict[tuple, list] = {}
    for row in Path(args.submitlines).read_text().splitlines():
        jobid, state, submit, cmd = row.split("|", 3)
        if state != "COMPLETED" or submit < SPLIT_CUTOFF or "--shuffle-labels" in cmd:
            continue
        m = re.search(r"--layer (\d+) --probe (\S+) --run-id (\S+) ", cmd)
        if m:
            lines.setdefault((m.group(3), m.group(2), int(m.group(1))), []).append((submit, cmd))

    manifest = []
    for bench, pooled in COMBOS:
        for probe in PROBES:
            for K, ks in WINDOWS.items():
                for k in ks:
                    rid = f"{pooled}_shift{k}_max{K}"
                    out_rid = f"{rid}_shuffled_traj"
                    layer = val_best_layer(Path("results", bench, rid, probe, "results.pt"))
                    cands = sorted(lines.get((rid, probe, layer), []))
                    if not cands:
                        raise SystemExit(f"no task-split submit line for {rid} {probe} L{layer}")
                    sweeps = {re.search(r"--from-sweep (\S+)", c).group(1) for _, c in cands}
                    if len(sweeps) > 1:
                        raise SystemExit(f"ambiguous sweeps for {rid} {probe} L{layer}: {sweeps}")
                    if Path("results", bench, out_rid, probe, "results.pt").exists():
                        res = torch.load(Path("results", bench, out_rid, probe, "results.pt"), weights_only=False)
                        if layer in res:
                            manifest.append({"bench": bench, "run_id": out_rid, "probe": probe, "layer": layer, "status": "exists"})
                            continue
                    toks = shlex.split(cands[-1][1])
                    toks = [t for t in toks if not t.startswith("--dependency")]
                    toks[toks.index("--run-id") + 1] = out_rid
                    fi = toks.index("final")
                    toks[fi:fi] = ["--shuffle-labels", "--shuffle-unit", "trajectory"]
                    si = toks.index("sbatch") + 1
                    toks[si:si] = ["-A", args.account, "--exclude=node226", "-t", "04:00:00"]
                    entry = {"bench": bench, "run_id": out_rid, "probe": probe, "layer": layer, "cmd": shlex.join(toks)}
                    if not args.dry_run:
                        r = subprocess.run(toks, capture_output=True, text=True, check=True)
                        entry["job_id"] = r.stdout.strip()
                    manifest.append(entry)
                    print(entry.get("job_id", "dry"), out_rid, probe, f"L{layer}", flush=True)
                    if args.dry_run and len(manifest) == 1:
                        print(entry["cmd"])
    if not args.dry_run:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(len(manifest), "entries")


if __name__ == "__main__":
    main()
