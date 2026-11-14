"""Brief HP: provenance of the rerun's probe hyperparameters (metadata only, no training).

Sources:
  sacct submit lines since --since (slurm/probe_sweep_coordinator.sh, probe_sweep.sh, probe_final.sh jobs);
  logs/probe_sweep_coordinator_<job>.out (sweep id, count per agent, worker jobs);
  logs/probe_final_<job>.out ("Best sweep run" and the config final training used);
  W&B (assert-kth/program-probes): every sweep's runs, states, search space, metric and best run.

Checks (raise on failure): every sweep has n_agents x count_per_agent == count trials, all W&B runs finished;
every completed final job's --from-sweep is one of these sweeps, for the same benchmark, model, probe and
layer; the config each final job logged equals its sweep's best W&B config.
Writes paper_rebuttal_candidate/hparams/json/{sweeps.json, final_jobs.json}.

Usage: uv run python scripts/rebuttal_hparam_provenance.py [--since 2026-09-20] [--sacct-dump file]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/hparams/json")
KEYS = ("lr", "weight_decay", "batch_size", "patience")
JOB_FIELDS = ("job", "results", "run", "probe", "layer", "sub", "source", "sweep", "weights", "after_edit")


def sacct_lines(args) -> list[list[str]]:
    if args.sacct_dump:
        text = Path(args.sacct_dump).read_text()
    else:
        text = subprocess.run(["sacct", "-u", args.user, "-S", args.since, "-X", "-n", "-P",
                               "-o", "JobID,JobName,State,Submit,SubmitLine%3000"],
                              capture_output=True, text=True, check=True).stdout
    return [p for p in (ln.split("|", 4) for ln in text.splitlines()) if len(p) == 5]


def opts(cmd: str) -> dict:
    toks = cmd.split()
    get = lambda k: toks[toks.index(k) + 1] if k in toks else None  # noqa: E731
    return {"run": get("--run-id"), "cache": get("--cache-run-id"), "results": get("--results-dir"),
            "probe": get("--probe"), "layer": get("--layer"), "sweep": get("--from-sweep"), "sweep_id": get("--sweep-id"),
            "count": get("--count"), "n_agents": get("--n-agents"), "lr": get("--lr"),
            "sub": "final" if " final" in cmd else ("eval" if " eval" in cmd else "?"),
            "shuffle": "--shuffle-labels" in toks, "shuffle_unit": get("--shuffle-unit"),
            "weights_run": get("--weights-run-id"), "weights_results": get("--weights-results-dir") or get("--results-dir"),
            "output_run": get("--output-run-id"), "after_edit": "--after-edit-only" in toks}


def logged_config(jid: str) -> dict:
    t = Path(f"logs/probe_final_{jid}.out").read_text()
    m = re.search(r"Best sweep run: (\S+)\s+mean_val_auc=(\S+)", t)
    c = dict(re.findall(r"^\s+(lr|patience|batch_size|weight_decay): (\S+)", t, re.M))
    return {"best_run": m.group(1) if m else None, **{k: float(v) for k, v in c.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-20")
    ap.add_argument("--user", default="x_andaf")
    ap.add_argument("--sacct-dump")
    args = ap.parse_args()
    lines = sacct_lines(args)

    sweeps, workers, finals = {}, collections.defaultdict(list), []
    for jid, name, state, submit, cmd in lines:
        o = opts(cmd)
        if name == "pp-probe-coord":
            log = Path(f"logs/probe_sweep_coordinator_{jid}.out").read_text()
            sid = re.search(r"Sweep ID: (\w+)", log).group(1)
            sweeps[sid] = {"coordinator_job": jid, "submit": submit, "results": o["results"], "run": o["run"],
                           "cache": o["cache"], "probe": o["probe"], "layer": int(o["layer"]),
                           "n_agents": int(o["n_agents"] or 4), "count": int(o["count"] or 20),
                           "count_per_agent": int(re.search(r"count_per_agent=(\d+)", log).group(1))}
        elif name == "pp-probe-sweep":
            workers[o["sweep_id"]].append({"job": jid, "state": state, "count": int(o["count"])})
        elif name == "pp-probe-final":
            finals.append({"job": jid, "state": state, "submit": submit, **o})

    import wandb
    api = wandb.Api(timeout=60)
    for sid, s in sweeps.items():
        s["workers"] = workers[sid]
        if s["n_agents"] * s["count_per_agent"] != s["count"] or sum(w["count"] for w in s["workers"]) != s["count"] \
                or any(w["state"] != "COMPLETED" for w in s["workers"]):
            raise ValueError(f"{sid}: workers {s['workers']} do not add up to {s['count']} trials")
        sw = api.sweep(f"assert-kth/program-probes/{sid}")
        states = collections.Counter(r.state for r in sw.runs)
        best = sw.best_run()
        s.update({"wandb_n_runs": len(sw.runs), "wandb_states": dict(states), "method": sw.config.get("method"),
                  "metric": sw.config.get("metric"), "search_space": sw.config.get("parameters"),
                  "best_run": best.name, "best_mean_val_auc": best.summary.get("mean_val_auc"),
                  "config": {k: best.config.get(k) for k in KEYS},
                  "config_extra_keys": sorted(set(best.config) - set(KEYS)),
                  "trials": sorted(({**{k: r.config.get(k) for k in KEYS}, "mean_val_auc": r.summary.get("mean_val_auc"),
                                     "name": r.name} for r in sw.runs),
                                   key=lambda t: -(t["mean_val_auc"] if t["mean_val_auc"] is not None else -1))})
        if s["wandb_n_runs"] != s["count"] or states.get("finished") != s["count"]:
            raise ValueError(f"{sid}: W&B has {len(sw.runs)} runs {dict(states)}, expected {s['count']} finished")
        print(sid, s["results"], s["run"], s["probe"], s["layer"], s["config"], flush=True)

    # Latest completed final job per (results dir, run id, probe, layer, subcommand): what the results.pt holds.
    latest = {}
    for f in sorted(finals, key=lambda f: f["submit"]):
        if f["state"] == "COMPLETED":
            if f["sub"] == "eval":
                f = {**f, "run": f["output_run"]}
            latest[(f["results"], f["run"], f["probe"], f["layer"], f["sub"])] = f
    rows, kinds = [], collections.Counter()
    for f in latest.values():
        row = {k: f[k] for k in ("job", "submit", "results", "run", "probe", "layer", "sub", "shuffle", "shuffle_unit")}
        if f["sub"] == "eval":
            row.update(weights=f"{f['weights_results']}/{f['weights_run']}", after_edit=f["after_edit"])
        if f["sweep"]:
            s = sweeps.get(f["sweep"])
            if s is None:
                row["source"] = "sweep not in the rerun's task-split set"
            else:
                same = (s["results"] == f["results"] and s["probe"] == f["probe"] and s["layer"] == int(f["layer"])
                        and s["run"].split("_pooled")[0] == f["run"].split("_pooled")[0])
                cfg = logged_config(f["job"])
                if not same or any(abs(cfg[k] - float(s["config"][k])) > 1e-12 * max(1.0, abs(cfg[k])) for k in KEYS):
                    raise ValueError(f"final {f['job']} {f['run']}/{f['probe']}/L{f['layer']}: sweep {f['sweep']} or config mismatch")
                row["source"] = "task-split sweep of the same benchmark/model/probe/layer"
            row["sweep"] = f["sweep"]
        elif f["lr"]:
            row["source"] = "typed-in hyperparameters"
        elif f["sub"] == "eval":
            row["source"] = "eval only (no training)"
            trained = [g for g in latest.values() if g["sub"] == "final" and g["results"] == f["weights_results"]
                       and g["run"] == f["weights_run"] and g["probe"] == f["probe"]]
            srcs = {("sweep" if g["sweep"] in sweeps else "typed" if g["lr"] else "other") for g in trained}
            row["weights_trained_with"] = sorted(srcs)
            row["source"] += " of weights trained with " + ("/".join(sorted(srcs)) or "NO FINAL JOB FOUND")
        else:
            row["source"] = "?"
        kinds[(f["results"], re.sub(r"shift\d+_", "shift{k}_", f["run"] or "?"), row["source"])] += 1
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweeps.json").write_text(json.dumps(sweeps, indent=1))
    (OUT / "final_jobs.json").write_text(json.dumps(
        {"by_run_family": [{"results": k[0], "run": k[1], "source": k[2], "n_jobs": n} for k, n in sorted(kinds.items())],
         "job_fields": JOB_FIELDS, "jobs": [[r.get(k) for k in JOB_FIELDS] for r in rows]},
        separators=(",", ":")).replace('],[', '],\n['))
    for k, n in sorted(kinds.items()):
        print(n, *k)
    print("[done]")


if __name__ == "__main__":
    main()
