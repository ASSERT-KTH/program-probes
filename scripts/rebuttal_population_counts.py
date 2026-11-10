"""Brief E14 (reviewer wokJ-Q3): population counts after filtering. Counting only, no probes.

For one run (model x benchmark):

Rerun (task split, real turns), filters k0 (pooled cache, no window), max15, max50 (the shift0 cache of
each lookahead window: tokens with step_idx + K < n_turns), per probe and split (train/val/test):
  tasks / trajectories with >= 1 kept token / hidden states, where kept = y >= 0 (the tokens the probes
  train and test on); cache rows (y any) too; trajectories in the split from the generations
  (gen_stats, one per trajectory JSON, assigned to a split by task id) as the denominator;
  test positive rate (y_t). Checked: the counts equal the stored results.pt n_train/n_val/n_test of the
  pooled runs, and the windowed test counts equal the lookahead figures' "n = ... h_t"
  (reports/rerun/lookahead_refs).

Trajectory length: turns per trajectory from the label files (len(turn_labels), the K of the window
filter) and steps from the generations (len(command_history), as in tab:dataset_stats).

Legacy (submitted) populations, rebuilt from snapshots/ (the legacy k0 caches' y, sample_id, group_id,
step_idx) and the activation files' n_turns, with the legacy build_cache window rule (git d64c02f^):
step_idx = min(i*K/T, K-1), keep step_idx + W < n_turns, y = label of the turn bucket's first position;
legacy split = _split_groups(legacy group_id, 42) (Verified: repository; Pro: instance id without the
version suffix). Checked against the stored n_train/n_val/n_test of the *_legacyturns runs.

Usage: uv run python scripts/rebuttal_population_counts.py --run laguna_verified \
           --output paper_rebuttal_candidate/counts/json/counts_laguna_verified.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.probe import _split_groups  # noqa: E402

PROBES = ["currently_compiles", "currently_correct", "currently_reduces_failing", "currently_has_regressions"]
SPLITS = ["train", "val", "test"]
RUNS = {
    "laguna_verified": ("swebench", "laguna_xs2_full", "laguna_xs2_full", "laguna_xs2_full_pooled"),
    "laguna_pro": ("swebench_pro", "laguna_xs2_full", "laguna_xs2_pro_full", "laguna_xs2_pro_full_pooled"),
    "qwen_verified": ("swebench", "qwen36_35b_a3b_full", "qwen36_35b_a3b_full", "qwen36_35b_a3b_full_pooled"),
    "qwen_pro": ("swebench_pro", "qwen36_35b_a3b_full", "qwen36_35b_a3b_pro_full", "qwen36_35b_a3b_pro_full_pooled"),
}
WINDOWS = [15, 50]


def log(*a) -> None:
    print(*a, flush=True)


def task_of(sid: str) -> str:
    return re.sub(r"_run\d+$", "", sid)


def factorize(xs: list[str]) -> tuple[np.ndarray, list[str]]:
    d: dict[str, int] = {}
    codes = np.fromiter((d.setdefault(x, len(d)) for x in xs), dtype=np.int64, count=len(xs))
    return codes, list(d)


def stored_n(path: Path) -> dict | None:
    if not path.exists():
        return None
    res = torch.load(path, weights_only=False)
    for L in sorted(res):
        if res[L]:
            if len(res[L]) != 1:
                raise ValueError(f"{path}: expected one pooled bin per layer")
            r = res[L][0]
            return {"layer": int(L), "train": int(r.n_train), "val": int(r.n_val), "test": int(r.n_test),
                    "n_pos_test": int(r.n_pos_test)}
    return None


def count(sid_codes, sid_names, gid_codes, gid_names, y, split_of_gid, traj_den, task_key=None) -> dict:
    """Counts per split. split_of_gid: array over gid codes with 0/1/2 (train/val/test), -1 unassigned."""
    split_tok = split_of_gid[gid_codes]
    keep = y >= 0
    out = {}
    tasks_of_sid = None
    if task_key is not None:
        t_codes, t_names = factorize([task_key(s) for s in sid_names])
        tasks_of_sid = t_codes
    for si, sp in enumerate(SPLITS):
        m = split_tok == si
        mk = m & keep
        traj_rows = np.unique(sid_codes[m])
        traj_kept = np.unique(sid_codes[mk])
        if tasks_of_sid is None:
            tasks_rows, tasks_kept = np.unique(gid_codes[m]), np.unique(gid_codes[mk])
        else:
            tasks_rows, tasks_kept = np.unique(tasks_of_sid[traj_rows]), np.unique(tasks_of_sid[traj_kept])
        yk = y[mk]
        out[sp] = {"tasks": int(len(tasks_kept)), "trajectories": int(len(traj_kept)), "states": int(mk.sum()),
                   "tasks_with_rows": int(len(tasks_rows)), "trajectories_with_rows": int(len(traj_rows)),
                   "rows": int(m.sum()), "n_pos": int((yk == 1).sum()),
                   "pos_rate": float((yk == 1).mean()) if len(yk) else float("nan"),
                   "trajectories_in_split": int(traj_den[si]),
                   "frac_trajectories_kept": float(len(traj_kept) / traj_den[si]) if traj_den[si] else float("nan")}
    return out


def load_cache(path: Path, extra: tuple[str, ...] = ()) -> dict:
    d = torch.load(path, weights_only=False, mmap=True)
    out = {"y": d["y"].numpy().copy(), "sample_id": d["sample_id"], "group_id": d["group_id"]}
    for k in extra:
        out[k] = d[k].numpy().copy() if hasattr(d[k], "numpy") else d[k]
    del d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    bench, cache, win, pooled = RUNS[args.run]
    croot, rroot = Path("cache", bench), Path("results", bench)
    odir = Path("outputs", bench, cache)
    model_slug = cache
    gen = json.loads(Path(f"paper/cache/gen_stats__{bench}__{model_slug}.json").read_text())
    out: dict = {"run": args.run, "bench": bench, "cache": cache, "pooled": pooled}

    # ---- Trajectory lengths (label files: turns = K of the window rule; generations: steps).
    turns, sid_steps = {}, {}
    for f in sorted(odir.glob("*_labels.pt")):
        lab = torch.load(f, weights_only=False)
        tl = lab["turn_labels"]
        ks = {len(v) for v in tl.values()}
        if len(ks) != 1:
            raise ValueError(f"{f}: turn_labels lengths differ across probes {ks}")
        turns[f.name[: -len("_labels.pt")]] = ks.pop()
    for g in gen:
        sid_steps[g["file"][: -len(".json")]] = g["steps"]
    log("label files", len(turns), "generations", len(gen))

    def length_stats(v: list[int]) -> dict:
        a = np.asarray(v)
        return {"n": int(len(a)), "median": float(np.median(a)), "mean": float(a.mean()),
                **{f"frac_ge{t}": float((a >= t).mean()) for t in (15, 16, 50, 51)},
                **{f"n_ge{t}": int((a >= t).sum()) for t in (15, 16, 50, 51)}}

    common = sorted(set(turns) & set(sid_steps))
    out["length"] = {"turns_label_files": length_stats(list(turns.values())),
                     "steps_generations": length_stats(list(sid_steps.values())),
                     "n_label_files": len(turns), "n_generations": len(gen),
                     "turns_equal_steps": int(sum(turns[s] == sid_steps[s] for s in common)),
                     "n_common": len(common)}
    log("length", json.dumps(out["length"]))

    # Splits: like src/probe.py::run_final, each cache is split on ITS OWN set of group ids
    # (_split_groups(group_ids, 42): sorted unique ids, shuffled, 70/15/15). A window that drops every
    # trajectory of a task therefore changes the split, so the split is computed per filter.
    gen_stems = [g["file"][: -len(".json")] for g in gen]

    def make_split(gid_names: list[str], traj_group: dict[str, str]) -> tuple[np.ndarray, dict, np.ndarray, int]:
        tr, va, te = _split_groups(gid_names, args.seed)
        sname = {g: (0 if g in tr else 1 if g in va else 2) for g in gid_names}
        den = np.zeros(3, dtype=int)
        unassigned = 0
        for stem in gen_stems:
            sp = sname.get(traj_group.get(stem, "\0"))
            if sp is None:
                unassigned += 1
            else:
                den[sp] += 1
        return np.array([sname[g] for g in gid_names]), sname, den, unassigned

    def finish(r: dict, sname: dict, den: np.ndarray, unassigned: int, test_ref: set | None, task_names) -> None:
        test_groups = {g for g, sp in sname.items() if sp == 2}
        r["split_groups"] = {sp: sum(1 for v in sname.values() if v == i) for i, sp in enumerate(SPLITS)}
        r["generations_unassigned"] = unassigned
        kept_all = sum(r[sp]["trajectories"] for sp in SPLITS)
        r["all"] = {"tasks": sum(r[sp]["tasks"] for sp in SPLITS), "trajectories": kept_all,
                    "states": sum(r[sp]["states"] for sp in SPLITS), "generations": len(gen_stems),
                    "frac_generations_kept": kept_all / len(gen_stems)}
        if test_ref is not None:
            tt = {task_names(g) for g in test_groups}
            r["test_tasks_shared_with_k0_test"] = len(tt & test_ref)
            r["test_tasks_not_in_k0_test"] = len(tt - test_ref)

    refs_dir = Path("reports/rerun/lookahead_refs")
    out["rerun"] = {}
    k0_test_tasks = None
    traj_task = {stem: task_of(stem) for stem in gen_stems}  # rerun: group_id is the task id
    for filt in ["k0"] + [f"max{K}" for K in WINDOWS]:
        cdir = croot / (cache if filt == "k0" else f"{win}_shift0_{filt}")
        rid = pooled if filt == "k0" else f"{pooled}_shift0_{filt}"
        res_f, first_sids = {}, None
        for probe in PROBES:
            c = load_cache(cdir / probe / "layer_0.pt")
            sc, sn = factorize(c["sample_id"])
            gc, gn = factorize(c["group_id"])
            if first_sids is None:
                first_sids = (len(c["y"]), sn)
            sog, sname, den, unas = make_split(gn, traj_task)
            r = count(sc, sn, gc, gn, c["y"], sog, den)
            if filt == "k0" and k0_test_tasks is None:
                k0_test_tasks = {g for g, sp in sname.items() if sp == 2}
            finish(r, sname, den, unas, k0_test_tasks, lambda g: g)
            r["same_rows_as_first_probe"] = (len(c["y"]), sn) == first_sids
            st = stored_n(rroot / rid / probe / "results.pt")
            r["stored"] = st
            if st is None:
                raise FileNotFoundError(f"{rid}/{probe}: no results.pt")
            for sp in SPLITS:
                if st[sp] != r[sp]["states"]:
                    raise ValueError(f"{rid}/{probe} {sp}: counted {r[sp]['states']} != stored n_{sp} {st[sp]}")
            if st["n_pos_test"] != r["test"]["n_pos"]:
                raise ValueError(f"{rid}/{probe}: n_pos_test {r['test']['n_pos']} != stored {st['n_pos_test']}")
            if filt != "k0":
                lr = json.loads((refs_dir / f"la_refs_{bench}_{pooled}_{probe}_{filt}.json").read_text())["rows"][0]
                if lr["n_test"] != r["test"]["states"]:
                    raise ValueError(f"{filt}/{probe}: test {r['test']['states']} != figure n {lr['n_test']}")
                r["figure_n_test"] = lr["n_test"]
            res_f[probe] = r
            log(filt, probe, json.dumps({sp: {k: r[sp][k] for k in ("tasks", "trajectories", "states", "pos_rate")}
                                         for sp in SPLITS}), "split groups", r["split_groups"],
                "test tasks shared with k0 test", r.get("test_tasks_shared_with_k0_test"))
            del c
        out["rerun"][filt] = res_f

    # ---- Legacy (submitted) populations.
    n_turns_legacy, n_cap = {}, {}
    for f in sorted(odir.glob("*.pt")):
        if f.name.endswith("_labels.pt"):
            continue
        d = torch.load(f, weights_only=False, mmap=True)
        n_turns_legacy[d["sample_id"]] = d.get("n_turns") or 1
        n_cap[d["sample_id"]] = d.get("n_captured_steps")
        del d
    log("activation files", len(n_turns_legacy))
    legacy_pooled = f"{pooled}_legacyturns"
    out["legacy"] = {}
    lk0_test_tasks = None
    for probe in PROBES:
        s = torch.load(Path("snapshots") / f"{bench}_{cache}_{probe}_k0.pt", weights_only=False)
        y0 = s["y"].numpy().copy()
        step = s["step_idx_legacy"].numpy().copy()
        sc, sn = factorize(s["sample_id"])
        gc, gn = factorize(s["group_id"])
        first_idx = np.r_[0, np.flatnonzero(np.diff(sc)) + 1]
        if len(first_idx) != len(sn):
            raise ValueError("snapshot rows are not contiguous per trajectory")
        traj_group = {sn[sc[i]]: gn[gc[i]] for i in first_idx}
        group_tasks: dict[str, set] = {}
        for t, g in traj_group.items():
            group_tasks.setdefault(g, set()).add(task_of(t))
        ntl = np.array([n_turns_legacy[x] for x in sn])[sc]
        rows_per = np.bincount(sc, minlength=len(sn))
        bad = sum(1 for i, x in enumerate(sn) if n_cap.get(x) is not None and n_cap[x] != rows_per[i])
        # first position of each (trajectory, step) bucket: the legacy windowed caches' y at k=0
        new_bucket = np.r_[True, (np.diff(sc) != 0) | (np.diff(step) != 0)]
        bucket_id = np.cumsum(new_bucket) - 1
        y_first = y0[new_bucket][bucket_id]
        lr = {"rows_mismatch_n_captured_steps": bad,
              "step_idx_max_plus1_equals_n_turns": float(np.mean(
                  np.maximum.reduceat(step, first_idx) + 1 == np.array([n_turns_legacy[x] for x in sn])))}
        for filt, W in [("k0", 0)] + [(f"max{K}", K) for K in WINDOWS]:
            if W == 0:
                yv, win_mask = y0, np.ones(len(y0), dtype=bool)
            else:
                win_mask = step + W < ntl
                yv = np.where(win_mask, y_first, -1)
            present = sorted({gn[g] for g in np.unique(gc[win_mask])})  # groups with rows in this cache
            _, sname, den, unas = make_split(present, traj_group)
            sog = np.array([sname.get(g, -1) for g in gn])
            yv = np.where(win_mask, yv, -1)
            r = count(sc, sn, gc, gn, yv, sog, den, task_key=task_of)
            if W == 0 and lk0_test_tasks is None:
                lk0_test_tasks = {t for g, sp in sname.items() if sp == 2 for t in group_tasks[g]}
            finish(r, sname, den, unas, None, None)
            r["test_tasks_shared_with_k0_test"] = len({t for g, sp in sname.items() if sp == 2
                                                       for t in group_tasks[g]} & lk0_test_tasks)
            r["test_groups"] = sorted(g for g, sp in sname.items() if sp == 2) if len(sname) < 60 else None
            for i, sp in enumerate(SPLITS):
                r[sp]["rows_in_window"] = int((win_mask & (sog[gc] == i)).sum())
            rid = legacy_pooled if W == 0 else f"{pooled}_shift0_{filt}_legacyturns"
            st = stored_n(rroot / rid / probe / "results.pt")
            r["stored"] = st
            r["matches_stored"] = st is not None and all(st[sp] == r[sp]["states"] for sp in SPLITS)
            lr[filt] = r
            log("legacy", filt, probe, json.dumps({sp: {k: r[sp][k] for k in ("tasks", "trajectories", "states")}
                                                   for sp in SPLITS}), "groups", r["split_groups"],
                "stored", st, "match", r["matches_stored"])
        out["legacy"][probe] = lr
        del s, y0, step

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log("[done] wrote", args.output)
    os._exit(0)


if __name__ == "__main__":
    main()
