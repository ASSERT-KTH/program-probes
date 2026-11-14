"""Brief EI (HaQJ-W1c, HaQJ-Q3): edit-indexed horizon. From the hidden state at turn t, predict the label of
the program after the m-th FUTURE edit (m = 1, 2, 3), so the state always precedes the target edit.

Edits: the real edit events of edit_step_index.pt (as in AE0): sid -> [N_e + 1], edit e written in assistant
turn N_e. Outcome of e = the turn label of turn N_e + 1 (outputs/<bench>/<run>/<sid>_labels.pt), the first turn
whose label reflects the edit; undefined (None, or no such turn) -> that outcome is skipped, i.e. a token whose
e_m has an undefined outcome is excluded for that m (e_m is still the m-th future edit; edits are not renumbered).
Future edits of a token in turn t: N_e > t (an edit written in turn t itself is excluded: the token may be inside
it). Target_m = outcome of the m-th future edit; tokens with fewer than m future edits are excluded for m.
Tokens: the pooled k=0 cache (cache/<bench>/<run>/<probe>/layer_L.pt), its task split (seed 42; Table 6 test
tasks), y_t >= 0 (the k=0 probe's population), no window exclusion.

Probe: a new linear probe per (run, property, m, layer) trained exactly as src/probe.py's train_probe_layer
(pooled, mean-centred on train, nn.Linear, Adam, cross-entropy, early stopping on validation loss with patience,
best weights restored, seed 42), on train tokens with Target_m defined. The probes have no C: their
hyperparameters (lr, weight_decay, batch_size, patience) are the rerun's task-split sweep optimum of the k=0
pooled probe of the same benchmark / model / property / layer (paper_rebuttal_candidate/hparams/json/
hparams_per_layer.json); no new sweep. The weight decay (the analogue of C) is chosen on validation AUC from
{the k=0 sweep's value} + --wd-grid (default 1e-2, 1e-1; the sweep range is 1e-5..1e-1), because with the k=0
value alone every probe early-stopped after one or two epochs. Layer chosen by validation AUC (pooled over
validation tokens); with --k0-val-layer only the k=0 probe's validation-best layer (E17) is trained.

Baselines (fit on train tokens with Target_m defined, scored on the same test tokens):
  persistence   y_t (population A only; constant in B and C)
  k0            the k=0 probe score (results/<bench>/<pooled>/<probe>/weights.pt at the Table 6 test-best layer)
  pos_causal    E16 spline LR on log1p(step), log1p(absolute position)       (causal)
  pos_rel       E16 spline LR on relative position (NON-causal: uses the trajectory length)
  edits_so_far  spline LR on log1p(number of edits written before turn t)
  observable    spline LR on log1p(step), log1p(edits so far), log1p(turns since the last edit; t + 1 if none)
  traj          trajectory-shuffle probe: same training at the probe's selected layer on Target_m of another
                training (validation) trajectory, resampled by relative position (src/probe.py
                _shuffle_by_trajectory), scored against the real targets
  prompt_hs     PO's baseline before the first action (hidden state of the first cached token, at the probe's
                selected layer), refit on Target_m
  yT            final label (oracle; reported, never in "beats all")
Populations (test): A all tokens with Target_m defined; B A & y_t = 0; C B & y_T = 1.
AUC pooled over tokens; paired 95% task bootstrap (1000 resamples, seed 42) of probe - every baseline.

Usage: uv run python scripts/rebuttal_edit_indexed.py --run laguna_verified --probe currently_correct \
           --output paper_rebuttal_candidate/edit_indexed/json/ei_laguna_verified_currently_correct.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.probe import _shuffle_by_trajectory  # noqa: E402
from rebuttal_edit_outcome_strata import MIN_TASKS  # noqa: E402
from rebuttal_persistence_analysis import Metric  # noqa: E402
from rebuttal_phase_confound import abs_positions, bootstrap, load_probe, score_layer, split_masks  # noqa: E402
from rebuttal_prompt_only import Trajs, fit_prompt  # noqa: E402

RUNS = {  # bench, pooled run id, k=0 cache dir (= outputs/ and generations/ run name), hparam key
    "laguna_verified": ("swebench", "laguna_xs2_full_pooled", "laguna_xs2_full", "Verified|Laguna-XS.2"),
    "qwen_verified": ("swebench", "qwen36_35b_a3b_full_pooled", "qwen36_35b_a3b_full", "Verified|Qwen3.6-35B-A3B"),
    "laguna_pro": ("swebench_pro", "laguna_xs2_pro_full_pooled", "laguna_xs2_full", "Pro|Laguna-XS.2"),
    "qwen_pro": ("swebench_pro", "qwen36_35b_a3b_pro_full_pooled", "qwen36_35b_a3b_full", "Pro|Qwen3.6-35B-A3B"),
}
LAYERS = [0, 10, 20, 30, 39]
MS = [1, 2, 3]
POPS = ["A", "B", "C"]
CAUSAL = ["persistence", "k0", "pos_causal", "edits_so_far", "observable", "traj", "prompt_hs"]
ALL_BASE = CAUSAL + ["pos_rel"]
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def task(sid: str) -> str:
    return re.sub(r"_run\d+$", "", sid)


def turn_label(v) -> int:
    return -1 if v is None else int(bool(v))


def edit_targets(trajs: Trajs, sids: list[str], step: np.ndarray, y: np.ndarray, index: dict, outdir: Path,
                 probe: str) -> dict:
    """Per row: Target_m, turns to e_m, label changed by e_m, edits so far, turns since last edit; per traj: y_T."""
    n = len(step)
    tgt = {m: np.full(n, -1, np.int8) for m in MS}
    gap = {m: np.full(n, -1, np.int32) for m in MS}
    chg = {m: np.full(n, -1, np.int8) for m in MS}
    eid = {m: np.full(n, -1, np.int32) for m in MS}  # index of e_m within the trajectory's edits
    so_far = np.zeros(n, np.int32)
    since = np.zeros(n, np.int32)
    yT = np.full(trajs.n, -1, np.int8)
    bounds = np.r_[trajs.first, n]
    chk = {"rows_label_mismatch": 0, "trajectories_with_edits": 0, "edits": 0, "edits_outcome_undefined": 0,
           "edits_before_undefined": 0}
    for j, sid in enumerate(trajs.sids):
        r0, r1 = bounds[j], bounds[j + 1]
        st = step[r0:r1].astype(np.int64)
        tl = [turn_label(v) for v in torch.load(outdir / f"{sid}_labels.pt", weights_only=False)["turn_labels"][probe]]
        lab = np.array(tl + [-1], dtype=np.int8)  # lab[len(tl)] = -1 (turn past the end)
        yT[j] = tl[-1] if tl else -1
        chk["rows_label_mismatch"] += int((lab[np.minimum(st, len(tl))] != y[r0:r1]).sum())
        N = np.array(sorted(t - 1 for t in index.get(sid, [])), dtype=np.int64)
        if len(N):
            chk["trajectories_with_edits"] += 1
        out_lab = lab[np.minimum(N + 1, len(tl))]
        before_lab = lab[np.minimum(N, len(tl))]
        chk["edits"] += len(N)
        chk["edits_outcome_undefined"] += int((out_lab < 0).sum())
        chk["edits_before_undefined"] += int((before_lab < 0).sum())
        nxt = np.searchsorted(N, st, side="right")  # first edit with N_e > t
        prev = np.searchsorted(N, st, side="left")  # edits with N_e < t
        so_far[r0:r1] = prev
        since[r0:r1] = np.where(prev > 0, st - N[np.maximum(prev - 1, 0)] if len(N) else 0, st + 1)
        for m in MS:
            e = nxt + m - 1
            ok = e < len(N)
            ee = np.minimum(e, max(len(N) - 1, 0))
            if not len(N):
                continue
            t_m = np.where(ok, out_lab[ee], -1)
            tgt[m][r0:r1] = t_m
            gap[m][r0:r1] = np.where(ok, N[ee] - st, -1)
            b = before_lab[ee]
            chg[m][r0:r1] = np.where(ok & (t_m >= 0) & (b >= 0), (t_m != b).astype(np.int8), -1)
            eid[m][r0:r1] = np.where(ok, ee, -1)
    return {"tgt": tgt, "gap": gap, "chg": chg, "eid": eid, "so_far": so_far, "since": since, "yT": yT, "check": chk}


def fit_spline(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int, max_n: int = 2_000_000) -> tuple[np.ndarray, str]:
    """E16's spline logistic regression (rebuttal_phase_confound.fit_position); uniform knots if quantile knots tie."""
    rng = np.random.default_rng(seed)
    if len(ytr) > max_n:
        sel = rng.choice(len(ytr), max_n, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean()) if len(ytr) else 0.5), "constant"
    for knots in ("quantile", "uniform"):
        try:
            model = make_pipeline(SplineTransformer(n_knots=8, degree=3, knots=knots, extrapolation="constant"),
                                  LogisticRegression(max_iter=2000))
            model.fit(Xtr, ytr)
            return model.predict_proba(Xte)[:, 1], knots
        except ValueError:
            continue
    raise ValueError("spline fit failed")


def train_linear(H: torch.Tensor, tr: np.ndarray, ytr: np.ndarray, va: np.ndarray, yva: np.ndarray, hp: dict,
                 seed: int) -> tuple[torch.nn.Module, torch.Tensor, dict]:
    """src/probe.py train_probe_layer (pooled): centre on train mean, Adam, CE, early stop on val loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = H.shape[1]
    mean = torch.zeros(D, dtype=torch.float64)
    for s in range(0, len(tr), 200_000):
        mean += H[torch.from_numpy(tr[s:s + 200_000])].double().sum(0)
    mean = (mean / len(tr)).float()
    Ht = H[torch.from_numpy(tr)].float()
    Ht -= mean
    Hv = H[torch.from_numpy(va)].float()
    Hv -= mean
    yt, yv = torch.from_numpy(ytr.astype(np.int64)), torch.from_numpy(yva.astype(np.int64))
    model = torch.nn.Linear(D, 2)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit = torch.nn.CrossEntropyLoss()
    best, best_w, no_imp, n_ep = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    bs, n = int(hp["batch_size"]), len(tr)
    for epoch in range(1000):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = crit(model(Ht[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Hv), yv).item()
        n_ep = epoch + 1
        if vl < best:
            best, best_w, no_imp = vl, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= hp["patience"]:
                break
    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(Hv), dim=1)[:, 1].numpy()
    del Ht, Hv
    va_auc = float(roc_auc_score(yva, pv)) if yva.min() != yva.max() else float("nan")
    return model, mean, {"n_epochs": n_ep, "best_val_loss": best, "val_auc": va_auc, "n_train": int(n),
                         "n_val": int(len(va)), "train_pos_rate": float(ytr.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--probe", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=LAYERS, help="0-indexed layers to train probes on")
    ap.add_argument("--k0-val-layer", action="store_true",
                    help="train only at the k=0 probe's validation-best layer (E17 val_best_layer), overrides --layers")
    ap.add_argument("--wd-grid", type=float, nargs="*", default=[1e-2, 1e-1],
                    help="weight decays tried besides the k=0 sweep's value; chosen on validation AUC")
    ap.add_argument("--no-traj", action="store_true", help="skip the trajectory-shuffle control")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    bench, pooled, k0dir, hkey = RUNS[args.run]
    probe = args.probe
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    cdir = Path("cache", bench, k0dir, probe)
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = list(meta["sample_id"]), list(meta["group_id"])
    y = meta["y"].numpy().astype(np.int8)
    step = meta["step_idx"].numpy().astype(np.int64)
    rel = meta["rel_pos"].numpy().astype(np.float64)
    del meta
    trajs = Trajs(sids, step)
    apos = abs_positions(sids)
    in_tr, in_va, in_te = split_masks(gids, args.seed)
    index = torch.load(Path("cache", bench, k0dir, "edit_step_index.pt"), weights_only=False)
    E = edit_targets(trajs, sids, step, y, index, Path("outputs", bench, k0dir), probe)
    log("rows", len(y), "trajectories", trajs.n, E["check"])
    if E["check"]["rows_label_mismatch"]:
        raise ValueError(f"cache y differs from turn_labels on {E['check']['rows_label_mismatch']} rows")
    hp_all = json.loads(Path("paper_rebuttal_candidate/hparams/json/hparams_per_layer.json").read_text())[f"{hkey}|{probe}"]
    hps = {L: {k: hp_all[str(L + 1)][k] for k in ("lr", "weight_decay", "batch_size", "patience")} for L in args.layers}
    e17 = json.loads(Path("reports/rerun/phase", f"k0_{bench}_{pooled}_{probe}.json").read_text())
    L6 = int(e17["real_test_best_layer"])
    if args.k0_val_layer:
        args.layers = [int(e17["val_best_layer"]["real"])]
        hps = {L: {k: hp_all[str(L + 1)][k] for k in ("lr", "weight_decay", "batch_size", "patience")} for L in args.layers}
    yt_ok = y >= 0
    pop = {m: yt_ok & (E["tgt"][m] >= 0) for m in MS}
    union = np.flatnonzero(pop[1] | pop[2] | pop[3])
    pos_in_union = np.full(len(y), -1, np.int64)
    pos_in_union[union] = np.arange(len(union))
    rows_te = {m: np.flatnonzero(pop[m] & in_te) for m in MS}

    # Baselines that do not depend on the hidden states.
    feats = {"pos_causal": np.column_stack([np.log1p(step), np.log1p(apos)]), "pos_rel": rel[:, None],
             "edits_so_far": np.log1p(E["so_far"])[:, None].astype(np.float64),
             "observable": np.column_stack([np.log1p(step), np.log1p(E["so_far"]), np.log1p(E["since"])])}
    base_sc: dict[int, dict] = {m: {} for m in MS}
    knots: dict = {}
    for m in MS:
        trn = pop[m] & in_tr
        for name, X in feats.items():
            base_sc[m][name], knots[f"{name}|m{m}"] = fit_spline(X[trn], E["tgt"][m][trn].astype(np.int64),
                                                                X[rows_te[m]], args.seed)
        base_sc[m]["persistence"] = y[rows_te[m]].astype(np.float64)
        base_sc[m]["yT"] = E["yT"][trajs.of_row[rows_te[m]]].astype(np.float64)
    log("position / count baselines fit", knots)

    # Probes per layer.
    probes: dict = {m: {} for m in MS}
    test_scores: dict = {m: {} for m in MS}
    first_states: dict[int, np.ndarray] = {}
    layers_needed = sorted(set(args.layers) | {L6})
    k0_scores = {}
    for L in layers_needed:
        t_load = time.time()
        Hm = torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"]
        H = Hm[torch.from_numpy(union)].clone()  # float16, population rows only
        first_states[L] = Hm[torch.from_numpy(trajs.first)].float().numpy().astype(np.float64)
        if L == L6:
            k0p = {"k0": load_probe(Path("results", bench, pooled, probe, "weights.pt"), L)}
            for m in MS:
                k0_scores[m] = score_layer(Hm, rows_te[m], k0p)["k0"]
        del Hm
        log(f"L{L} loaded {tuple(H.shape)} in {time.time() - t_load:.0f}s")
        if L not in args.layers:
            del H
            continue
        for m in MS:
            tgt = E["tgt"][m]
            tr, va = np.flatnonzero(pop[m] & in_tr), np.flatnonzero(pop[m] & in_va)
            grid = sorted({hps[L]["weight_decay"], *args.wd_grid})
            best = None
            for wd in grid:
                hp = {**hps[L], "weight_decay": wd}
                t_fit = time.time()
                model, mean, fit = train_linear(H, pos_in_union[tr], tgt[tr], pos_in_union[va], tgt[va], hp, args.seed)
                fit["seconds"] = round(time.time() - t_fit, 1)
                fit["weight_decay"] = wd
                log(f"L{L} m={m} wd={wd:.1e} {json.dumps(fit)}")
                if best is None or fit["val_auc"] > best[2]["val_auc"]:
                    best = (model, mean, fit, hp)
            model, mean, fit, hp = best
            fit = {**fit, "grid": grid}
            probes[m][L] = (model, mean, fit, hp)
            test_scores[m][L] = score_layer(H, pos_in_union[rows_te[m]], {"p": (model, mean)})["p"]
        del H
    for m in MS:
        base_sc[m]["k0"] = k0_scores[m]

    def vauc(m: int, L: int) -> float:
        v = probes[m][L][2]["val_auc"]
        return v if v == v else -1.0

    lsel = {m: max(args.layers, key=lambda L: vauc(m, L)) for m in MS}
    # Trajectory shuffle at each m's selected layer (one load per distinct layer).
    traj_sc, traj_fit = {}, {}
    if not args.no_traj:
        for L in sorted(set(lsel.values())):
            Hm = torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"]
            H = Hm[torch.from_numpy(union)].clone()
            del Hm
            for m in [m for m in MS if lsel[m] == L]:
                t_fit = time.time()
                tgt = E["tgt"][m]
                tr, va = np.flatnonzero(pop[m] & in_tr), np.flatnonzero(pop[m] & in_va)
                torch.manual_seed(args.seed)
                ytr_s = _shuffle_by_trajectory(torch.from_numpy(tgt[tr].astype(np.int64)), [sids[i] for i in tr]).numpy()
                yva_s = _shuffle_by_trajectory(torch.from_numpy(tgt[va].astype(np.int64)), [sids[i] for i in va]).numpy()
                model, mean, fit = train_linear(H, pos_in_union[tr], ytr_s, pos_in_union[va], yva_s, probes[m][L][3], args.seed)
                fit["weight_decay"] = probes[m][L][3]["weight_decay"]
                traj_sc[m] = score_layer(H, pos_in_union[rows_te[m]], {"p": (model, mean)})["p"]
                fit["seconds"] = round(time.time() - t_fit, 1)
                fit["note"] = "val_auc is against the shuffled validation targets"
                traj_fit[m] = fit
                log(f"traj m={m} L{L} {json.dumps(fit)}")
            del H

    out_rows = []
    for m in MS:
        tgt = E["tgt"][m]
        Lsel = lsel[m]
        sc = {"probe": test_scores[m][Lsel], **base_sc[m]}
        fits = {"probe": {str(L + 1): probes[m][L][2] for L in args.layers},
                "hparams_k0_sweep": {str(L + 1): hps[L] for L in args.layers},
                "hparams_selected": {str(L + 1): probes[m][L][3] for L in args.layers}}
        # prompt_hs (PO) at the probe's selected layer, refit on Target_m.
        p_hs, fits["prompt_hs"] = fit_prompt(first_states[Lsel], trajs.of_row, tgt, pop[m] & in_tr, pop[m] & in_va, dense=True)
        sc["prompt_hs"] = p_hs[trajs.of_row[rows_te[m]]]
        if m in traj_sc:
            sc["traj"], fits["traj"] = traj_sc[m], traj_fit[m]

        rte = rows_te[m]
        yy = tgt[rte].astype(np.int64)
        tsids = [sids[i] for i in rte]
        tasks = np.array([task(s) for s in tsids])
        yt_te = y[rte]
        yT_te = E["yT"][trajs.of_row[rte]]
        masks = {"A": np.ones(len(rte), bool), "B": yt_te == 0, "C": (yt_te == 0) & (yT_te == 1)}
        counts = {}
        for p, mk in masks.items():
            posm = mk & (yy == 1)
            ch = E["chg"][m][rte][mk]
            ed = {(int(trajs.of_row[i]), int(E["eid"][m][i])): int(E["chg"][m][i]) for i in rte[mk]}
            ed_def = [v for v in ed.values() if v >= 0]
            counts[p] = {"n": int(mk.sum()), "n_pos": int(posm.sum()), "pos_rate": float(yy[mk].mean()) if mk.any() else None,
                         "n_tasks": int(len(set(tasks[mk]))), "n_tasks_pos": int(len(set(tasks[posm]))),
                         "underpowered": bool(len(set(tasks[posm])) < MIN_TASKS),
                         "share_tokens_target_edit_changes_label": float((ch == 1).sum() / max((ch >= 0).sum(), 1)),
                         "n_target_edits": len(ed), "share_target_edits_change_label": float(np.mean(ed_def)) if ed_def else None,
                         "median_turns_to_e_m": float(np.median(E["gap"][m][rte][mk])) if mk.any() else None}
        names = [n for n in sc]
        metrics = {f"{n}|{p}": Metric(sc[n], yy, mk) for n in names for p, mk in masks.items()
                   if not (n == "persistence" and p != "A") and not (n == "yT" and p == "C")}
        diffs = {f"probe-{n}|{p}": (f"probe|{p}", f"{n}|{p}") for n in names if n != "probe"
                 for p in POPS if f"{n}|{p}" in metrics}
        point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
        out_rows.append({"m": m, "layer": Lsel + 1, "counts": counts, "fit": fits, "auc": point, "ci95": cis})
        log(f"m={m} L{Lsel + 1}", json.dumps({k: round(v, 3) for k, v in point.items() if v == v and "-" not in k}))
        log(f"m={m}", json.dumps(counts))

    out = {"run": args.run, "bench": bench, "pooled": pooled, "cache": str(cdir), "probe": probe,
           "k0_layer_table6": L6 + 1, "layers_trained": [L + 1 for L in args.layers], "k0_val_layer": args.k0_val_layer,
           "wd_grid_extra": args.wd_grid, "traj_shuffle": not args.no_traj,
           "n_boot": args.n_boot, "min_tasks": MIN_TASKS, "check": E["check"], "spline_knots": knots, "rows": out_rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1, default=float))
    log("wrote", args.output)
    os._exit(0)


if __name__ == "__main__":
    main()
