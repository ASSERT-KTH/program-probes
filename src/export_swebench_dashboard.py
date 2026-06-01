"""Export SWE-bench probe results to dashboard format."""
import json
import torch
from datetime import datetime
from pathlib import Path

from src.probe import _split_groups, _bin_index


def _compute_transitions(labels: list) -> dict:
    false_to_true = 0
    true_to_false = 0
    changes = 0
    prev = None
    for val in labels:
        if val is None:
            continue
        if prev is not None and val != prev:
            changes += 1
            if prev is False and val is True:
                false_to_true += 1
            elif prev is True and val is False:
                true_to_false += 1
        prev = val
    return {"total": changes, "false_to_true": false_to_true, "true_to_false": true_to_false}



def _per_bin_majority_baseline_from_cache(
    cache_dir: Path, probe_name: str, n_bins: int = 10, seed: int = 42,
    eval_bin_axis: str = "position",
) -> dict[str, float]:
    """Compute per-bin majority baseline from the test split of the probe cache."""
    probe_dir = cache_dir / probe_name
    layer_files = sorted(probe_dir.glob("layer_*.pt"))
    if not layer_files:
        raise FileNotFoundError(f"No cache files found for probe '{probe_name}' in {probe_dir}")
    data = torch.load(layer_files[0], weights_only=False)
    y = data["y"]
    rel_pos = data["rel_pos"]
    step_idx = data.get("step_idx")
    group_ids = data["group_id"]
    _, _, test_groups = _split_groups(group_ids, seed)
    result = {}

    if eval_bin_axis == "step_absolute":
        if step_idx is None:
            raise ValueError(f"eval_bin_axis='step_absolute' requires step_idx in cache for probe '{probe_name}'")
        test_mask = torch.tensor([group_ids[i] in test_groups and y[i].item() >= 0 for i in range(len(y))])
        unique_steps = sorted(step_idx[test_mask].unique().tolist())
        for s in unique_steps:
            bin_mask = test_mask & (step_idx == s)
            bin_y = y[bin_mask]
            if len(bin_y) == 0:
                continue
            pos = (bin_y == 1).sum().item()
            total = len(bin_y)
            result[str(s)] = max(pos / total, 1 - pos / total)
    elif eval_bin_axis == "step_relative":
        if step_idx is None:
            raise ValueError(f"eval_bin_axis='step_relative' requires step_idx in cache for probe '{probe_name}'")
        # Per-trajectory max step for normalization
        sample_max_step: dict[str, int] = {}
        for sid, s in zip(group_ids, step_idx.tolist()):
            if s > sample_max_step.get(sid, 0):
                sample_max_step[sid] = s
        for b in range(n_bins):
            bin_mask = torch.tensor([
                group_ids[i] in test_groups
                and y[i].item() >= 0
                and _bin_index(step_idx[i].item() / max(sample_max_step.get(group_ids[i], 1), 1), n_bins) == b
                for i in range(len(y))
            ])
            bin_y = y[bin_mask]
            if len(bin_y) == 0:
                continue  # static probes only have data at step 0; skip empty bins
            pos = (bin_y == 1).sum().item()
            total = len(bin_y)
            result[str(b)] = max(pos / total, 1 - pos / total)
    else:  # position
        for b in range(n_bins):
            bin_mask = torch.tensor([
                group_ids[i] in test_groups
                and y[i].item() >= 0
                and _bin_index(rel_pos[i].item(), n_bins) == b
                for i in range(len(y))
            ])
            bin_y = y[bin_mask]
            if len(bin_y) == 0:
                raise ValueError(f"No test samples in bin {b} for probe '{probe_name}' in {probe_dir}")
            pos = (bin_y == 1).sum().item()
            total = len(bin_y)
            result[str(b)] = max(pos / total, 1 - pos / total)
    return result


def _per_turn_labels(segments: list[dict], n_tokens: int, label_seq: list) -> list:
    """Return one label per assistant turn — the value at the last extracted position in that turn.

    Infers stride from total assistant tokens vs label sequence length.
    """
    mask = [0] * n_tokens
    for seg in segments:
        if seg.get("role") == "assistant":
            for pos in range(seg["start_token"], min(seg["end_token"], n_tokens)):
                mask[pos] = 1

    total_asst = sum(mask)
    n_labels = len(label_seq)
    if n_labels == 0 or total_asst == 0:
        asst_count = sum(1 for s in segments if s.get("role") == "assistant")
        return [None] * asst_count

    stride = max(1, round(total_asst / n_labels))

    # Build list of strided extracted positions with their label index
    extracted: list[int] = []
    counter = 0
    for pos, m in enumerate(mask):
        if m == 1:
            if counter % stride == 0:
                extracted.append(pos)
            counter += 1

    # For each assistant segment, find the last extracted position and its label
    result = []
    for seg in segments:
        if seg.get("role") != "assistant":
            continue
        start, end = seg["start_token"], min(seg["end_token"], n_tokens)
        # Last extracted position within this segment
        seg_extracted = [i for i, pos in enumerate(extracted) if start <= pos < end]
        if seg_extracted:
            idx = min(seg_extracted[-1], len(label_seq) - 1)
            result.append(label_seq[idx])
        else:
            result.append(None)
    return result


def _load_traj_data(traj_dir: Path, instance_id: str) -> tuple[list[dict], list[dict], int, int]:
    """Load messages, segments, total token count, and turn count from a trajectory JSON.

    Returns (messages, segments, n_tokens, n_turns). messages contains only role+content.
    """
    fname = traj_dir / f"{instance_id.replace('/', '_')}.json"
    if not fname.exists():
        return [], [], 0, 0
    with open(fname) as f:
        traj = json.load(f)
    messages = [{"role": m["role"], "content": m.get("content", "")}
                for m in traj.get("messages", [])]
    tok = traj.get("tokenization", {})
    segments = tok.get("segments", [])
    n_tokens = len(tok.get("token_ids", []))
    n_turns = sum(1 for s in segments if s.get("role") == "assistant")
    return messages, segments, n_tokens, n_turns


def export_swebench_dashboard(
    run_id: str,
    probe_names: list[str],
    probe_layers: list[int],
    model_name: str,
    output_dir: str = "outputs/swebench",
    output_run_id: str | None = None,
    results_dir: str = "results/swebench",
    cache_dir: str = "cache/swebench",
    dashboard_dir: str = "dashboard",
    traj_dir: str | None = None,
    n_bins: int = 10,
    eval_bin_axis: str = "position",
) -> None:
    data_dir = Path(dashboard_dir) / "data"
    run_dir = data_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(output_dir) / (output_run_id or run_id)
    pt_files = sorted(f for f in in_dir.glob("*.pt") if not f.stem.endswith("_labels"))
    print(f"[export] {len(pt_files)} .pt files in {in_dir}")

    traj_path = Path(traj_dir) if traj_dir else None

    samples_list = []
    stats_n_tokens: list[int] = []
    stats_n_turns: list[int] = []
    stats_outcomes: list[bool] = []
    probe_trans: dict[str, list[dict]] = {p: [] for p in probe_names}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        instance_id = data["instance_id"]
        outcome = bool(data.get("outcome", False))
        n_captured_steps = data.get("n_captured_steps", 0)
        label_file = pt_file.parent / f"{pt_file.stem}_labels.pt"
        if label_file.exists():
            labels = torch.load(label_file, weights_only=False)["labels"]
        else:
            labels = data.get("labels", {})

        # Load messages and accurate token/turn counts from trajectory JSON if available
        messages: list[dict] = []
        segments: list[dict] = []
        if traj_path:
            messages, segments, n_tokens, n_turns = _load_traj_data(traj_path, instance_id)
        else:
            n_tokens = data.get("n_tokens", n_captured_steps)
            n_turns = data.get("n_turns", None)

        stats_n_tokens.append(n_tokens)
        if n_turns:
            stats_n_turns.append(n_turns)
        stats_outcomes.append(outcome)

        label_entry: dict = {}
        per_turn_labels: dict[str, list] = {}
        for probe_name in probe_names:
            lbl = labels.get(probe_name)
            if isinstance(lbl, list):
                label_entry[probe_name] = lbl
                probe_trans[probe_name].append(_compute_transitions(lbl))
                if segments:
                    per_turn_labels[probe_name] = _per_turn_labels(segments, n_tokens, lbl)
            else:
                label_entry[probe_name] = outcome
                probe_trans[probe_name].append({"total": 0, "false_to_true": 0, "true_to_false": 0})

        samples_list.append({
            "sample_id": instance_id,
            "group_id": instance_id,
            "outcome": outcome,
            "messages": messages,
            "generations": [{
                "generation_idx": 0,
                "outcome": outcome,
                "labels": label_entry,
                "per_turn_labels": per_turn_labels,
            }],
        })

    # Per-probe transition stats
    probe_stats: dict = {}
    for probe_name in probe_names:
        trans_list = probe_trans[probe_name]
        def _load_labels_for(f: Path) -> dict:
            lf = f.parent / f"{f.stem}_labels.pt"
            if lf.exists():
                return torch.load(lf, weights_only=False)["labels"]
            return torch.load(f, weights_only=False).get("labels", {})
        has_dynamic = any(isinstance(_load_labels_for(f).get(probe_name), list) for f in pt_files[:1])
        probe_stats[probe_name] = {
            "is_dynamic": has_dynamic,
            "changes_per_traj": [t["total"] for t in trans_list],
            "false_to_true_per_traj": [t["false_to_true"] for t in trans_list],
            "true_to_false_per_traj": [t["true_to_false"] for t in trans_list],
            "total_false_to_true": sum(t["false_to_true"] for t in trans_list),
            "total_true_to_false": sum(t["true_to_false"] for t in trans_list),
        }

    stats_out = {
        "is_agentic": True,
        "n_tokens": stats_n_tokens,
        "n_turns": stats_n_turns if stats_n_turns else None,
        "outcomes": stats_outcomes,
        "probe_transitions": probe_stats,
    }
    (run_dir / "stats.json").write_text(json.dumps(stats_out))

    # Per-bin majority baselines
    cache_run = output_run_id or run_id
    majority_baselines_per_bin = {
        p: _per_bin_majority_baseline_from_cache(Path(cache_dir) / cache_run, p, n_bins, eval_bin_axis=eval_bin_axis)
        for p in probe_names
    }

    # Probe results
    probe_results: dict = {}
    for probe_name in probe_names:
        results_path = Path(results_dir) / run_id / probe_name / "results.pt"
        if not results_path.exists():
            continue
        all_results = torch.load(results_path, weights_only=False)
        probe_results[probe_name] = {}
        for layer_idx, layer_results in all_results.items():
            probe_results[probe_name][str(layer_idx)] = {}
            for r in layer_results:
                bin_idx = r.bin_idx if hasattr(r, "bin_idx") else r["bin_idx"]
                probe_results[probe_name][str(layer_idx)][str(bin_idx)] = {
                    "test_acc": r.test_acc if hasattr(r, "test_acc") else r["test_acc"],
                    "val_acc": r.val_acc if hasattr(r, "val_acc") else r["val_acc"],
                    "test_auc": r.test_auc if hasattr(r, "test_auc") else None,
                    "val_auc": r.val_auc if hasattr(r, "val_auc") else None,
                    "test_ece": r.test_ece if hasattr(r, "test_ece") else None,
                    "val_ece": r.val_ece if hasattr(r, "val_ece") else None,
                    "n_train": r.n_train if hasattr(r, "n_train") else r["n_train"],
                    "n_val": r.n_val if hasattr(r, "n_val") else r["n_val"],
                    "n_test": r.n_test if hasattr(r, "n_test") else r["n_test"],
                }

    meta_out = {
        "run_id": run_id,
        "task": "swebench",
        "model": model_name,
        "probes": probe_names,
        "probe_layers": probe_layers,
        "n_samples": len(samples_list),
        "n_generations": 1,
        "majority_baseline_per_bin": majority_baselines_per_bin,
        "eval_bin_axis": eval_bin_axis,
        "is_agentic": True,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
    (run_dir / "samples.json").write_text(json.dumps(samples_list, indent=2))
    (run_dir / "probe_results.json").write_text(json.dumps(probe_results, indent=2))

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [e for e in manifest if e["run_id"] != run_id]
    manifest.append({
        "run_id": run_id,
        "task": "swebench",
        "model": model_name,
        "probes": probe_names,
        "is_agentic": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Copy any figures for this run into dashboard/data/figures/ and update figures manifest
    _export_figures(run_id=run_id, figures_src_dir="figures", data_dir=data_dir)

    print(f"[export] wrote dashboard data to {run_dir}")


def _export_figures(run_id: str, figures_src_dir: str, data_dir: Path) -> None:
    import shutil
    figures_src = Path(figures_src_dir) / run_id
    if not figures_src.exists():
        return

    figures_dst = data_dir / "figures" / run_id
    figures_dst.mkdir(parents=True, exist_ok=True)

    figures_manifest_path = data_dir / "figures" / "manifest.json"
    figures_manifest = json.loads(figures_manifest_path.read_text()) if figures_manifest_path.exists() else []
    # Remove stale entries for this run
    figures_manifest = [e for e in figures_manifest if e.get("run_id") != run_id]

    for png in sorted(figures_src.glob("*.png")):
        shutil.copy2(png, figures_dst / png.name)
        stem = png.stem  # e.g. "will_resolve_heatmap" or "will_resolve_lookahead"
        figures_manifest.append({
            "run_id": run_id,
            "path": f"{run_id}/{png.name}",
            "title": f"{run_id} / {stem.replace('_', ' ')}",
        })
        print(f"[export] copied figure {png.name}")

    figures_manifest_path.write_text(json.dumps(figures_manifest, indent=2))
