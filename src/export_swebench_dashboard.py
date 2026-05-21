"""Export SWE-bench probe results to dashboard format."""
import json
import math
import random
import torch
from datetime import datetime
from pathlib import Path


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


def _majority_baseline_from_cache(cache_dir: Path, probe_name: str, seed: int = 42) -> float:
    probe_dir = cache_dir / probe_name
    layer_files = sorted(probe_dir.glob("layer_*.pt"))
    if not layer_files:
        return 0.5
    data = torch.load(layer_files[0], weights_only=False)
    y = data["y"]
    group_ids = data["group_id"]
    unique_groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    n = len(unique_groups)
    test_groups = set(unique_groups[math.floor(0.85 * n):])
    test_mask = torch.tensor([
        group_ids[i] in test_groups and y[i].item() >= 0
        for i in range(len(y))
    ])
    test_y = y[test_mask]
    if len(test_y) == 0:
        return 0.5
    pos = (test_y == 1).sum().item()
    total = len(test_y)
    return max(pos / total, 1 - pos / total)


def export_swebench_dashboard(
    run_id: str,
    probe_names: list[str],
    probe_layers: list[int],
    model_name: str,
    output_dir: str = "outputs/swebench",
    results_dir: str = "results/swebench",
    cache_dir: str = "cache/swebench",
    dashboard_dir: str = "dashboard",
    n_bins: int = 10,
) -> None:
    data_dir = Path(dashboard_dir) / "data"
    run_dir = data_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(output_dir) / run_id
    pt_files = sorted(in_dir.glob("*.pt"))
    print(f"[export] {len(pt_files)} .pt files in {in_dir}")

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
        n_tokens = data.get("n_tokens", n_captured_steps)
        n_turns = data.get("n_turns", None)
        labels = data["labels"]

        stats_n_tokens.append(n_tokens)
        if n_turns is not None:
            stats_n_turns.append(n_turns)
        stats_outcomes.append(outcome)

        label_entry: dict = {}
        for probe_name in probe_names:
            lbl = labels.get(probe_name)
            if isinstance(lbl, list):
                label_entry[probe_name] = lbl
                probe_trans[probe_name].append(_compute_transitions(lbl))
            else:
                # scalar probe — use outcome as trajectory-level label
                label_entry[probe_name] = outcome
                probe_trans[probe_name].append({"total": 0, "false_to_true": 0, "true_to_false": 0})

        samples_list.append({
            "sample_id": instance_id,
            "group_id": instance_id,
            "prompt": "",
            "outcome": outcome,
            "generations": [{
                "generation_idx": 0,
                "generated_text": "",
                "outcome": outcome,
                "labels": label_entry,
            }],
        })

    # Per-probe transition stats
    probe_stats: dict = {}
    for probe_name in probe_names:
        trans_list = probe_trans[probe_name]
        has_dynamic = any(isinstance(labels.get(probe_name), list)
                         for labels in (torch.load(f, weights_only=False)["labels"] for f in pt_files[:1]))
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

    # Majority baselines
    majority_baselines = {
        p: _majority_baseline_from_cache(Path(cache_dir) / run_id, p)
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
        "majority_baseline": majority_baselines,
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
    print(f"[export] wrote dashboard data to {run_dir}")
