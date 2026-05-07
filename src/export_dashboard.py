import json
import math
import torch
from pathlib import Path
from datetime import datetime


def _majority_baseline(labels: list) -> float:
    flat = []
    for lbl in labels:
        if isinstance(lbl, list):
            flat.extend(v for v in lbl if v is not None)
        elif lbl is not None:
            flat.append(lbl)
    if not flat:
        return 0.5
    pos = sum(1 for v in flat if v)
    return max(pos / len(flat), 1 - pos / len(flat))


def export_dashboard(
    run_id: str,
    probe_names: list[str],
    probe_layers: list[int],
    task_name: str,
    model_name: str,
    output_dir: str = "outputs",
    results_dir: str = "results",
    dashboard_dir: str = "dashboard",
    n_bins: int = 10,
) -> None:
    data_dir = Path(dashboard_dir) / "data"
    run_dir = data_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(output_dir) / run_id
    pt_files = sorted(in_dir.glob("*.pt"))

    # Load all sample data
    samples_by_id: dict[str, dict] = {}
    all_labels_by_probe: dict[str, list] = {p: [] for p in probe_names}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        sid = data["sample_id"]
        gen_idx = data["generation_idx"]
        meta = data["metadata"]
        labels = data["labels"]

        if sid not in samples_by_id:
            samples_by_id[sid] = {
                "sample_id": sid,
                "group_id": data["group_id"],
                "prompt": meta.get("prompt", meta.get("prompt_token_ids", "")),
                "generations": [],
            }

        gen_entry = {
            "generation_idx": gen_idx,
            "generated_text": meta.get("raw_text", meta.get("generated_text", "")),
            "labels": {k: v for k, v in labels.items() if k in probe_names},
        }
        samples_by_id[sid]["generations"].append(gen_entry)

        for probe_name in probe_names:
            if probe_name in labels:
                all_labels_by_probe[probe_name].append(labels[probe_name])

    samples_list = sorted(samples_by_id.values(), key=lambda s: s["sample_id"])
    for s in samples_list:
        s["generations"].sort(key=lambda g: g["generation_idx"])

    # Compute majority baselines
    majority_baselines = {p: _majority_baseline(all_labels_by_probe[p]) for p in probe_names}

    # Load probe results
    probe_results: dict[str, dict] = {}
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

    # Write meta.json
    meta_out = {
        "run_id": run_id,
        "task": task_name,
        "model": model_name,
        "probes": probe_names,
        "n_samples": len(samples_list),
        "n_generations": max(
            (len(s["generations"]) for s in samples_list), default=0
        ),
        "majority_baseline": majority_baselines,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))

    # Write samples.json
    (run_dir / "samples.json").write_text(json.dumps(samples_list, indent=2))

    # Write probe_results.json
    (run_dir / "probe_results.json").write_text(json.dumps(probe_results, indent=2))

    # Update manifest.json
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest = [e for e in manifest if e["run_id"] != run_id]
    else:
        manifest = []

    manifest.append({
        "run_id": run_id,
        "task": task_name,
        "model": model_name,
        "probes": probe_names,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
