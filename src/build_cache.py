import torch
from pathlib import Path


def build_cache(
    run_id: str,
    probe_names: list[str],
    output_dir: str = "outputs",
    cache_dir: str = "cache",
    label_shift: int = 0,
    max_label_shift: int | None = None,
    cache_run_id: str | None = None,
    group_by: str = "project",
) -> None:
    in_dir = Path(output_dir) / run_id
    pt_files = sorted(f for f in in_dir.glob("*.pt") if not f.stem.endswith("_labels"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {in_dir}")

    print(f"Found {len(pt_files)} output files", flush=True)
    first = torch.load(pt_files[0], weights_only=False)
    probe_layer_indices = list(first["activations"].keys())
    del first
    print(f"Probe layers: {probe_layer_indices}", flush=True)

    for probe_name in probe_names:
        out_dir = Path(cache_dir) / (cache_run_id or run_id) / probe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Process one layer at a time to cap peak memory at ~2x one layer's size
        for layer_idx in probe_layer_indices:
            print(f"  [{probe_name}] Building layer {layer_idx}...", flush=True)
            H_list, y_list, rel_list, step_list, sid_list, gid_list = [], [], [], [], [], []
            target_step_list: list[int] = []
            y_original_list: list[int] = []  # only populated when label_shift > 0

            for fi, f in enumerate(pt_files):
                if fi % 500 == 0:
                    print(f"    file {fi}/{len(pt_files)}: {f.name}", flush=True)
                data = torch.load(f, weights_only=False)
                label_path = f.parent / f"{f.stem}_labels.pt"
                pos_turns = None
                turn_labels = None
                if label_path.exists():
                    label_data = torch.load(label_path, weights_only=False)
                    label = label_data["labels"].get(probe_name)
                    pos_turns = label_data.get("step_idx")
                    turn_labels = label_data.get("turn_labels", {}).get(probe_name)
                else:
                    label = data.get("labels", {}).get(probe_name)
                sample = data["sample_id"]
                raw_group = data["group_id"]
                group = raw_group.rsplit('-', 1)[0] if group_by == "project" else raw_group

                acts = data["activations"][layer_idx]  # [T, hidden_dim] float16
                T = acts.shape[0]
                del data

                if T == 0:
                    continue

                rel_pos = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)

                if isinstance(label, list):
                    if pos_turns is None or turn_labels is None:
                        raise ValueError(
                            f"{label_path} has no per-position turn indices for dynamic probe "
                            f"{probe_name!r}; re-run run_attach_labels_swebench.py."
                        )
                    if len(pos_turns) != T or len(label) != T:
                        raise ValueError(
                            f"{f.name}: {T} activations but {len(label)} labels / "
                            f"{len(pos_turns)} turn indices."
                        )
                    # Per-turn labels, including turns with no extracted position.
                    turn_label = [-1 if raw is None else (1 if raw else 0) for raw in turn_labels]
                    K = len(turn_label)
                    y_vals = [-1 if label[t] is None else (1 if label[t] else 0) for t in range(T)]
                    step_idx_vals = list(pos_turns)
                else:
                    y_int = -1 if label is None else (1 if label else 0)
                    y_vals = [y_int] * T
                    step_idx_vals = [0] * T

                # window_shift determines which tokens are kept (max_label_shift if set,
                # else label_shift) — ensures same token set across all k when max_label_shift is fixed
                window_shift = max_label_shift if max_label_shift is not None else label_shift
                if window_shift > 0 and isinstance(label, list):
                    # Keep tokens within the fixed window (window_shift), label from label_shift ahead
                    valid = [t for t in range(T) if step_idx_vals[t] + window_shift < K]
                    if not valid:
                        continue
                    valid_t = torch.tensor(valid)
                    acts = acts[valid_t]
                    rel_pos = rel_pos[valid_t]
                    y_original = [y_vals[t] for t in valid]
                    y_vals = [turn_label[step_idx_vals[t] + label_shift] for t in valid]
                    step_idx_vals = [step_idx_vals[t] for t in valid]
                    T = len(valid)
                    y_original_list.extend(y_original)

                H_list.append(acts)
                y_list.extend(y_vals)
                rel_list.extend(rel_pos.tolist())
                step_list.extend(step_idx_vals)
                # Turn whose label y refers to (differs from step_idx when label_shift > 0).
                shift = label_shift if isinstance(label, list) else 0
                target_step_list.extend(s + shift for s in step_idx_vals)
                sid_list.extend([sample] * T)
                gid_list.extend([group] * T)

            if not H_list:
                continue

            print(f"  [{probe_name}] Saving layer {layer_idx}...", flush=True)
            cache = {
                "H": torch.cat(H_list, dim=0),  # float16
                "y": torch.tensor(y_list, dtype=torch.int64),
                "rel_pos": torch.tensor(rel_list, dtype=torch.float32),
                "step_idx": torch.tensor(step_list, dtype=torch.int64),
                "target_step_idx": torch.tensor(target_step_list, dtype=torch.int64),
                "sample_id": sid_list,
                "group_id": gid_list,
            }
            if label_shift > 0:
                cache["y_original"] = torch.tensor(y_original_list, dtype=torch.int64)
            torch.save(cache, out_dir / f"layer_{layer_idx}.pt")
            del cache, H_list, y_list, rel_list, step_list, target_step_list, sid_list, gid_list, y_original_list
