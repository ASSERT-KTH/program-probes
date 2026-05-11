import torch
from pathlib import Path


def build_cache(
    run_id: str,
    probe_names: list[str],
    output_dir: str = "outputs",
    cache_dir: str = "cache",
) -> None:
    in_dir = Path(output_dir) / run_id
    pt_files = sorted(in_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {in_dir}")

    print(f"Found {len(pt_files)} output files", flush=True)
    first = torch.load(pt_files[0], weights_only=False)
    probe_layer_indices = list(first["activations"].keys())
    del first
    print(f"Probe layers: {probe_layer_indices}", flush=True)

    for probe_name in probe_names:
        out_dir = Path(cache_dir) / run_id / probe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # accumulators: one list per layer, holding (T, hidden_dim) float16 tensors
        H_acc: dict[int, list[torch.Tensor]] = {l: [] for l in probe_layer_indices}
        y_acc: dict[int, list] = {l: [] for l in probe_layer_indices}
        rel_acc: dict[int, list] = {l: [] for l in probe_layer_indices}
        sid_acc: dict[int, list] = {l: [] for l in probe_layer_indices}
        gid_acc: dict[int, list] = {l: [] for l in probe_layer_indices}

        for fi, f in enumerate(pt_files):
            if fi % 500 == 0:
                print(f"  [{probe_name}] Loading file {fi}/{len(pt_files)}: {f.name}", flush=True)
            data = torch.load(f, weights_only=False)
            label = data["labels"].get(probe_name)
            sample = data["sample_id"]
            group = data["group_id"]

            for layer_idx in probe_layer_indices:
                acts = data["activations"][layer_idx]  # [T, hidden_dim] float16
                T = acts.shape[0]
                if T == 0:
                    continue

                rel_pos = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)

                if isinstance(label, list):
                    y_vals = []
                    for t in range(T):
                        lval = label[t] if t < len(label) else None
                        y_vals.append(-1 if lval is None else (1 if lval else 0))
                else:
                    y_int = -1 if label is None else (1 if label else 0)
                    y_vals = [y_int] * T

                H_acc[layer_idx].append(acts)
                y_acc[layer_idx].extend(y_vals)
                rel_acc[layer_idx].extend(rel_pos.tolist())
                sid_acc[layer_idx].extend([sample] * T)
                gid_acc[layer_idx].extend([group] * T)

            del data

        for layer_idx in probe_layer_indices:
            if not H_acc[layer_idx]:
                continue
            print(f"  [{probe_name}] Saving layer {layer_idx} ({len(H_acc[layer_idx])} chunks)...", flush=True)
            cache = {
                "H": torch.cat(H_acc[layer_idx], dim=0),  # float16
                "y": torch.tensor(y_acc[layer_idx], dtype=torch.int64),
                "rel_pos": torch.tensor(rel_acc[layer_idx], dtype=torch.float32),
                "sample_id": sid_acc[layer_idx],
                "group_id": gid_acc[layer_idx],
            }
            torch.save(cache, out_dir / f"layer_{layer_idx}.pt")
            del cache, H_acc[layer_idx], y_acc[layer_idx], rel_acc[layer_idx]
            del sid_acc[layer_idx], gid_acc[layer_idx]
            H_acc[layer_idx] = []
