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

        # Process one layer at a time to cap peak memory at ~2x one layer's size
        for layer_idx in probe_layer_indices:
            print(f"  [{probe_name}] Building layer {layer_idx}...", flush=True)
            H_list, y_list, rel_list, sid_list, gid_list = [], [], [], [], []

            for fi, f in enumerate(pt_files):
                if fi % 500 == 0:
                    print(f"    file {fi}/{len(pt_files)}: {f.name}", flush=True)
                data = torch.load(f, weights_only=False)
                label = data["labels"].get(probe_name)
                sample = data["sample_id"]
                group = data["group_id"]

                acts = data["activations"][layer_idx]  # [T, hidden_dim] float16
                T = acts.shape[0]
                del data

                if T == 0:
                    continue

                rel_pos = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)

                if isinstance(label, list):
                    y_vals = [-1 if (label[t] if t < len(label) else None) is None
                              else (1 if label[t] else 0) for t in range(T)]
                else:
                    y_int = -1 if label is None else (1 if label else 0)
                    y_vals = [y_int] * T

                H_list.append(acts)
                y_list.extend(y_vals)
                rel_list.extend(rel_pos.tolist())
                sid_list.extend([sample] * T)
                gid_list.extend([group] * T)

            if not H_list:
                continue

            print(f"  [{probe_name}] Saving layer {layer_idx}...", flush=True)
            cache = {
                "H": torch.cat(H_list, dim=0),  # float16
                "y": torch.tensor(y_list, dtype=torch.int64),
                "rel_pos": torch.tensor(rel_list, dtype=torch.float32),
                "sample_id": sid_list,
                "group_id": gid_list,
            }
            torch.save(cache, out_dir / f"layer_{layer_idx}.pt")
            del cache, H_list, y_list, rel_list, sid_list, gid_list
