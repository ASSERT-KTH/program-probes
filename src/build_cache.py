import math
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

    # Collect all samples to determine probe layers and structure
    all_data = [torch.load(f, weights_only=False) for f in pt_files]
    probe_layer_indices = list(all_data[0]["activations"].keys())

    for probe_name in probe_names:
        out_dir = Path(cache_dir) / run_id / probe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in probe_layer_indices:
            H_list, y_list, rel_pos_list, sample_id_list, group_id_list = [], [], [], [], []

            for data in all_data:
                acts = data["activations"][layer_idx]  # [T, hidden_dim]
                T = acts.shape[0]
                if T == 0:
                    continue

                label = data["labels"].get(probe_name)
                sample = data["sample_id"]
                group = data["group_id"]

                if isinstance(label, list):
                    # dynamic probe: per-step labels
                    assert len(label) == data["n_captured_steps"]
                    for t in range(T):
                        lval = label[t] if t < len(label) else None
                        y_int = -1 if lval is None else (1 if lval else 0)
                        rel = t / max(T - 1, 1)
                        H_list.append(acts[t].float().unsqueeze(0))
                        y_list.append(y_int)
                        rel_pos_list.append(rel)
                        sample_id_list.append(sample)
                        group_id_list.append(group)
                else:
                    # static probe: broadcast scalar to all positions
                    y_int = -1 if label is None else (1 if label else 0)
                    for t in range(T):
                        rel = t / max(T - 1, 1)
                        H_list.append(acts[t].float().unsqueeze(0))
                        y_list.append(y_int)
                        rel_pos_list.append(rel)
                        sample_id_list.append(sample)
                        group_id_list.append(group)

            if not H_list:
                continue

            cache = {
                "H": torch.cat(H_list, dim=0),
                "y": torch.tensor(y_list, dtype=torch.int64),
                "rel_pos": torch.tensor(rel_pos_list, dtype=torch.float32),
                "sample_id": sample_id_list,
                "group_id": group_id_list,
            }
            torch.save(cache, out_dir / f"layer_{layer_idx}.pt")
