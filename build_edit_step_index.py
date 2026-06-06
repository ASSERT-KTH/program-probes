"""Build an edit-step index from label JSON files.

For each trajectory, records which assistant-turn indices are the "first turn
after a code edit" (edit_turn = cmd_idx + 1 for real edits with cmd_idx >= 0).
The baseline edit (cmd_idx=-1) is excluded — it represents the clean repo state
before any model edit.

Output: cache/<run_id>/edit_step_index.pt
    dict[str, list[int]]  — sample_id → sorted list of edit-step turn indices
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_edit_step_index(label_dir: Path, output_path: Path) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    label_files = sorted(label_dir.glob("*_labels.json"))
    if not label_files:
        raise FileNotFoundError(f"No *_labels.json files found in {label_dir}")

    for p in label_files:
        data = json.loads(p.read_text())
        sample_id = p.stem.removesuffix("_labels")
        edits = data.get("edits", [])
        edit_turns = sorted({e["cmd_idx"] + 1 for e in edits if e.get("cmd_idx", -1) >= 0})
        index[sample_id] = edit_turns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, output_path)

    n_with_edits = sum(1 for v in index.values() if v)
    print(f"Indexed {len(index)} samples, {n_with_edits} with at least one edit")
    print(f"Saved to {output_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", required=True,
                        help="Directory containing *_labels.json files")
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    args = parser.parse_args()
    build_edit_step_index(Path(args.label_dir), Path(args.output))


if __name__ == "__main__":
    main()
