"""Load SWE-bench trajectories and build extraction inputs."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SWEBenchTrajectory:
    instance_id: str
    outcome: bool
    token_ids: list[int]
    extraction_mask: list[int]   # 1 at assistant-token positions, 0 elsewhere
    # Maps captured-step index → assistant segment index (for future use)
    step_segment_indices: list[int]


def load_trajectories(traj_dir: str | Path) -> list[SWEBenchTrajectory]:
    """Load all trajectory JSON files from *traj_dir*."""
    traj_dir = Path(traj_dir)
    trajs: list[SWEBenchTrajectory] = []

    for path in sorted(traj_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        tokenization = data.get("tokenization", {})
        token_ids: list[int] = tokenization.get("token_ids", [])
        segments: list[dict] = tokenization.get("segments", [])

        if not token_ids or not segments:
            continue

        outcome: bool = bool(data.get("metadata", {}).get("outcome", False))
        instance_id: str = data.get("metadata", {}).get("instance_id", path.stem)

        # Build extraction mask: 1 for every token that belongs to an assistant segment
        mask = [0] * len(token_ids)
        step_segment_indices: list[int] = []
        for seg_idx, seg in enumerate(segments):
            if seg.get("role") != "assistant":
                continue
            start, end = seg["start_token"], seg["end_token"]
            for pos in range(start, min(end, len(token_ids))):
                mask[pos] = 1
            step_segment_indices.append(seg_idx)

        trajs.append(SWEBenchTrajectory(
            instance_id=instance_id,
            outcome=outcome,
            token_ids=token_ids,
            extraction_mask=mask,
            step_segment_indices=step_segment_indices,
        ))

    return trajs
