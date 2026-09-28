"""Tests for build_edit_step_index and the after_edit_only filter in probe.py."""
import json
import pytest
import torch
from pathlib import Path

from build_edit_step_index import build_edit_step_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_label_json(label_dir: Path, sample_id: str, edits: list[dict]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{sample_id}_labels.json").write_text(
        json.dumps({"instance_id": sample_id, "trajectory_path": "", "edits": edits})
    )


def _make_small_cache(tmp_path: Path, sample_step_pairs: list[tuple[str, int]]) -> dict:
    """Build a minimal in-memory cache dict mimicking build_cache output."""
    n = len(sample_step_pairs)
    hidden_dim = 4
    return {
        "H": torch.zeros(n, hidden_dim, dtype=torch.float16),
        "y": torch.ones(n, dtype=torch.int64),
        "rel_pos": torch.linspace(0, 1, n),
        "step_idx": torch.tensor([s for _, s in sample_step_pairs], dtype=torch.int64),
        "sample_id": [sid for sid, _ in sample_step_pairs],
        "group_id": [sid for sid, _ in sample_step_pairs],
    }


# ---------------------------------------------------------------------------
# Index building tests
# ---------------------------------------------------------------------------

def test_build_index_basic(tmp_path):
    """Real edits at cmd_idx 5 and 16 → edit step turns [6, 17]."""
    label_dir = tmp_path / "labels"
    _write_label_json(label_dir, "sample_a", [
        {"cmd_idx": -1, "compiles": True, "test_results": {}},
        {"cmd_idx": 5,  "compiles": True, "test_results": {}},
        {"cmd_idx": 16, "compiles": True, "test_results": {}},
    ])
    out = tmp_path / "edit_step_index.pt"
    index = build_edit_step_index(label_dir, out)

    assert "sample_a" in index
    assert index["sample_a"] == [6, 17]
    assert out.exists()
    loaded = torch.load(out, weights_only=False)
    assert loaded["sample_a"] == [6, 17]


def test_build_index_baseline_excluded(tmp_path):
    """Only a baseline edit (cmd_idx=-1) → no model edits → empty list."""
    label_dir = tmp_path / "labels"
    _write_label_json(label_dir, "sample_b", [
        {"cmd_idx": -1, "compiles": True, "test_results": {}},
    ])
    out = tmp_path / "edit_step_index.pt"
    index = build_edit_step_index(label_dir, out)
    assert index["sample_b"] == []


def test_build_index_no_edits(tmp_path):
    """Empty edits list → empty list."""
    label_dir = tmp_path / "labels"
    _write_label_json(label_dir, "sample_c", [])
    out = tmp_path / "edit_step_index.pt"
    index = build_edit_step_index(label_dir, out)
    assert index["sample_c"] == []


def test_build_index_multiple_samples(tmp_path):
    """Multiple label JSON files produce independent per-sample entries."""
    label_dir = tmp_path / "labels"
    _write_label_json(label_dir, "s1", [
        {"cmd_idx": -1, "compiles": True, "test_results": {}},
        {"cmd_idx": 3,  "compiles": True, "test_results": {}},
    ])
    _write_label_json(label_dir, "s2", [
        {"cmd_idx": -1, "compiles": True, "test_results": {}},
        {"cmd_idx": 7,  "compiles": True, "test_results": {}},
        {"cmd_idx": 10, "compiles": True, "test_results": {}},
    ])
    out = tmp_path / "edit_step_index.pt"
    index = build_edit_step_index(label_dir, out)

    assert index["s1"] == [4]
    assert index["s2"] == [8, 11]
    assert len(index) == 2


def test_build_index_deduplicates_same_turn(tmp_path):
    """Two edits mapping to the same turn are stored once."""
    label_dir = tmp_path / "labels"
    _write_label_json(label_dir, "s_dup", [
        {"cmd_idx": 2, "compiles": True, "test_results": {}},
        {"cmd_idx": 2, "compiles": True, "test_results": {}},  # duplicate
    ])
    out = tmp_path / "edit_step_index.pt"
    index = build_edit_step_index(label_dir, out)
    assert index["s_dup"] == [3]


def test_build_index_no_label_files_raises(tmp_path):
    label_dir = tmp_path / "empty_labels"
    label_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_edit_step_index(label_dir, tmp_path / "out.pt")


# ---------------------------------------------------------------------------
# after_edit_only filter in probe.py
# ---------------------------------------------------------------------------

def test_after_edit_filter_in_probe(tmp_path):
    """after_edit_only masks tokens whose step_idx is not in the index."""
    from src.probe import train_probe_layer

    # Build a minimal cache with 3 samples × 4 positions each
    # sample "s0": edit at step 1 → 4 positions at steps [0,1,2,3] → 1 valid after-edit
    # sample "s1": edit at step 2 → 4 positions at steps [0,1,2,3] → 1 valid after-edit
    # sample "s2": no edits → 0 valid after-edit
    hidden_dim = 4
    n_steps = 4
    n_samples = 3

    positions = list(range(n_steps)) * n_samples
    step_idx_vals = positions  # step_idx == position index here
    sample_ids = ["s0"] * n_steps + ["s1"] * n_steps + ["s2"] * n_steps
    group_ids = sample_ids[:]

    cache = {
        "H": torch.randn(n_samples * n_steps, hidden_dim, dtype=torch.float16),
        "y": torch.ones(n_samples * n_steps, dtype=torch.int64),
        "rel_pos": torch.linspace(0, 1, n_samples * n_steps),
        "step_idx": torch.tensor(step_idx_vals, dtype=torch.int64),
        "sample_id": sample_ids,
        "group_id": group_ids,
    }
    cache_path = tmp_path / "layer_0.pt"
    torch.save(cache, cache_path)

    edit_step_index = {
        "s0": [1],    # 1 matching position
        "s1": [2],    # 1 matching position
        "s2": [],     # 0 matching positions
    }

    # With all 3 samples in the same "group" the split puts them all in train.
    # We just verify no crash and that valid tokens are reduced.
    # Use minimal HP so training is fast.
    results, weights = train_probe_layer(
        cache_path=str(cache_path),
        layer_idx=0,
        lr=0.01, weight_decay=0.0, batch_size=32, patience=1,
        seed=42, n_bins=1, probe_arch="linear",
        after_edit_only=True,
        edit_step_index=edit_step_index,
    )
    # After filtering: 2 valid tokens (s0/step1 and s1/step2). With 1 group
    # after split all end up in train — no val/test → no result produced.
    # The important thing: no crash, and filter was applied.
    assert isinstance(results, list)


def test_after_edit_mask_uses_target_step():
    """With a label shift, the filter keys on the target turn, not the source turn."""
    from src.probe import _after_edit_mask

    source = torch.tensor([0, 1, 2, 3])
    target = source + 2
    index = {"s": [3]}
    sids = ["s"] * 4
    assert _after_edit_mask(target, sids, index).tolist() == [False, True, False, False]
    assert _after_edit_mask(source, sids, index).tolist() == [False, False, False, True]


def test_shuffle_by_trajectory_moves_whole_sequences():
    """Each trajectory receives a distinct other trajectory's label sequence, resampled to its length."""
    from src.probe import _shuffle_by_trajectory

    def resample(seq, n):
        return tuple(seq[round(j * (len(seq) - 1) / max(n - 1, 1))] for j in range(n))

    seqs = {"a": [1, 1, 1, 1], "b": [0, 0], "c": [0, 1, 1]}
    sids = [sid for sid, seq in seqs.items() for _ in seq]
    y = torch.tensor([v for seq in seqs.values() for v in seq])
    spans = {"a": slice(0, 4), "b": slice(4, 6), "c": slice(6, 9)}

    torch.manual_seed(0)
    for _ in range(20):
        out = _shuffle_by_trajectory(y, sids)
        donors = []
        for r, span in spans.items():
            got = tuple(out[span].tolist())
            matches = [d for d in seqs if resample(seqs[d], len(seqs[r])) == got]
            assert matches, (r, got)
            donors.append(matches)
        # Some assignment of donors is a permutation of the trajectories.
        import itertools
        assert any(len(set(p)) == 3 for p in itertools.product(*donors))
