import pytest
import torch
from pathlib import Path
from src.build_cache import build_cache
from tests.helpers import make_synthetic_pt


PROBE_LAYERS = [0, 1]
HIDDEN_DIM = 16
N_STEPS = 5
RUN_ID = "cache_test"


def _setup_pt_files(tmp_path, labels_fn):
    for i in range(6):
        make_synthetic_pt(
            tmp_path=tmp_path,
            run_id=RUN_ID,
            sample_id=f"s{i}",
            gen_idx=0,
            probe_layers=PROBE_LAYERS,
            hidden_dim=HIDDEN_DIM,
            n_steps=N_STEPS,
            labels=labels_fn(i),
        )


def _write_label_pt(tmp_path, run_id, sample_id, probe, pos_labels, step_idx, turn_labels):
    out = tmp_path / "outputs" / run_id / f"{sample_id}_gen0_labels.pt"
    torch.save({"labels": {probe: pos_labels}, "step_idx": step_idx,
                "turn_labels": {probe: turn_labels}}, out)



def test_static_probe_broadcast(tmp_path):
    def labels_fn(i):
        return {"will_be_correct": i % 2 == 0}

    _setup_pt_files(tmp_path, labels_fn)
    build_cache(RUN_ID, ["will_be_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))

    cache = torch.load(tmp_path / "cache" / RUN_ID / "will_be_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    # 6 samples × N_STEPS each
    assert cache["H"].shape[0] == 6 * N_STEPS
    assert cache["y"].shape[0] == 6 * N_STEPS
    assert cache["rel_pos"].shape[0] == 6 * N_STEPS
    assert len(cache["sample_id"]) == 6 * N_STEPS
    assert len(cache["group_id"]) == 6 * N_STEPS


def test_static_probe_y_values(tmp_path):
    def labels_fn(i):
        return {"will_be_correct": True if i < 3 else False}

    _setup_pt_files(tmp_path, labels_fn)
    build_cache(RUN_ID, ["will_be_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))

    cache = torch.load(tmp_path / "cache" / RUN_ID / "will_be_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    y = cache["y"]
    assert set(y.tolist()).issubset({0, 1})


def test_dynamic_probe_per_step(tmp_path):
    pos_labels = [True, False, None, True, False]
    step_idx = [0, 1, 2, 3, 4]

    def labels_fn(i):
        return {}

    _setup_pt_files(tmp_path, labels_fn)
    for i in range(6):
        _write_label_pt(tmp_path, RUN_ID, f"s{i}", "currently_correct", pos_labels, step_idx, pos_labels)
    build_cache(RUN_ID, ["currently_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))

    cache = torch.load(tmp_path / "cache" / RUN_ID / "currently_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    y = cache["y"]
    assert cache["H"].shape[0] == 6 * N_STEPS
    # None → -1
    assert -1 in y.tolist()


def test_dynamic_probe_without_turns_raises(tmp_path):
    """Dynamic labels without per-position turn indices are rejected, not approximated."""
    def labels_fn(i):
        return {"currently_correct": [True, False, None, True, False]}

    _setup_pt_files(tmp_path, labels_fn)
    with pytest.raises(ValueError, match="turn indices"):
        build_cache(RUN_ID, ["currently_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))


def test_none_becomes_minus_one(tmp_path):
    def labels_fn(i):
        return {"will_be_correct": None}

    _setup_pt_files(tmp_path, labels_fn)
    build_cache(RUN_ID, ["will_be_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))

    cache = torch.load(tmp_path / "cache" / RUN_ID / "will_be_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    assert all(v == -1 for v in cache["y"].tolist())


def test_total_N_equals_sum_T(tmp_path):
    n_samples = 4
    n_steps_per = 7
    for i in range(n_samples):
        make_synthetic_pt(
            tmp_path=tmp_path,
            run_id=RUN_ID + "_n",
            sample_id=f"s{i}",
            gen_idx=0,
            probe_layers=PROBE_LAYERS,
            hidden_dim=HIDDEN_DIM,
            n_steps=n_steps_per,
            labels={"will_be_correct": True},
        )

    build_cache(RUN_ID + "_n", ["will_be_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))
    cache = torch.load(tmp_path / "cache" / (RUN_ID + "_n") / "will_be_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    assert cache["H"].shape[0] == n_samples * n_steps_per


def test_real_step_idx_used(tmp_path):
    """step_idx comes from the label file, not a uniform split of positions."""
    run_id = "real_steps"
    # 6 positions: turn 0 is long (4 positions), turns 1 and 2 are short.
    step_idx = [0, 0, 0, 0, 1, 2]
    turn_labels = [False, True, True]
    pos_labels = [turn_labels[s] for s in step_idx]
    make_synthetic_pt(tmp_path, run_id, "s0", 0, PROBE_LAYERS, HIDDEN_DIM, len(step_idx),
                      labels={"currently_correct": pos_labels})
    _write_label_pt(tmp_path, run_id, "s0", "currently_correct", pos_labels, step_idx, turn_labels)

    build_cache(run_id, ["currently_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))
    cache = torch.load(tmp_path / "cache" / run_id / "currently_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    assert cache["step_idx"].tolist() == step_idx
    assert cache["target_step_idx"].tolist() == step_idx
    assert cache["y"].tolist() == [0, 0, 0, 0, 1, 1]


def test_label_shift_uses_real_turns(tmp_path):
    """With label_shift=k, y is the label of real turn step+k; target_step_idx = step+k."""
    run_id = "shift_steps"
    # 4 turns; turn 2 has no extracted position but still has a label.
    step_idx = [0, 0, 0, 1, 3]
    turn_labels = [False, False, True, False]
    pos_labels = [turn_labels[s] for s in step_idx]
    make_synthetic_pt(tmp_path, run_id, "s0", 0, PROBE_LAYERS, HIDDEN_DIM, len(step_idx),
                      labels={"currently_correct": pos_labels})
    _write_label_pt(tmp_path, run_id, "s0", "currently_correct", pos_labels, step_idx, turn_labels)

    build_cache(run_id, ["currently_correct"], output_dir=str(tmp_path / "outputs"),
                cache_dir=str(tmp_path / "cache"), label_shift=1, max_label_shift=1)
    cache = torch.load(tmp_path / "cache" / run_id / "currently_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    # Positions at turn 3 are dropped (3 + 1 >= 4 turns).
    assert cache["step_idx"].tolist() == [0, 0, 0, 1]
    assert cache["target_step_idx"].tolist() == [1, 1, 1, 2]
    # Turn 1 → False, turn 2 → True (read from turn_labels even though turn 2 has no position).
    assert cache["y"].tolist() == [0, 0, 0, 1]
    assert cache["y_original"].tolist() == [0, 0, 0, 0]
