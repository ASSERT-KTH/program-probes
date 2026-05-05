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
    def labels_fn(i):
        return {"currently_correct": [True, False, None, True, False]}

    _setup_pt_files(tmp_path, labels_fn)
    build_cache(RUN_ID, ["currently_correct"], output_dir=str(tmp_path / "outputs"), cache_dir=str(tmp_path / "cache"))

    cache = torch.load(tmp_path / "cache" / RUN_ID / "currently_correct" / f"layer_{PROBE_LAYERS[0]}.pt", weights_only=False)
    y = cache["y"]
    assert cache["H"].shape[0] == 6 * N_STEPS
    # None → -1
    assert -1 in y.tolist()


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
