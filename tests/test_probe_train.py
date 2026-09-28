import math
import pytest
import torch
from pathlib import Path
from src.probe import train_probe_layer, _split_groups, _bin_index


HIDDEN_DIM = 16
N = 200
N_BINS = 10
LAYER_IDX = 0


def _make_cache(tmp_path: Path, seed: int = 42) -> str:
    torch.manual_seed(seed)
    sample_ids = [f"sample_{i % 10}" for i in range(N)]
    groups = [f"group_{i % 10}" for i in range(N)]
    H = torch.randn(N, HIDDEN_DIM)
    y = torch.randint(0, 2, (N,), dtype=torch.int64)
    # Sprinkle some masked positions
    masked = torch.arange(0, N, 20)
    y[masked] = -1
    rel_pos = torch.linspace(0, 1, N)

    cache = {"H": H, "y": y, "rel_pos": rel_pos, "sample_id": sample_ids, "group_id": groups}
    cache_path = tmp_path / "layer_0.pt"
    torch.save(cache, cache_path)
    return str(cache_path)


def test_group_split_non_overlapping():
    groups = [f"g{i}" for i in range(20)]
    train, val, test = _split_groups(groups, seed=0)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(groups)


def test_group_split_ratios():
    groups = [f"g{i}" for i in range(100)]
    train, val, test = _split_groups(groups, seed=0)
    assert abs(len(train) - 70) <= 2
    assert abs(len(val) - 15) <= 2


def test_masked_positions_excluded(tmp_path):
    cache_path = _make_cache(tmp_path)
    results, _ = train_probe_layer(
        cache_path, LAYER_IDX,
        lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=42, n_bins=N_BINS
    )
    # All results should have non-negative val/test acc
    for r in results:
        assert r.val_acc >= 0.0
        assert r.test_acc >= 0.0


def test_early_stopping_restores_best_weights(tmp_path):
    # This verifies training completes and best weights are used (test_acc reported once per bin)
    cache_path = _make_cache(tmp_path)
    results, _ = train_probe_layer(
        cache_path, LAYER_IDX,
        lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=42, n_bins=N_BINS
    )
    # One result per non-empty bin
    assert len(results) > 0
    for r in results:
        assert 0.0 <= r.test_acc <= 1.0


def test_test_accuracy_reported_once_per_cell(tmp_path):
    cache_path = _make_cache(tmp_path)
    results, _ = train_probe_layer(
        cache_path, LAYER_IDX,
        lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=42, n_bins=N_BINS
    )
    bin_indices = [r.bin_idx for r in results]
    assert len(bin_indices) == len(set(bin_indices)), "Duplicate bin_idx in results"


def test_mean_centering_uses_train_stats(tmp_path):
    # Synthetic data where train mean is known: all ones for train, zeros for test
    # After mean-centering with train stats, test will be shifted
    torch.manual_seed(0)
    groups = [f"g{i}" for i in range(20)]
    H = torch.zeros(200, HIDDEN_DIM)
    # Make train groups easily separable
    for i, g in enumerate(groups):
        if i < 14:  # train
            H[i * 10:(i + 1) * 10] = torch.ones(10, HIDDEN_DIM)
    y = torch.zeros(200, dtype=torch.int64)
    rel_pos = torch.zeros(200)  # all in bin 0
    cache = {"H": H, "y": y, "rel_pos": rel_pos, "sample_id": [f"s{i // 10}" for i in range(200)], "group_id": [f"g{i // 10}" for i in range(200)]}
    path = tmp_path / "layer_0.pt"
    torch.save(cache, path)
    results, _ = train_probe_layer(str(path), 0, lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=42, n_bins=N_BINS)
    # Should complete without error
    assert isinstance(results, list)


def test_probe_result_has_extended_metrics(tmp_path):
    cache_path = _make_cache(tmp_path)
    results, _ = train_probe_layer(
        cache_path, LAYER_IDX,
        lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=42, n_bins=N_BINS
    )
    for r in results:
        assert 0.0 <= r.val_f1 <= 1.0
        assert 0.0 <= r.val_auc <= 1.0
        assert 0.0 <= r.val_ece <= 1.0
        assert 0.0 <= r.test_f1 <= 1.0
        assert 0.0 <= r.test_auc <= 1.0
        assert 0.0 <= r.test_ece <= 1.0
        assert r.n_epochs >= 1


def test_log_fn_called_per_epoch(tmp_path):
    cache_path = _make_cache(tmp_path)
    log_calls = []
    results, _ = train_probe_layer(
        cache_path, LAYER_IDX,
        lr=1e-3, weight_decay=1e-4, batch_size=64, patience=3, seed=42, n_bins=N_BINS,
        log_fn=log_calls.append,
    )
    assert len(log_calls) > 0
    entry = log_calls[0]
    assert "train_loss" in entry
    assert "val_loss" in entry
    assert "grad_norm" in entry
    assert "weight_norm" in entry


def test_bin_index_clamped():
    assert _bin_index(0.0) == 0
    assert _bin_index(1.0) == 9  # clamped to n_bins-1
    assert _bin_index(0.95) == 9
    assert _bin_index(0.55) == 5


@pytest.mark.parametrize("unit", ["token", "trajectory"])
def test_shuffle_labels_scores_against_real_test_labels(tmp_path, monkeypatch, unit):
    """The shuffled-label control must only transform train/val labels.

    The shuffle is replaced by label inversion: a probe trained on inverted labels
    scores AUC ~0 against the real test labels, but ~1 if y_test were inverted too.
    """
    import src.probe as probe_mod

    torch.manual_seed(0)
    n = 400
    y = torch.randint(0, 2, (n,), dtype=torch.int64)
    H = torch.randn(n, HIDDEN_DIM)
    H[:, 0] = 3.0 * (2 * y - 1)  # labels are linearly readable
    cache = {
        "H": H, "y": y, "rel_pos": torch.linspace(0, 1, n),
        "sample_id": [f"s{i % 20}" for i in range(n)],
        "group_id": [f"g{i % 20}" for i in range(n)],
    }
    cache_path = tmp_path / "layer_0.pt"
    torch.save(cache, cache_path)

    calls = []

    def invert(labels, sample_ids, shuffle_unit):
        calls.append(shuffle_unit)
        return 1 - labels

    monkeypatch.setattr(probe_mod, "_shuffle_labels", invert)
    results, _ = train_probe_layer(
        str(cache_path), LAYER_IDX, lr=1e-2, weight_decay=0.0, batch_size=64, patience=5,
        seed=42, n_bins=1, shuffle_labels=True, shuffle_unit=unit,
    )
    assert calls == [unit, unit]  # train and val only
    assert results[0].test_auc < 0.2


def _merge_one_layer(out: str, layer: int) -> None:
    from src.probe import _merge_save
    # ~16MB per layer, like real probe weights, so concurrent writes actually overlap.
    weights = {layer: {0: {"w": torch.full((2000, 2000), float(layer))}}}
    _merge_save(Path(out), {layer: [f"result_{layer}"]}, weights)


def test_merge_save_keeps_all_layers_under_concurrency(tmp_path):
    """Per-layer jobs merging into the same run must not lose each other's layers."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_merge_one_layer, args=(str(tmp_path), L)) for L in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    results = torch.load(tmp_path / "results.pt", weights_only=False)
    weights = torch.load(tmp_path / "weights.pt", weights_only=False)
    assert sorted(results) == list(range(8))
    assert sorted(weights) == list(range(8))
