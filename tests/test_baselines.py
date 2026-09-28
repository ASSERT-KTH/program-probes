import torch
from pathlib import Path

from src.baselines import compute_persistence_baseline
from src.probe import _split_groups


def _make_cache(tmp_path: Path, seed: int = 0) -> str:
    # 10 groups x 20 tokens; y_original == y for 3/4 of tokens (persistence
    # correct), flipped for the rest, so persistence AUC is well above 0.5
    # but not perfect.
    n_groups = 10
    per_group = 20
    sample_ids, groups, y_list, y_original_list, rel_pos_list, step_list = [], [], [], [], [], []
    for g in range(n_groups):
        for i in range(per_group):
            sample_ids.append(f"sample_{g}")
            groups.append(f"group_{g}")
            y_orig = i % 2
            y = y_orig if i % 4 != 3 else 1 - y_orig
            y_original_list.append(y_orig)
            y_list.append(y)
            rel_pos_list.append(i / (per_group - 1))
            step_list.append(i)

    cache = {
        "y": torch.tensor(y_list, dtype=torch.int64),
        "y_original": torch.tensor(y_original_list, dtype=torch.int64),
        "rel_pos": torch.tensor(rel_pos_list, dtype=torch.float32),
        "step_idx": torch.tensor(step_list, dtype=torch.int64),
        "sample_id": sample_ids,
        "group_id": groups,
    }
    path = tmp_path / "layer_0.pt"
    torch.save(cache, path)
    return str(path)


def test_persistence_baseline_requires_y_original(tmp_path):
    cache = {
        "y": torch.tensor([0, 1]),
        "rel_pos": torch.tensor([0.0, 1.0]),
        "step_idx": torch.tensor([0, 1]),
        "sample_id": ["s0", "s0"],
        "group_id": ["g0", "g0"],
    }
    path = tmp_path / "layer_0.pt"
    torch.save(cache, path)
    try:
        compute_persistence_baseline(str(path), seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_persistence_baseline_scores_test_split_only(tmp_path):
    cache_path = _make_cache(tmp_path)
    results = compute_persistence_baseline(cache_path, seed=0, n_eval_bins=1)
    assert len(results) == 1
    r = results[0]

    data = torch.load(cache_path, weights_only=False)
    _, _, test_groups = _split_groups(data["group_id"], seed=0)
    expected_n = sum(1 for g in data["group_id"] if g in test_groups)
    assert r.n_test == expected_n
    assert r.n_test < len(data["y"])  # only a fraction of tokens is test split
    assert 0.0 <= r.test_auc <= 1.0
    assert 0.0 <= r.test_acc <= 1.0
    # 3/4 of tokens have y == y_original, so persistence accuracy should
    # clearly beat chance.
    assert r.test_acc > 0.6


def test_persistence_baseline_bin_idx_unique(tmp_path):
    cache_path = _make_cache(tmp_path)
    results = compute_persistence_baseline(cache_path, seed=0, n_eval_bins=5)
    bin_indices = [r.bin_idx for r in results]
    assert len(bin_indices) == len(set(bin_indices))
