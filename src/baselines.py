import torch
from dataclasses import dataclass

from src.probe import _split_groups, _bin_index, _clf_metrics


@dataclass
class BaselineResult:
    bin_idx: int
    test_auc: float
    test_acc: float
    n_test: int
    n_pos_test: int


def compute_persistence_baseline(
    cache_path: str,
    seed: int,
    n_eval_bins: int = 10,
    eval_bin_axis: str = "position",
) -> list[BaselineResult]:
    """Naive persistence baseline: predict y_{t+k} = y_t.

    Reads the same cache used by the real probes and uses the un-shifted
    label ``y_original`` (the label at t) as a hard prediction for the
    shifted label ``y`` (the label at t+k), scored on the same test split
    the probes are evaluated on.
    """
    data = torch.load(cache_path, weights_only=False)
    if "y_original" not in data:
        raise ValueError(
            f"{cache_path} has no 'y_original' field — persistence baseline "
            "requires a cache built with label_shift > 0."
        )

    y = data["y"]
    y_original = data["y_original"]
    rel_pos = data["rel_pos"]
    step_idx = data.get("step_idx")
    sample_ids = data["sample_id"]
    group_ids = data["group_id"]

    train_groups, val_groups, test_groups = _split_groups(group_ids, seed)
    test_samples = {sid for sid, gid in zip(sample_ids, group_ids) if gid in test_groups}

    valid = (y >= 0) & (y_original >= 0)
    test_mask = valid & torch.tensor(
        [sid in test_samples for sid in sample_ids], dtype=torch.bool
    )
    if not test_mask.any():
        return []

    test_idx = test_mask.nonzero(as_tuple=True)[0]
    test_y = y[test_idx]
    test_probs = y_original[test_idx].float()
    test_rel_pos = rel_pos[test_idx]
    test_step_idx = step_idx[test_idx] if step_idx is not None else None
    test_sample_ids = [sample_ids[i] for i in test_idx.tolist()]

    sample_max_step: dict[str, int] = {}
    if test_step_idx is not None:
        for sid, s in zip(test_sample_ids, test_step_idx.tolist()):
            if s > sample_max_step.get(sid, 0):
                sample_max_step[sid] = s

    results = []
    for eb in range(n_eval_bins):
        if eval_bin_axis == "step_relative":
            if test_step_idx is None:
                raise ValueError("eval_bin_axis='step_relative' requires step_idx in cache")
            bin_mask = torch.tensor([
                _bin_index(s.item() / max(sample_max_step.get(sid, 1), 1), n_eval_bins) == eb
                for sid, s in zip(test_sample_ids, test_step_idx)
            ])
        else:  # position
            bin_mask = torch.tensor([_bin_index(p.item(), n_eval_bins) == eb for p in test_rel_pos])

        if not bin_mask.any():
            continue

        probs = test_probs[bin_mask].numpy()
        labels = test_y[bin_mask].numpy()
        preds = (probs >= 0.5).astype(int)

        m = _clf_metrics(probs, preds, labels)
        results.append(BaselineResult(
            bin_idx=eb,
            test_auc=m["auc"],
            test_acc=float((preds == labels).mean()),
            n_test=len(labels),
            n_pos_test=int((labels == 1).sum()),
        ))

    return results
