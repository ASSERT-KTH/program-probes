import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProbeResult:
    layer: int
    bin_idx: int
    val_acc: float
    test_acc: float
    n_train: int
    n_val: int
    n_test: int


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_groups(group_ids: list[str], seed: int) -> tuple[set, set, set]:
    unique = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = math.floor(0.70 * n)
    n_val = math.floor(0.15 * n)
    train_groups = set(unique[:n_train])
    val_groups = set(unique[n_train:n_train + n_val])
    test_groups = set(unique[n_train + n_val:])
    return train_groups, val_groups, test_groups


def _bin_index(rel_pos: float, n_bins: int = 10) -> int:
    return min(int(math.floor(rel_pos * n_bins)), n_bins - 1)


def train_probe_layer(
    cache_path: str,
    layer_idx: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    seed: int,
    n_bins: int = 10,
) -> list[ProbeResult]:
    _set_seeds(seed)
    data = torch.load(cache_path, weights_only=False)
    H = data["H"]
    y = data["y"]
    rel_pos = data["rel_pos"]
    group_ids = data["group_id"]

    train_groups, val_groups, test_groups = _split_groups(group_ids, seed)

    hidden_dim = H.shape[1]
    results = []

    for bin_idx in range(n_bins):
        # Gather indices for this bin, excluding masked positions
        bin_mask = torch.tensor([
            _bin_index(rel_pos[i].item(), n_bins) == bin_idx and y[i].item() >= 0
            for i in range(len(y))
        ])
        if bin_mask.sum() == 0:
            continue

        bin_H = H[bin_mask]
        bin_y = y[bin_mask]
        bin_groups = [group_ids[i] for i in range(len(group_ids)) if bin_mask[i]]

        train_mask = torch.tensor([g in train_groups for g in bin_groups])
        val_mask = torch.tensor([g in val_groups for g in bin_groups])
        test_mask = torch.tensor([g in test_groups for g in bin_groups])

        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        H_train, y_train = bin_H[train_mask], bin_y[train_mask]
        H_val, y_val = bin_H[val_mask], bin_y[val_mask]
        H_test, y_test = bin_H[test_mask], bin_y[test_mask]

        # Mean-center using train statistics only
        mean = H_train.mean(dim=0)
        H_train = H_train - mean
        H_val = H_val - mean
        H_test = H_test - mean

        model = nn.Linear(hidden_dim, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_weights = copy.deepcopy(model.state_dict())
        no_improve = 0

        n_train_samples = H_train.shape[0]
        for epoch in range(1000):
            model.train()
            perm = torch.randperm(n_train_samples)
            for start in range(0, n_train_samples, batch_size):
                idx = perm[start:start + batch_size]
                logits = model(H_train[idx])
                loss = criterion(logits, y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(H_val)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            test_preds = model(H_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        results.append(ProbeResult(
            layer=layer_idx,
            bin_idx=bin_idx,
            val_acc=best_val_acc,
            test_acc=test_acc,
            n_train=H_train.shape[0],
            n_val=H_val.shape[0],
            n_test=H_test.shape[0],
        ))

    return results


def run_sweep(
    run_id: str,
    probe_name: str,
    probe_layers: list[int],
    seed: int,
    cache_dir: str = "cache",
    n_bins: int = 10,
) -> None:
    import wandb
    cache_base = Path(cache_dir) / run_id / probe_name

    middle_layer = probe_layers[len(probe_layers) // 2]
    middle_cache = str(cache_base / f"layer_{middle_layer}.pt")

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "mean_val_acc", "goal": "maximize"},
        "parameters": {
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
            "batch_size": {"values": [256, 512, 1024]},
            "patience": {"values": [5, 10, 20]},
        },
    }

    def sweep_fn():
        with wandb.init() as run:
            cfg = run.config
            results = train_probe_layer(
                middle_cache, middle_layer,
                lr=cfg.lr, weight_decay=cfg.weight_decay,
                batch_size=cfg.batch_size, patience=cfg.patience,
                seed=seed, n_bins=n_bins,
            )
            mean_val = np.mean([r.val_acc for r in results]) if results else 0.0
            wandb.log({"mean_val_acc": mean_val})

    sweep_id = wandb.sweep(sweep_config, project="program-probes")
    wandb.agent(sweep_id, sweep_fn)


def run_final(
    run_id: str,
    probe_name: str,
    probe_layers: list[int],
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    seed: int,
    cache_dir: str = "cache",
    results_dir: str = "results",
    n_bins: int = 10,
) -> dict:
    import wandb
    cache_base = Path(cache_dir) / run_id / probe_name
    all_results: dict[int, list[ProbeResult]] = {}

    with wandb.init(project="program-probes", job_type="final") as run:
        for layer_idx in probe_layers:
            cache_path = str(cache_base / f"layer_{layer_idx}.pt")
            results = train_probe_layer(
                cache_path, layer_idx,
                lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, patience=patience,
                seed=seed, n_bins=n_bins,
            )
            all_results[layer_idx] = results
            for r in results:
                wandb.log({
                    f"layer_{layer_idx}/bin_{r.bin_idx}/test_acc": r.test_acc,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/val_acc": r.val_acc,
                })

    out = Path(results_dir) / run_id / probe_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, out / "results.pt")
    return {layer: [vars(r) for r in rs] for layer, rs in all_results.items()}
