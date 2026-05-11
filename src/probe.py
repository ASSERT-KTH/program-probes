import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Callable


@dataclass
class ProbeResult:
    layer: int
    bin_idx: int
    val_acc: float
    val_f1: float
    val_precision: float
    val_recall: float
    val_auc: float
    test_acc: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_auc: float
    n_train: int
    n_val: int
    n_test: int
    n_epochs: int


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


def _clf_metrics(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    return {
        "f1": f1_score(labels, preds, zero_division=0.0),
        "precision": precision_score(labels, preds, zero_division=0.0),
        "recall": recall_score(labels, preds, zero_division=0.0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
    }


def train_probe_layer(
    cache_path: str,
    layer_idx: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    seed: int,
    n_bins: int = 10,
    log_fn: Callable[[dict], None] | None = None,
) -> list[ProbeResult]:
    _set_seeds(seed)
    data = torch.load(cache_path, weights_only=False)
    H = data["H"].float()
    y = data["y"]
    rel_pos = data["rel_pos"]
    sample_ids = data["sample_id"]
    group_ids = data["group_id"]

    train_samples, val_samples, test_samples = _split_groups(sample_ids, seed)

    hidden_dim = H.shape[1]
    criterion = nn.CrossEntropyLoss()
    results = []

    for bin_idx in range(n_bins):
        bin_mask = torch.tensor([
            _bin_index(rel_pos[i].item(), n_bins) == bin_idx and y[i].item() >= 0
            for i in range(len(y))
        ])
        if bin_mask.sum() == 0:
            continue

        bin_H = H[bin_mask]
        bin_y = y[bin_mask]
        bin_samples = [sample_ids[i] for i in range(len(sample_ids)) if bin_mask[i]]

        train_mask = torch.tensor([s in train_samples for s in bin_samples])
        val_mask = torch.tensor([s in val_samples for s in bin_samples])
        test_mask = torch.tensor([s in test_samples for s in bin_samples])

        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        H_train, y_train = bin_H[train_mask], bin_y[train_mask]
        H_val, y_val = bin_H[val_mask], bin_y[val_mask]
        H_test, y_test = bin_H[test_mask], bin_y[test_mask]

        mean = H_train.mean(dim=0)
        H_train = H_train - mean
        H_val = H_val - mean
        H_test = H_test - mean

        model = nn.Linear(hidden_dim, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        no_improve = 0
        n_epochs = 0

        n_train_samples = H_train.shape[0]
        for epoch in range(1000):
            model.train()
            perm = torch.randperm(n_train_samples)
            epoch_grad_norms = []
            for start in range(0, n_train_samples, batch_size):
                idx = perm[start:start + batch_size]
                logits = model(H_train[idx])
                loss = criterion(logits, y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                epoch_grad_norms.append(model.weight.grad.norm().item())
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(H_val)
                val_loss = criterion(val_logits, y_val).item()
                train_loss = criterion(model(H_train), y_train).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
                weight_norm = model.weight.norm().item()

            n_epochs = epoch + 1

            if log_fn is not None:
                log_fn({
                    "layer": layer_idx,
                    "bin_idx": bin_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "grad_norm": float(np.mean(epoch_grad_norms)),
                    "weight_norm": weight_norm,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(H_val), dim=1)[:, 1].numpy()
            val_preds_np = (val_probs >= 0.5).astype(int)
            val_labels_np = y_val.numpy()

            test_logits = model(H_test)
            test_probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
            test_preds_np = (test_probs >= 0.5).astype(int)
            test_labels_np = y_test.numpy()

        val_m = _clf_metrics(val_probs, val_preds_np, val_labels_np)
        test_m = _clf_metrics(test_probs, test_preds_np, test_labels_np)
        val_acc = (val_preds_np == val_labels_np).mean()
        test_acc = (test_preds_np == test_labels_np).mean()

        results.append(ProbeResult(
            layer=layer_idx,
            bin_idx=bin_idx,
            val_acc=float(val_acc),
            val_f1=val_m["f1"],
            val_precision=val_m["precision"],
            val_recall=val_m["recall"],
            val_auc=val_m["auc"],
            test_acc=float(test_acc),
            test_f1=test_m["f1"],
            test_precision=test_m["precision"],
            test_recall=test_m["recall"],
            test_auc=test_m["auc"],
            n_train=H_train.shape[0],
            n_val=H_val.shape[0],
            n_test=H_test.shape[0],
            n_epochs=n_epochs,
        ))

    return results


def create_sweep() -> str:
    import wandb
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "mean_val_f1", "goal": "maximize"},
        "parameters": {
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
            "batch_size": {"values": [256, 512, 1024]},
            "patience": {"values": [10, 100]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="program-probes")
    print(f"Created sweep: {sweep_id}", flush=True)
    return sweep_id


def run_sweep(
    run_id: str,
    probe_name: str,
    probe_layers: list[int],
    seed: int,
    sweep_id: str | None = None,
    count: int | None = None,
    cache_dir: str = "cache",
    n_bins: int = 10,
) -> None:
    import wandb
    cache_base = Path(cache_dir) / run_id / probe_name

    middle_layer = probe_layers[len(probe_layers) // 2]
    middle_cache = str(cache_base / f"layer_{middle_layer}.pt")

    if sweep_id is None:
        sweep_id = create_sweep()

    def sweep_fn():
        with wandb.init() as run:
            cfg = run.config

            def log_fn(metrics: dict) -> None:
                b = metrics["bin_idx"]
                wandb.log({
                    f"bin_{b}/train_loss": metrics["train_loss"],
                    f"bin_{b}/val_loss": metrics["val_loss"],
                    f"bin_{b}/val_acc": metrics["val_acc"],
                    f"bin_{b}/grad_norm": metrics["grad_norm"],
                    f"bin_{b}/weight_norm": metrics["weight_norm"],
                })

            results = train_probe_layer(
                middle_cache, middle_layer,
                lr=cfg.lr, weight_decay=cfg.weight_decay,
                batch_size=cfg.batch_size, patience=cfg.patience,
                seed=seed, n_bins=n_bins,
                log_fn=log_fn,
            )
            wandb.log({
                "mean_val_f1": np.mean([r.val_f1 for r in results]) if results else 0.0,
                "mean_val_acc": np.mean([r.val_acc for r in results]) if results else 0.0,
                "mean_val_auc": np.mean([r.val_auc for r in results]) if results else 0.5,
            })

    wandb.agent(sweep_id, sweep_fn, project="program-probes", count=count)


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

            def make_log_fn(layer):
                def log_fn(metrics: dict) -> None:
                    b = metrics["bin_idx"]
                    wandb.log({
                        f"layer_{layer}/bin_{b}/train_loss": metrics["train_loss"],
                        f"layer_{layer}/bin_{b}/val_loss": metrics["val_loss"],
                        f"layer_{layer}/bin_{b}/val_acc": metrics["val_acc"],
                        f"layer_{layer}/bin_{b}/grad_norm": metrics["grad_norm"],
                        f"layer_{layer}/bin_{b}/weight_norm": metrics["weight_norm"],
                    })
                return log_fn

            results = train_probe_layer(
                cache_path, layer_idx,
                lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, patience=patience,
                seed=seed, n_bins=n_bins,
                log_fn=make_log_fn(layer_idx),
            )
            all_results[layer_idx] = results
            for r in results:
                wandb.log({
                    f"layer_{layer_idx}/bin_{r.bin_idx}/test_acc": r.test_acc,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/test_f1": r.test_f1,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/test_precision": r.test_precision,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/test_recall": r.test_recall,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/test_auc": r.test_auc,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/val_f1": r.val_f1,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/val_auc": r.val_auc,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/n_epochs": r.n_epochs,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/n_train": r.n_train,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/n_val": r.n_val,
                    f"layer_{layer_idx}/bin_{r.bin_idx}/n_test": r.n_test,
                })

    out = Path(results_dir) / run_id / probe_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, out / "results.pt")
    return {layer: [vars(r) for r in rs] for layer, rs in all_results.items()}
