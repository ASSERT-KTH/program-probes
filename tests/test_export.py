import json
import pytest
import torch
from pathlib import Path
from src.export_dashboard import export_dashboard
from tests.helpers import make_synthetic_pt


PROBE_LAYERS = [0, 1]
HIDDEN_DIM = 16
N_STEPS = 4
RUN_ID = "export_test"
PROBE_NAMES = ["will_be_correct"]


def _setup_run(tmp_path: Path, run_id: str = RUN_ID, n_samples: int = 4, n_gen: int = 2):
    for i in range(n_samples):
        for g in range(n_gen):
            make_synthetic_pt(
                tmp_path=tmp_path,
                run_id=run_id,
                sample_id=f"s{i}",
                gen_idx=g,
                probe_layers=PROBE_LAYERS,
                hidden_dim=HIDDEN_DIM,
                n_steps=N_STEPS,
                labels={"will_be_correct": i % 2 == 0},
            )


def test_manifest_updated(tmp_path):
    _setup_run(tmp_path)
    export_dashboard(
        run_id=RUN_ID,
        probe_names=PROBE_NAMES,
        probe_layers=PROBE_LAYERS,
        task_name="mock",
        model_name="mock-model",
        output_dir=str(tmp_path / "outputs"),
        results_dir=str(tmp_path / "results"),
        dashboard_dir=str(tmp_path / "dashboard"),
    )
    manifest = json.loads((tmp_path / "dashboard" / "data" / "manifest.json").read_text())
    run_ids = [e["run_id"] for e in manifest]
    assert RUN_ID in run_ids


def test_samples_json_contains_all(tmp_path):
    n_samples = 4
    n_gen = 2
    _setup_run(tmp_path, n_samples=n_samples, n_gen=n_gen)
    export_dashboard(
        run_id=RUN_ID,
        probe_names=PROBE_NAMES,
        probe_layers=PROBE_LAYERS,
        task_name="mock",
        model_name="mock-model",
        output_dir=str(tmp_path / "outputs"),
        results_dir=str(tmp_path / "results"),
        dashboard_dir=str(tmp_path / "dashboard"),
    )
    samples = json.loads((tmp_path / "dashboard" / "data" / RUN_ID / "samples.json").read_text())
    assert len(samples) == n_samples
    for s in samples:
        assert len(s["generations"]) == n_gen


def test_probe_results_structure(tmp_path):
    _setup_run(tmp_path)

    # Create a fake results file
    from src.probe import ProbeResult
    results_dir = tmp_path / "results" / RUN_ID / "will_be_correct"
    results_dir.mkdir(parents=True)
    fake_results = {
        0: [ProbeResult(layer=0, bin_idx=i, val_acc=0.7, val_f1=0.6, val_precision=0.6, val_recall=0.6, val_auc=0.7, test_acc=0.65, test_f1=0.55, test_precision=0.55, test_recall=0.55, test_auc=0.65, n_train=50, n_val=10, n_test=10, n_epochs=5) for i in range(3)],
        1: [ProbeResult(layer=1, bin_idx=i, val_acc=0.8, val_f1=0.7, val_precision=0.7, val_recall=0.7, val_auc=0.8, test_acc=0.75, test_f1=0.65, test_precision=0.65, test_recall=0.65, test_auc=0.75, n_train=50, n_val=10, n_test=10, n_epochs=8) for i in range(3)],
    }
    torch.save(fake_results, results_dir / "results.pt")

    export_dashboard(
        run_id=RUN_ID,
        probe_names=PROBE_NAMES,
        probe_layers=PROBE_LAYERS,
        task_name="mock",
        model_name="mock-model",
        output_dir=str(tmp_path / "outputs"),
        results_dir=str(tmp_path / "results"),
        dashboard_dir=str(tmp_path / "dashboard"),
    )

    probe_results = json.loads((tmp_path / "dashboard" / "data" / RUN_ID / "probe_results.json").read_text())
    assert "will_be_correct" in probe_results
    for layer_key in ["0", "1"]:
        assert layer_key in probe_results["will_be_correct"]
        for bin_key in ["0", "1", "2"]:
            cell = probe_results["will_be_correct"][layer_key][bin_key]
            assert "test_acc" in cell
            assert "val_acc" in cell
            assert "n_train" in cell


def test_reexport_overwrites(tmp_path):
    _setup_run(tmp_path)

    for _ in range(2):
        export_dashboard(
            run_id=RUN_ID,
            probe_names=PROBE_NAMES,
            probe_layers=PROBE_LAYERS,
            task_name="mock",
            model_name="mock-model",
            output_dir=str(tmp_path / "outputs"),
            results_dir=str(tmp_path / "results"),
            dashboard_dir=str(tmp_path / "dashboard"),
        )

    manifest = json.loads((tmp_path / "dashboard" / "data" / "manifest.json").read_text())
    assert sum(1 for e in manifest if e["run_id"] == RUN_ID) == 1
