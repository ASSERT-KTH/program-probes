import math
import pytest
import torch
from pathlib import Path
from src.configs import ModelConfig, HardwareConfig, TaskConfig, GenerationConfig
from src.extract import run_extraction
from tests.helpers import MockModelAdapter, MockTaskAdapter


def _make_adapters_patch(monkeypatch, mock_model_adapter, mock_task_adapter):
    monkeypatch.setattr(
        "src.extract._load_model_adapter",
        lambda name: mock_model_adapter,
    )
    monkeypatch.setattr(
        "src.extract._load_task_adapter",
        lambda name: mock_task_adapter,
    )


def test_output_files_created(monkeypatch, tmp_output_dir):
    model_cfg = ModelConfig(model_id="mock", probe_layers=[0, 1], adapter="mock")
    hw_cfg = HardwareConfig(device_map="cpu", dtype="float32")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=3, seed=42)

    adapter = MockModelAdapter()
    adapter.load(model_cfg, hw_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        hardware_config=hw_cfg,
        generation_config=gen_cfg,
        probe_names=["will_be_correct"],
        run_id="test_run",
        output_dir=str(tmp_output_dir / "outputs"),
    )

    out_files = list((tmp_output_dir / "outputs" / "test_run").glob("*.pt"))
    # 20 samples × 1 generation
    assert len(out_files) == 20


def test_output_keys_and_shapes(monkeypatch, tmp_output_dir):
    probe_layers = [0, 1]
    model_cfg = ModelConfig(model_id="mock", probe_layers=probe_layers, adapter="mock")
    hw_cfg = HardwareConfig(device_map="cpu", dtype="float32")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=3, seed=42)

    adapter = MockModelAdapter()
    adapter.load(model_cfg, hw_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        hardware_config=hw_cfg,
        generation_config=gen_cfg,
        probe_names=["will_be_correct"],
        run_id="test_keys",
        output_dir=str(tmp_output_dir / "outputs"),
    )

    pt_files = list((tmp_output_dir / "outputs" / "test_keys").glob("*.pt"))
    data = torch.load(pt_files[0], weights_only=False)

    assert "activations" in data
    assert "labels" in data
    assert "sample_id" in data
    assert "group_id" in data
    assert "generation_idx" in data
    assert "n_captured_steps" in data
    assert "metadata" in data

    for li in probe_layers:
        assert li in data["activations"]
        acts = data["activations"][li]
        assert acts.dtype == torch.float16
        assert acts.ndim == 2
        assert acts.shape[1] == 64  # MockModelAdapter hidden_dim


def test_stride_subsampling(monkeypatch, tmp_output_dir):
    stride = 3
    n_tokens = 10  # MockModelAdapter._n_tokens
    expected_steps = math.ceil(n_tokens / stride)  # ceil(10/3) = 4

    model_cfg = ModelConfig(model_id="mock", probe_layers=[0], adapter="mock")
    hw_cfg = HardwareConfig(device_map="cpu", dtype="float32")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=stride, seed=42)

    adapter = MockModelAdapter()
    adapter.load(model_cfg, hw_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        hardware_config=hw_cfg,
        generation_config=gen_cfg,
        probe_names=["will_be_correct"],
        run_id="test_stride",
        output_dir=str(tmp_output_dir / "outputs"),
    )

    pt_files = list((tmp_output_dir / "outputs" / "test_stride").glob("*.pt"))
    data = torch.load(pt_files[0], weights_only=False)
    assert data["n_captured_steps"] == expected_steps


def test_hooks_removed_after_run(monkeypatch, tmp_output_dir):
    probe_layers = [0, 1]
    model_cfg = ModelConfig(model_id="mock", probe_layers=probe_layers, adapter="mock")
    hw_cfg = HardwareConfig(device_map="cpu", dtype="float32")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=3, seed=42)

    adapter = MockModelAdapter()
    adapter.load(model_cfg, hw_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        hardware_config=hw_cfg,
        generation_config=gen_cfg,
        probe_names=["will_be_correct"],
        run_id="test_hooks",
        output_dir=str(tmp_output_dir / "outputs"),
    )

    layers = adapter.get_layer_modules()
    for li in probe_layers:
        assert len(layers[li]._forward_hooks) == 0, f"Layer {li} still has hooks after run"


