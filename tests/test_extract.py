import json
import pytest
import torch
from src.configs import ModelConfig, TaskConfig, GenerationConfig
from src.extract import run_extraction
from tests.helpers import MockModelAdapter, MockTaskAdapter


def _write_generations(tmp_path, run_id, n_samples=3, n_gens=1):
    gen_dir = tmp_path / "generations"
    gen_dir.mkdir(exist_ok=True)
    entries = []
    for i in range(n_samples):
        for gi in range(n_gens):
            entries.append({
                "sample_id": f"task_{i}",
                "group_id": f"group_{i % 5}",
                "gen_idx": gi,
                "prompt_token_ids": [0, 1, 2, 3, 4],
                "generated_token_ids": list(range(10)),
                "raw_text": "```python\ndef f(x): return x\n```",
                "task_sample": {"task_id": f"task_{i}", "idx": i},
            })
    shard_file = gen_dir / f"{run_id}_shard000.json"
    with open(shard_file, "w") as f:
        json.dump(entries, f)
    return str(gen_dir)


def test_output_files_created(monkeypatch, tmp_path):
    model_cfg = ModelConfig(model_id="mock", probe_layers=[0, 1], adapter="mock")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=3, seed=42,
                               extraction_batch_size=4)

    adapter = MockModelAdapter()
    adapter.load_for_extraction(model_cfg, gen_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    gens_dir = _write_generations(tmp_path, "test_run", n_samples=5, n_gens=1)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        gen_config=gen_cfg,
        probe_names=["will_be_correct"],
        run_id="test_run",
        generations_dir=gens_dir,
        output_dir=str(tmp_path / "outputs"),
    )

    out_files = list((tmp_path / "outputs" / "test_run").glob("*.pt"))
    assert len(out_files) == 5


def test_output_keys_and_shapes(monkeypatch, tmp_path):
    probe_layers = [0, 1]
    model_cfg = ModelConfig(model_id="mock", probe_layers=probe_layers, adapter="mock")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=3, seed=42,
                               extraction_batch_size=4)

    adapter = MockModelAdapter()
    adapter.load_for_extraction(model_cfg, gen_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    gens_dir = _write_generations(tmp_path, "test_keys", n_samples=2, n_gens=1)

    run_extraction(
        model_config=model_cfg,
        task_config=task_cfg,
        gen_config=gen_cfg,
        probe_names=["will_be_correct"],
        run_id="test_keys",
        generations_dir=gens_dir,
        output_dir=str(tmp_path / "outputs"),
    )

    pt_files = list((tmp_path / "outputs" / "test_keys").glob("*.pt"))
    data = torch.load(pt_files[0], weights_only=False)

    assert "activations" in data
    assert "labels" in data
    assert "sample_id" in data
    assert "group_id" in data
    assert "generation_idx" in data
    assert "metadata" in data

    for li in probe_layers:
        assert li in data["activations"]
        acts = data["activations"][li]
        assert acts.dtype == torch.float16
        assert acts.ndim == 2
        assert acts.shape[1] == MockModelAdapter.HIDDEN_DIM


def test_resume_skips_existing(monkeypatch, tmp_path):
    model_cfg = ModelConfig(model_id="mock", probe_layers=[0], adapter="mock")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=1, temperature=0.7, stride=3, seed=42,
                               extraction_batch_size=4)

    adapter = MockModelAdapter()
    adapter.load_for_extraction(model_cfg, gen_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    gens_dir = _write_generations(tmp_path, "test_resume", n_samples=3, n_gens=1)
    out_dir = str(tmp_path / "outputs")

    run_extraction(model_config=model_cfg, task_config=task_cfg, gen_config=gen_cfg,
                   probe_names=["will_be_correct"], run_id="test_resume",
                   generations_dir=gens_dir, output_dir=out_dir)

    files_first = set(f.name for f in (tmp_path / "outputs" / "test_resume").glob("*.pt"))

def test_hooks_removed_after_run(monkeypatch, tmp_output_dir):
    probe_layers = [0, 1]
    model_cfg = ModelConfig(model_id="mock", probe_layers=probe_layers, adapter="mock")
    task_cfg = TaskConfig(dataset="mock", adapter="mock")
    gen_cfg = GenerationConfig(n_generations=3, temperature=0.7, stride=3, seed=42,
                               extraction_batch_size=4)

    adapter = MockModelAdapter()
    adapter.load_for_extraction(model_cfg, gen_cfg)
    task = MockTaskAdapter()

    monkeypatch.setattr("src.extract._load_model_adapter", lambda n: adapter)
    monkeypatch.setattr("src.extract._load_task_adapter", lambda n: task)

    gens_dir = _write_generations(tmp_output_dir, "test_mulgen", n_samples=4, n_gens=3)

    run_extraction(model_config=model_cfg, task_config=task_cfg, gen_config=gen_cfg,
                   probe_names=["will_be_correct"], run_id="test_mulgen",
                   generations_dir=gens_dir, output_dir=str(tmp_output_dir / "outputs"))

    out_files = list((tmp_output_dir / "outputs" / "test_mulgen").glob("*.pt"))
    assert len(out_files) == 12  # 4 samples × 3 gens
