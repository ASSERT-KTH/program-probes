import math
import os
import random
import numpy as np
import torch
from pathlib import Path
from src.configs import GenerationConfig, ModelConfig, HardwareConfig, TaskConfig
from src.probes.base import TrajectoryContext


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_adapter(adapter_name: str):
    if adapter_name == "qwen":
        from src.models.qwen import QwenAdapter
        return QwenAdapter()
    raise ValueError(f"Unknown model adapter: {adapter_name}")


def _load_task_adapter(adapter_name: str):
    if adapter_name == "humaneval":
        from src.tasks.humaneval import HumanEvalAdapter
        return HumanEvalAdapter()
    raise ValueError(f"Unknown task adapter: {adapter_name}")


def _load_probe(probe_name: str, task_adapter):
    if probe_name == "will_be_correct":
        from src.probes.will_be_correct import WillBeCorrectProbe
        return WillBeCorrectProbe(task_adapter)
    if probe_name == "currently_correct":
        from src.probes.currently_correct import CurrentlyCorrectProbe
        return CurrentlyCorrectProbe()
    if probe_name == "currently_compiles":
        from src.probes.currently_compiles import CurrentlyCompilesProbe
        return CurrentlyCompilesProbe()
    if probe_name == "currently_reduces_failing":
        from src.probes.currently_reduces_failing import CurrentlyReducesFailingProbe
        return CurrentlyReducesFailingProbe()
    if probe_name == "currently_has_regressions":
        from src.probes.currently_has_regressions import CurrentlyHasRegressionsProbe
        return CurrentlyHasRegressionsProbe()
    raise ValueError(f"Unknown probe: {probe_name}")


def run_extraction(
    model_config: ModelConfig,
    task_config: TaskConfig,
    hardware_config: HardwareConfig,
    generation_config: GenerationConfig,
    probe_names: list[str],
    run_id: str,
    output_dir: str = "outputs",
    shard_rank: int = 0,
    num_shards: int = 1,
    max_samples: int | None = None,
) -> None:
    _set_seeds(generation_config.seed)

    model_adapter = _load_model_adapter(model_config.adapter)
    model_adapter.load(model_config, hardware_config)

    task_adapter = _load_task_adapter(task_config.adapter)
    all_samples = task_adapter.load_dataset(task_config)
    samples = all_samples[shard_rank::num_shards]
    if max_samples is not None:
        samples = samples[:max_samples]

    probes = [_load_probe(name, task_adapter) for name in probe_names]

    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_modules = model_adapter.get_layer_modules()
    probe_layer_indices = model_config.probe_layers

    for sample in samples:
        sample_id = task_adapter.sample_id(sample)
        chat_prompt = task_adapter.format_prompt(sample)
        inputs = model_adapter.tokenize(chat_prompt)

        for gen_idx in range(generation_config.n_generations):
            fname = out_dir / f"{sample_id.replace('/', '_')}_gen{gen_idx}.pt"
            if fname.exists():
                continue

            _set_seeds(generation_config.seed + hash((sample_id, gen_idx)) % (2**31))

            captured: dict[int, list[torch.Tensor]] = {li: [] for li in probe_layer_indices}
            handles = []

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # In some transformers versions the layer returns a tuple (hidden, ...),
                    # in others a plain tensor. Normalise to [batch, seq, hidden].
                    h = output[0] if isinstance(output, tuple) else output
                    if h.dim() == 2:
                        # already [batch, hidden] (single-token step with seq squeezed)
                        last = h
                    else:
                        last = h[:, -1, :]
                    captured[layer_idx].append(last.detach().to(torch.float16).cpu())
                return hook

            for li in probe_layer_indices:
                handle = layer_modules[li].register_forward_hook(make_hook(li))
                handles.append(handle)

            generated_text, token_ids, raw_text = model_adapter.generate(
                inputs, generation_config.max_new_tokens, generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                min_p=generation_config.min_p,
            )

            for handle in handles:
                handle.remove()

            n_tokens = len(token_ids)
            stride = generation_config.stride
            step_indices = list(range(0, n_tokens, stride))
            n_captured_steps = len(step_indices)

            activations: dict[int, torch.Tensor] = {}
            for li in probe_layer_indices:
                raw = captured[li]
                # raw has one entry per forward pass token; subsample by stride
                subsampled = [raw[i] for i in step_indices if i < len(raw)]
                if subsampled:
                    activations[li] = torch.cat(subsampled, dim=0).to(torch.float16)
                else:
                    hidden_dim = model_adapter.get_hidden_dim()
                    activations[li] = torch.zeros(0, hidden_dim, dtype=torch.float16)

            ctx = TrajectoryContext(
                sample=sample,
                generated_text=generated_text,
                n_captured_steps=n_captured_steps,
                edit_history=[],
            )

            labels = {}
            for probe in probes:
                try:
                    labels[probe.name] = probe.compute_label(ctx)
                except NotImplementedError:
                    labels[probe.name] = [None] * n_captured_steps if probe.is_dynamic else None

            out = {
                "activations": activations,
                "labels": labels,
                "sample_id": sample_id,
                "group_id": task_adapter.group_id(sample),
                "generation_idx": gen_idx,
                "n_captured_steps": n_captured_steps,
                "metadata": {
                    "prompt": chat_prompt.user_content,
                    "generated_text": generated_text,
                    "raw_text": raw_text,
                    "task_sample": sample,
                },
            }

            torch.save(out, fname)
