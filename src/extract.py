import json
import random
import numpy as np
import torch
from pathlib import Path
from src.configs import GenerationConfig, ModelConfig, TaskConfig
from src.probes.base import EditEvent, TrajectoryContext


def _carry_forward(
    edit_labels: list[bool | None],
    edit_history: list[EditEvent],
    n_steps: int,
) -> list[bool | None]:
    """Expand per-edit labels to per-stride-step labels via carry-forward.

    At each stride step *t*, the label is taken from the most recent EditEvent
    with ``step_idx <= t``.  Steps before the first edit get *None*.
    """
    if not edit_labels:
        return [None] * n_steps

    result: list[bool | None] = []
    ei = 0
    for step in range(n_steps):
        while ei < len(edit_history) and edit_history[ei].step_idx <= step:
            ei += 1
        if ei == 0:
            result.append(None)
        else:
            result.append(edit_labels[ei - 1])
    return result


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
    if adapter_name == "qwen35":
        from src.models.qwen35 import Qwen35Adapter
        return Qwen35Adapter()
    if adapter_name == "cwm":
        from src.models.cwm import CwmAdapter
        return CwmAdapter()
    raise ValueError(f"Unknown model adapter: {adapter_name}")


def _load_task_adapter(adapter_name: str):
    if adapter_name == "humaneval":
        from src.tasks.humaneval import HumanEvalAdapter
        return HumanEvalAdapter()
    if adapter_name == "humaneval_fix":
        from src.tasks.humaneval_fix import HumanEvalFixAdapter
        return HumanEvalFixAdapter()
    if adapter_name == "cruxeval_fix":
        from src.tasks.cruxeval_fix import CruxEvalFixAdapter
        return CruxEvalFixAdapter()
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
    gen_config: GenerationConfig,
    probe_names: list[str],
    run_id: str,
    generations_dir: str = "generations",
    output_dir: str = "outputs",
    shard_rank: int = 0,
    num_shards: int = 1,
) -> None:
    _set_seeds(gen_config.seed)

    model_adapter = _load_model_adapter(model_config.adapter)
    model_adapter.load_for_extraction(model_config, gen_config)

    task_adapter = _load_task_adapter(task_config.adapter)
    probes = [_load_probe(name, task_adapter) for name in probe_names]

    # Load all generation shards for this run
    gen_dir = Path(generations_dir)
    shard_files = sorted(gen_dir.glob(f"{run_id}_shard*.json"))
    all_entries: list[dict] = []
    for sf in shard_files:
        with open(sf) as f:
            all_entries.extend(json.load(f))

    # Shard entries across extraction jobs
    entries = all_entries[shard_rank::num_shards]

    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = gen_config.extraction_batch_size
    layer_indices = model_config.probe_layers
    stride = gen_config.stride

    for batch_start in range(0, len(entries), batch_size):
        batch = entries[batch_start: batch_start + batch_size]

        # Skip entries where output already exists
        pending = []
        for entry in batch:
            fname = out_dir / f"{entry['sample_id'].replace('/', '_')}_gen{entry['gen_idx']}.pt"
            if not fname.exists():
                pending.append((entry, fname))
        if not pending:
            continue

        pending_entries = [e for e, _ in pending]
        sequences = [
            e["prompt_token_ids"] + e["generated_token_ids"] for e in pending_entries
        ]
        prompt_lengths = [len(e["prompt_token_ids"]) for e in pending_entries]

        per_seq_hs = model_adapter.extract_hidden_states(
            sequences, prompt_lengths, layer_indices, stride
        )

        for (entry, fname), hs in zip(pending, per_seq_hs):
            n_steps = min(len(v) for v in hs.values()) if hs else 0

            ctx = TrajectoryContext(
                sample=entry["task_sample"],
                generated_text=entry["raw_text"],
                edit_history=[],
            )

            labels = {}
            for probe in probes:
                try:
                    raw_label = probe.compute_label(ctx)
                except NotImplementedError:
                    raw_label = None

                if probe.is_dynamic and isinstance(raw_label, list):
                    labels[probe.name] = _carry_forward(raw_label, ctx.edit_history, n_steps)
                else:
                    labels[probe.name] = raw_label

            out = {
                "activations": hs,
                "labels": labels,
                "sample_id": entry["sample_id"],
                "group_id": entry["group_id"],
                "generation_idx": entry["gen_idx"],
                "metadata": {
                    "prompt_token_ids": entry["prompt_token_ids"],
                    "prompt_text": model_adapter._tokenizer.decode(
                        entry["prompt_token_ids"], skip_special_tokens=False
                    ),
                    "raw_text": entry["raw_text"],
                    "task_sample": entry["task_sample"],
                },
            }

            torch.save(out, fname)
