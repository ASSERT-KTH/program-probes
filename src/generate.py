import json
import random
import numpy as np
from pathlib import Path
from src.configs import GenerationConfig, ModelConfig, TaskConfig


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def run_generation(
    model_config: ModelConfig,
    task_config: TaskConfig,
    gen_config: GenerationConfig,
    run_id: str,
    generations_dir: str = "generations",
    shard_rank: int = 0,
    num_shards: int = 1,
    max_samples: int | None = None,
) -> None:
    _set_seeds(gen_config.seed)

    model_adapter = _load_model_adapter(model_config.adapter)

    task_adapter = _load_task_adapter(task_config.adapter)
    all_samples = task_adapter.load_dataset(task_config)
    samples = all_samples[shard_rank::num_shards]
    if max_samples is not None:
        samples = samples[:max_samples]

    out_path = Path(generations_dir) / f"{run_id}_shard{shard_rank:03d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed entries to support resuming
    completed: set[tuple[str, int]] = set()
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        for entry in existing:
            completed.add((entry["sample_id"], entry["gen_idx"]))

    results = list(existing)

    # Build prompts first (tokenizer only) so we can compute max prompt length
    # before initialising vLLM, which needs to know max_model_len for KV cache.
    model_adapter.load_tokenizer(model_config)
    pending: list[tuple[dict, int, list[int]]] = []
    for sample in samples:
        chat_prompt = task_adapter.format_prompt(sample)
        prompt_token_ids = model_adapter.build_prompt(chat_prompt)
        sample_id = task_adapter.sample_id(sample)
        for gi in range(gen_config.n_generations):
            if (sample_id, gi) not in completed:
                pending.append((sample, gi, prompt_token_ids))

    max_prompt_len = max((len(p[2]) for p in pending), default=0)
    max_model_len = gen_config.max_model_len or (max_prompt_len + gen_config.max_new_tokens)
    model_adapter.load_for_generation(model_config, gen_config, max_model_len)

    # Group pending by sample so we can checkpoint after each one
    from itertools import groupby
    pending_by_sample = [
        list(g) for _, g in groupby(pending, key=lambda x: task_adapter.sample_id(x[0]))
    ]

    for i, sample_pending in enumerate(pending_by_sample):
        sample_id = task_adapter.sample_id(sample_pending[0][0])
        print(f"Generating sample {i + 1}/{len(pending_by_sample)}: {sample_id}", flush=True)
        sample_prompts = [item[2] for item in sample_pending]
        sample_results = model_adapter.generate(sample_prompts, gen_config)

        for (sample, gi, _), gen_result in zip(sample_pending, sample_results):
            results.append({
                "sample_id": task_adapter.sample_id(sample),
                "group_id": task_adapter.group_id(sample),
                "gen_idx": gi,
                "prompt_token_ids": gen_result.prompt_token_ids,
                "generated_token_ids": gen_result.generated_token_ids,
                "raw_text": gen_result.raw_text,
                "task_sample": sample,
            })

        with open(out_path, "w") as f:
            json.dump(results, f)
