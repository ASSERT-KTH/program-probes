"""Compute per-turn mean NLL of tool output tokens for SWE-bench trajectories.

For each assistant turn k > 0, the model's mean negative log-likelihood over the
tool/environment output tokens that precede turn k is stored. This quantifies how
"surprised" the model was by the test runner or compiler response.

Output: <output> (a .pt file)
  = {sample_id: {turn_k (int): mean_nll (float)}}

Turn 0 is excluded because it has no preceding tool output (only the initial task).

Usage (array job, 8 shards):
    sbatch --array=0-7 slurm/build_tool_nll.sh \\
        --model-config configs/models/laguna_xs2.yaml \\
        --generation-config configs/generation_laguna_xs2.yaml \\
        --traj-dir generations/swebench/laguna_xs2_full \\
        --output cache/swebench/laguna_xs2_full/tool_nll_index.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.configs import GenerationConfig, ModelConfig, load_config
from src.tasks.swe_bench_extract import load_trajectories


def compute_trajectory_nll(
    token_ids: list[int],
    segments: list[dict],
    step_segment_indices: list[int],
    logits: torch.Tensor,
) -> dict[int, float]:
    """Per-turn mean NLL of tool output tokens for turns k > 0.

    logits[i] is the model's prediction for token i+1, so the NLL of a tool
    output spanning token positions [start, end) is:
        mean(-log softmax(logits[start-1 : end-1])[:, token_ids[start:end]])

    Args:
        token_ids: full token sequence (length = seq_len)
        segments: tokenization.segments list (role, start_token, end_token)
        step_segment_indices: indices into *segments* for each assistant turn
        logits: [seq_len, vocab_size] — output of one full forward pass

    Returns:
        {turn_k: mean_nll} for turns that have a preceding user (tool output) segment
    """
    turn_nll: dict[int, float] = {}
    for k, asst_seg_idx in enumerate(step_segment_indices):
        if k == 0 or asst_seg_idx == 0:
            continue  # turn 0 is preceded by the initial task prompt, not a tool response
        tool_seg = segments[asst_seg_idx - 1]
        if tool_seg.get("role") != "user":
            continue
        start = tool_seg["start_token"]
        end = tool_seg["end_token"]
        n_tokens = end - start
        if n_tokens <= 0 or start == 0:
            continue
        # logits[start-1] predicts token[start], …, logits[end-2] predicts token[end-1]
        tool_logits = logits[start - 1 : end - 1]  # [n_tokens, vocab_size]
        if tool_logits.shape[0] != n_tokens:
            continue
        actual = torch.tensor(token_ids[start:end], dtype=torch.long, device=logits.device)
        log_probs = torch.log_softmax(tool_logits.float(), dim=-1)
        token_nll = -log_probs[torch.arange(n_tokens, device=logits.device), actual]
        turn_nll[k] = float(token_nll.mean().item())
    return turn_nll


def _tool_output_ranges(
    segments: list[dict],
    step_segment_indices: list[int],
) -> dict[int, tuple[int, int]]:
    """Return {turn_k: (start_token, end_token)} for each tool output segment."""
    ranges: dict[int, tuple[int, int]] = {}
    for k, asst_seg_idx in enumerate(step_segment_indices):
        if k == 0 or asst_seg_idx == 0:
            continue
        tool_seg = segments[asst_seg_idx - 1]
        if tool_seg.get("role") != "user":
            continue
        start, end = tool_seg["start_token"], tool_seg["end_token"]
        if end > start and start > 0:
            ranges[k] = (start, end)
    return ranges


def compute_trajectory_nll_chunked(
    token_ids: list[int],
    segments: list[dict],
    step_segment_indices: list[int],
    model,
    device,
    chunk_size: int,
) -> dict[int, float]:
    """KV-cache chunked version of compute_trajectory_nll.

    Identical semantics to compute_trajectory_nll but processes the sequence in
    chunks of chunk_size tokens (with accumulated KV cache), so peak GPU memory is
    O(chunk_size × vocab_size) rather than O(seq_len × vocab_size). Handles
    arbitrarily long sequences without OOM.
    """
    turn_ranges = _tool_output_ranges(segments, step_segment_indices)
    if not turn_ranges:
        return {}

    seq_len = len(token_ids)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    # Accumulate per-token NLL values keyed by turn
    turn_nll_values: dict[int, list[float]] = {k: [] for k in turn_ranges}
    past_kv = None

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_input = input_ids[:, chunk_start:chunk_end]
        # Attention mask must span past (cached) + current tokens
        chunk_attn = torch.ones(1, chunk_end, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                input_ids=chunk_input,
                attention_mask=chunk_attn,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=False,
            )

        chunk_logits = out.logits[0]  # [chunk_len, vocab_size], on GPU

        for k, (tok_start, tok_end) in turn_ranges.items():
            # logits[j-1] predicts token[j]: need logit positions [tok_start-1, tok_end-1)
            logit_start = tok_start - 1
            logit_end = tok_end - 1
            # Intersect with the current chunk's logit positions [chunk_start, chunk_end)
            overlap_start = max(logit_start, chunk_start)
            overlap_end = min(logit_end, chunk_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - chunk_start
            local_end = overlap_end - chunk_start
            logit_slice = chunk_logits[local_start:local_end]  # [n, vocab]
            # Tokens predicted: token positions [overlap_start+1, overlap_end+1)
            actual = torch.tensor(
                token_ids[overlap_start + 1 : overlap_end + 1],
                dtype=torch.long, device=device,
            )
            n = actual.shape[0]
            log_probs = torch.log_softmax(logit_slice.float(), dim=-1)
            nll_vals = -log_probs[torch.arange(n, device=device), actual]
            turn_nll_values[k].extend(nll_vals.tolist())

        past_kv = out.past_key_values
        del out, chunk_logits

    return {k: sum(v) / len(v) for k, v in turn_nll_values.items() if v}


def _load_model_adapter(adapter_name: str):
    if adapter_name == "laguna":
        from src.models.laguna import LagunaAdapter
        return LagunaAdapter()
    if adapter_name == "qwen":
        from src.models.qwen import QwenAdapter
        return QwenAdapter()
    if adapter_name == "qwen35":
        from src.models.qwen35 import Qwen35Adapter
        return Qwen35Adapter()
    if adapter_name == "cwm":
        from src.models.cwm import CwmAdapter
        return CwmAdapter()
    raise ValueError(f"Unknown model adapter: {adapter_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--traj-dir", required=True, help="Directory of trajectory JSON files")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Tokens per forward-pass chunk (KV-cache chunking); "
                             "bounds peak GPU memory to O(chunk_size × vocab_size)")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    model_config: ModelConfig = load_config(args.model_config, ModelConfig)
    gen_config: GenerationConfig = load_config(args.generation_config, GenerationConfig)

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard_rank}.pt")

    print(f"Loading trajectories from {args.traj_dir}...")
    all_trajs = load_trajectories(args.traj_dir)
    trajs = all_trajs[args.shard_rank::args.num_shards]
    print(f"  {len(trajs)} trajectories (shard {args.shard_rank}/{args.num_shards}), "
          f"chunk_size={args.chunk_size}")

    model_adapter = _load_model_adapter(model_config.adapter)
    model_adapter.load_for_extraction(model_config, gen_config)
    device = next(model_adapter._model.parameters()).device

    index: dict[str, dict[int, float]] = {}

    for i, traj in enumerate(trajs):
        n_chunks = (len(traj.token_ids) + args.chunk_size - 1) // args.chunk_size
        print(f"  [{i + 1}/{len(trajs)}] {traj.sample_id} "
              f"({len(traj.token_ids)} tokens, {n_chunks} chunks)...", flush=True)
        turn_nll = compute_trajectory_nll_chunked(
            traj.token_ids, traj.segments, traj.step_segment_indices,
            model_adapter._model, device, args.chunk_size,
        )
        index[traj.sample_id] = turn_nll
        print(f"    {len(turn_nll)} turns with NLL "
              f"(out of {len(traj.step_segment_indices)} total turns)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, out_path)
    print(f"Saved tool NLL index ({len(index)} samples) → {out_path}")


if __name__ == "__main__":
    main()
