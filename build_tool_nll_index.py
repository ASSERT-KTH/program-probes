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
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    model_config: ModelConfig = load_config(args.model_config, ModelConfig)
    gen_config: GenerationConfig = load_config(args.generation_config, GenerationConfig)

    out_path = Path(args.output)
    # If sharded, accumulate into a shard-specific temp file; merge manually afterwards.
    # For simplicity we write the whole shard's results to the output path (caller merges).
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard_rank}.pt")

    print(f"Loading trajectories from {args.traj_dir}...")
    all_trajs = load_trajectories(args.traj_dir)
    trajs = all_trajs[args.shard_rank::args.num_shards]
    print(f"  {len(trajs)} trajectories (shard {args.shard_rank}/{args.num_shards})")

    model_adapter = _load_model_adapter(model_config.adapter)
    model_adapter.load_for_extraction(model_config, gen_config)
    device = next(model_adapter._model.parameters()).device

    index: dict[str, dict[int, float]] = {}

    for i, traj in enumerate(trajs):
        print(f"  [{i + 1}/{len(trajs)}] {traj.sample_id} ({len(traj.token_ids)} tokens)...", flush=True)
        input_ids = torch.tensor([traj.token_ids], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                out = model_adapter._model(input_ids=input_ids, output_hidden_states=False)
            logits = out.logits[0].cpu()  # [seq_len, vocab_size] — move to CPU immediately
            del out
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    OOM — skipping {traj.sample_id}")
            continue

        turn_nll = compute_trajectory_nll(
            traj.token_ids, traj.segments, traj.step_segment_indices, logits
        )
        del logits
        index[traj.sample_id] = turn_nll
        print(f"    {len(turn_nll)} turns with tool NLL (out of {len(traj.step_segment_indices)} total turns)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, out_path)
    print(f"Saved tool NLL index ({len(index)} samples) → {out_path}")


if __name__ == "__main__":
    main()
