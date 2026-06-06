"""Tests for build_tool_nll_index.compute_trajectory_nll and probe NLL correlation."""
import math
import numpy as np
import pytest
import torch

from build_tool_nll_index import compute_trajectory_nll


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_logits(seq_len: int, vocab_size: int) -> torch.Tensor:
    """All-zeros logits → uniform softmax → NLL = log(vocab_size) per token."""
    return torch.zeros(seq_len, vocab_size)


def _peaked_logits(seq_len: int, vocab_size: int, target_ids: list[int]) -> torch.Tensor:
    """Logits that assign probability ≈ 1 to target_ids[i] at position i."""
    logits = torch.full((seq_len, vocab_size), -1e9)
    for i, t in enumerate(target_ids):
        logits[i, t] = 0.0
    return logits


def _make_segments_and_steps(
    turns: list[tuple[str, int, int]],
) -> tuple[list[dict], list[int]]:
    """Build segments list and step_segment_indices from (role, start, end) tuples."""
    segments = [
        {"message_idx": i, "role": r, "start_token": s, "end_token": e}
        for i, (r, s, e) in enumerate(turns)
    ]
    step_segs = [i for i, (r, _, _) in enumerate(turns) if r == "assistant"]
    return segments, step_segs


# ---------------------------------------------------------------------------
# Tests for compute_trajectory_nll
# ---------------------------------------------------------------------------

def test_nll_basic():
    """Uniform logits → NLL = log(vocab_size) for every token in tool output."""
    vocab_size = 8
    # Layout: system[0:2] user[2:5] assistant[5:8] user[8:11] assistant[11:14]
    turns = [
        ("system",    0,  2),
        ("user",      2,  5),   # initial task prompt
        ("assistant", 5,  8),   # turn 0
        ("user",      8,  11),  # tool output → NLL for turn 1
        ("assistant", 11, 14),  # turn 1
    ]
    token_ids = [0] * 14
    segments, step_segs = _make_segments_and_steps(turns)
    logits = _uniform_logits(14, vocab_size)

    result = compute_trajectory_nll(token_ids, segments, step_segs, logits)

    assert set(result.keys()) == {1}, "Only turn 1 has a preceding tool output"
    expected = math.log(vocab_size)
    assert abs(result[1] - expected) < 1e-4, f"Expected NLL≈{expected:.4f}, got {result[1]:.4f}"


def test_nll_normalized_by_length():
    """Different tool output lengths with identical per-token NLL → same mean NLL."""
    vocab_size = 4
    # Turn 1 tool output: 3 tokens [8:11]; turn 2 tool output: 7 tokens [14:21]
    turns = [
        ("system",    0,  2),
        ("user",      2,  5),
        ("assistant", 5,  8),   # turn 0
        ("user",      8,  11),  # tool output (3 tokens) → turn 1
        ("assistant", 11, 14),  # turn 1
        ("user",      14, 21),  # tool output (7 tokens) → turn 2
        ("assistant", 21, 24),  # turn 2
    ]
    token_ids = [0] * 24
    segments, step_segs = _make_segments_and_steps(turns)
    logits = _uniform_logits(24, vocab_size)  # NLL = log(4) per token

    result = compute_trajectory_nll(token_ids, segments, step_segs, logits)

    assert set(result.keys()) == {1, 2}
    assert abs(result[1] - result[2]) < 1e-5, "Normalization should give equal NLL regardless of length"


def test_nll_skips_turn_0():
    """Turn 0 is never included because it is preceded only by the initial task prompt."""
    vocab_size = 4
    turns = [
        ("system",    0, 2),
        ("user",      2, 6),   # initial task prompt
        ("assistant", 6, 9),   # turn 0 — preceded by initial user prompt, NOT tool output
        ("user",      9, 12),  # tool output → turn 1
        ("assistant", 12, 15), # turn 1
    ]
    token_ids = [0] * 15
    segments, step_segs = _make_segments_and_steps(turns)
    logits = _uniform_logits(15, vocab_size)

    result = compute_trajectory_nll(token_ids, segments, step_segs, logits)

    assert 0 not in result, "Turn 0 must not appear in the index"
    assert 1 in result


def test_nll_exact_values():
    """Peaked logits → NLL ≈ 0 for correct tokens; verify exact computation."""
    vocab_size = 4
    turns = [
        ("system",    0, 1),
        ("user",      1, 3),
        ("assistant", 3, 5),   # turn 0
        ("user",      5, 8),   # tool output (3 tokens: ids [1,2,3]) → turn 1 NLL
        ("assistant", 8, 10),  # turn 1
    ]
    # Token IDs for the tool output at positions [5, 6, 7]
    token_ids = [0] * 10
    token_ids[5] = 1
    token_ids[6] = 2
    token_ids[7] = 3

    segments, step_segs = _make_segments_and_steps(turns)

    # Peaked logits: logits[i] puts all mass on token_ids[i+1]
    # For tool output at [5,6,7], we need logits at [4,5,6] to peak on [1,2,3]
    all_targets = token_ids[1:]  # logits[i] predicts token_ids[i+1]
    logits = _peaked_logits(10, vocab_size, all_targets[:9])  # only need first 9 rows

    result = compute_trajectory_nll(token_ids, segments, step_segs, logits)

    assert 1 in result
    assert result[1] < 1e-3, f"NLL should be ≈0 for correct-target logits, got {result[1]:.6f}"


def test_nll_multiple_turns():
    """Three tool output turns → index has entries for turns 1, 2, 3 only."""
    vocab_size = 4
    # Alternating: asst / user / asst / user / asst / user / asst
    turns = [
        ("system",    0,  2),
        ("user",      2,  4),   # initial task
        ("assistant", 4,  6),   # turn 0
        ("user",      6,  8),   # tool → turn 1
        ("assistant", 8,  10),  # turn 1
        ("user",      10, 13),  # tool → turn 2
        ("assistant", 13, 15),  # turn 2
        ("user",      15, 18),  # tool → turn 3
        ("assistant", 18, 20),  # turn 3
    ]
    token_ids = [0] * 20
    segments, step_segs = _make_segments_and_steps(turns)
    logits = _uniform_logits(20, vocab_size)

    result = compute_trajectory_nll(token_ids, segments, step_segs, logits)

    assert set(result.keys()) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Tests for Brier score and Spearman correlation logic
# ---------------------------------------------------------------------------

def _brier_per_step(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (probs - labels) ** 2


def test_brier_score_values():
    """Verify per-step Brier score: perfect probe → 0, wrong probe → 1."""
    labels = np.array([1, 0, 1, 0], dtype=float)

    perfect_probs = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.allclose(_brier_per_step(perfect_probs, labels), 0.0)

    wrong_probs = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(_brier_per_step(wrong_probs, labels), 1.0)

    uncertain_probs = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(_brier_per_step(uncertain_probs, labels), 0.25)


def test_spearman_direction():
    """Higher NLL → higher Brier score → Spearman ρ > 0 (hypothesis direction)."""
    from scipy.stats import spearmanr

    # Construct positions where NLL and Brier are positively correlated:
    # low NLL → probe is confident and correct (low Brier)
    # high NLL → probe is uncertain and wrong (high Brier)
    nll    = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0])
    brier  = np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])

    rho, _ = spearmanr(nll, brier)
    assert rho > 0.9, f"Expected strong positive Spearman ρ, got {rho:.3f}"

    # Reversed: ρ should be negative
    rho_neg, _ = spearmanr(-nll, brier)
    assert rho_neg < -0.9


def test_spearman_zero_when_uncorrelated():
    """Constant NLL → Spearman ρ = 0 (or undefined; handle gracefully)."""
    from scipy.stats import spearmanr

    nll   = np.array([1.0] * 10)  # constant — all ranks tied
    brier = np.linspace(0, 1, 10)

    rho, _ = spearmanr(nll, brier)
    # scipy returns nan when one variable is constant
    assert np.isnan(rho) or abs(rho) < 1e-9
