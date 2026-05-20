"""Unit tests for src/tasks/swe_bench_label_map.py.

Run with:
    uv run python -m pytest tests/test_label_map.py -v
"""
import json
from pathlib import Path

import pytest

from src.tasks.swe_bench_label_map import build_label_sequence


# ---------------------------------------------------------------------------
# Helpers to build minimal synthetic data
# ---------------------------------------------------------------------------

def _make_messages(n_assistant: int) -> list[dict]:
    """Alternating system/user/assistant... messages."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u0"}]
    for i in range(n_assistant):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "user", "content": f"u{i+1}"})
    return msgs


def _make_segments(messages: list[dict], tokens_per_turn: int = 10) -> list[dict]:
    """Assign contiguous token ranges; every message gets tokens_per_turn tokens."""
    segs = []
    pos = 0
    for i, m in enumerate(messages):
        segs.append({
            "message_idx": i,
            "role": m["role"],
            "start_token": pos,
            "end_token": pos + tokens_per_turn,
        })
        pos += tokens_per_turn
    return segs


def _make_mask(segments: list[dict], total_tokens: int) -> list[int]:
    """1 at assistant-token positions, 0 elsewhere."""
    mask = [0] * total_tokens
    for seg in segments:
        if seg["role"] == "assistant":
            for p in range(seg["start_token"], seg["end_token"]):
                mask[p] = 1
    return mask


def _build(n_assistant: int, tokens_per_turn: int = 10):
    msgs = _make_messages(n_assistant)
    segs = _make_segments(msgs, tokens_per_turn)
    total = len(msgs) * tokens_per_turn
    mask = _make_mask(segs, total)
    return msgs, segs, mask, total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCmdIdxToSegment:
    def test_assistant_turn_zero_maps_to_first_assistant_segment(self):
        msgs, segs, mask, _ = _build(n_assistant=3, tokens_per_turn=10)
        # cmd_idx=0 = first assistant turn; baseline applies before it
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 0,  "compiles": True,  "test_results": {"resolved": False}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        # First assistant segment (turn 0) should get cmd_idx=0's label (True)
        # because turn 0 >= edit_turn 0
        assert all(l is True for l in labels[:10]), f"first turn labels: {labels[:10]}"

    def test_token_range_boundaries(self):
        msgs, segs, mask, _ = _build(n_assistant=2, tokens_per_turn=5)
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 1,  "compiles": True,  "test_results": {"resolved": False}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        # turn 0 assistant: gets baseline (False); turn 1 assistant: gets cmd_idx=1 (True)
        assert labels[:5] == [False] * 5
        assert labels[5:] == [True] * 5


class TestBaselineEdit:
    def test_baseline_applied_to_all_positions_when_no_other_edits(self):
        msgs, segs, mask, _ = _build(n_assistant=4, tokens_per_turn=8)
        edits = [{"cmd_idx": -1, "compiles": True, "test_results": {"resolved": False}}]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        assert all(l is True for l in labels)

    def test_baseline_resolved_false(self):
        msgs, segs, mask, _ = _build(n_assistant=3, tokens_per_turn=6)
        edits = [{"cmd_idx": -1, "compiles": True, "test_results": {"resolved": False}}]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_correct")
        assert all(l is False for l in labels)


class TestCarryForward:
    def test_label_carries_forward_after_edit(self):
        msgs, segs, mask, _ = _build(n_assistant=5, tokens_per_turn=10)
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 2,  "compiles": True,  "test_results": {"resolved": True}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        # Turns 0,1 get False (baseline); turns 2,3,4 get True (cmd_idx=2)
        n_per_turn = 10
        assert labels[:n_per_turn * 2] == [False] * (n_per_turn * 2)
        assert labels[n_per_turn * 2:] == [True] * (n_per_turn * 3)

    def test_multiple_edits_carry_forward(self):
        msgs, segs, mask, _ = _build(n_assistant=6, tokens_per_turn=4)
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 1,  "compiles": True,  "test_results": {"resolved": False}},
            {"cmd_idx": 4,  "compiles": True,  "test_results": {"resolved": True}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_correct")
        assert labels[:4]   == [False] * 4   # turn 0: baseline
        assert labels[4:16] == [False] * 12  # turns 1,2,3: cmd_idx=1 resolved=False
        assert labels[16:]  == [True] * 8    # turns 4,5: cmd_idx=4 resolved=True


class TestNoEdits:
    def test_empty_edits_returns_all_none(self):
        msgs, segs, mask, _ = _build(n_assistant=3, tokens_per_turn=5)
        labels = build_label_sequence(segs, msgs, [], mask, stride=1, probe="currently_compiles")
        assert all(l is None for l in labels)

    def test_no_baseline_returns_none_before_first_edit(self):
        msgs, segs, mask, _ = _build(n_assistant=4, tokens_per_turn=5)
        edits = [{"cmd_idx": 2, "compiles": True, "test_results": {"resolved": True}}]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        assert labels[:10] == [None] * 10   # turns 0,1: no label yet
        assert labels[10:] == [True] * 10   # turns 2,3: cmd_idx=2


class TestStride:
    def test_stride_reduces_output_length(self):
        msgs, segs, mask, _ = _build(n_assistant=2, tokens_per_turn=10)
        edits = [{"cmd_idx": -1, "compiles": True, "test_results": {"resolved": False}}]
        labels_s1 = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        labels_s5 = build_label_sequence(segs, msgs, edits, mask, stride=5, probe="currently_compiles")
        assert len(labels_s1) == 20
        assert len(labels_s5) == 4  # 20 masked positions / stride 5

    def test_stride_does_not_change_label_values(self):
        msgs, segs, mask, _ = _build(n_assistant=2, tokens_per_turn=10)
        edits = [{"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}}]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=3, probe="currently_compiles")
        assert all(l is False for l in labels)


class TestRealTrajectory:
    """Integration test using actual trajectory and label files."""

    TRAJ = Path("generations/swebench/qwen36_27b_test/astropy__astropy-12907.json")
    LABELS = Path("labels/swebench/qwen36_27b_test/astropy__astropy-12907_labels.json")

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not self.TRAJ.exists() or not self.LABELS.exists():
            pytest.skip("real trajectory/label files not available")

    def setup_method(self):
        self.traj = json.loads(self.TRAJ.read_text())
        self.label_data = json.loads(self.LABELS.read_text())

    def _build_mask(self):
        segs = self.traj["tokenization"]["segments"]
        n = len(self.traj["tokenization"]["token_ids"])
        mask = [0] * n
        for seg in segs:
            if seg["role"] == "assistant":
                for p in range(seg["start_token"], seg["end_token"]):
                    mask[p] = 1
        return mask

    def test_label_count_matches_extracted_positions(self):
        mask = self._build_mask()
        stride = 5
        expected = sum(1 for i, m in enumerate(mask) if m == 1 and (
            sum(1 for j in range(i+1) if mask[j] == 1) - 1
        ) % stride == 0)
        labels = build_label_sequence(
            self.traj["tokenization"]["segments"],
            self.traj["messages"],
            self.label_data["edits"],
            mask,
            stride=stride,
            probe="currently_compiles",
        )
        assert len(labels) == expected

    def test_baseline_edit_at_cmd_idx_minus_one(self):
        """Positions in turn 0 should get the baseline (cmd_idx=-1) label."""
        mask = self._build_mask()
        edits = self.label_data["edits"]
        baseline = next((e for e in edits if e["cmd_idx"] == -1), None)
        assert baseline is not None, "no baseline edit in label file"

        labels = build_label_sequence(
            self.traj["tokenization"]["segments"],
            self.traj["messages"],
            edits,
            mask,
            stride=1,
            probe="currently_compiles",
        )
        # The first position belongs to turn 0 so gets the baseline label
        assert labels[0] == bool(baseline["compiles"])

    def test_edit_at_cmd_idx_9_changes_label(self):
        """After cmd_idx=9 the label should change to resolved=True."""
        mask = self._build_mask()
        edits = self.label_data["edits"]
        edit_9 = next((e for e in edits if e["cmd_idx"] == 9), None)
        assert edit_9 is not None, "expected edit at cmd_idx=9"

        labels = build_label_sequence(
            self.traj["tokenization"]["segments"],
            self.traj["messages"],
            edits,
            mask,
            stride=1,
            probe="currently_correct",
        )
        # All labels should be non-None (baseline covers everything before turn 9)
        assert all(l is not None for l in labels)
        # Labels after turn 9 should be True (resolved)
        segs = self.traj["tokenization"]["segments"]
        msgs = self.traj["messages"]
        asst_segs = [s for s in segs if s["role"] == "assistant"]
        # turn 9's end token
        turn9_end = asst_segs[9]["end_token"]
        # find indices into labels[] corresponding to token positions >= turn9_end
        extracted = [p for p in range(len(mask)) if mask[p] == 1]  # stride=1
        after_indices = [i for i, p in enumerate(extracted) if p >= turn9_end]
        assert after_indices, "no extracted positions after turn 9"
        for idx in after_indices:
            assert labels[idx] is True, f"labels[{idx}] should be True, got {labels[idx]}"

    def test_no_labels_are_none_with_baseline(self):
        """With a baseline edit present, no extracted position should return None."""
        mask = self._build_mask()
        labels = build_label_sequence(
            self.traj["tokenization"]["segments"],
            self.traj["messages"],
            self.label_data["edits"],
            mask,
            stride=5,
            probe="currently_correct",
        )
        assert all(l is not None for l in labels)
