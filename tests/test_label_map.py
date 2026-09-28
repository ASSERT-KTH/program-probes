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
        # cmd_idx=0 is issued during turn 0; effect visible from turn 1 onwards
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 0,  "compiles": True,  "test_results": {"resolved": False}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        # turn 0 still gets baseline (False); turns 1,2 get cmd_idx=0's label (True)
        assert labels[:10] == [False] * 10, f"turn 0 should still be False: {labels[:10]}"
        assert all(l is True for l in labels[10:]), f"turns 1,2 should be True: {labels[10:]}"

    def test_token_range_boundaries(self):
        msgs, segs, mask, _ = _build(n_assistant=3, tokens_per_turn=5)
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 1,  "compiles": True,  "test_results": {"resolved": False}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        # turns 0,1 get baseline (False); turn 2 gets cmd_idx=1 effect (True)
        assert labels[:10] == [False] * 10
        assert labels[10:] == [True] * 5


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
        # cmd_idx=2 effect visible from turn 3; turns 0,1,2 get False, turns 3,4 get True
        n_per_turn = 10
        assert labels[:n_per_turn * 3] == [False] * (n_per_turn * 3)
        assert labels[n_per_turn * 3:] == [True] * (n_per_turn * 2)

    def test_multiple_edits_carry_forward(self):
        msgs, segs, mask, _ = _build(n_assistant=6, tokens_per_turn=4)
        edits = [
            {"cmd_idx": -1, "compiles": False, "test_results": {"resolved": False}},
            {"cmd_idx": 1,  "compiles": True,  "test_results": {"resolved": False}},
            {"cmd_idx": 4,  "compiles": True,  "test_results": {"resolved": True}},
        ]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_correct")
        # cmd_idx=1 → edit_turn=2; cmd_idx=4 → edit_turn=5
        assert labels[:8]   == [False] * 8   # turns 0,1: baseline
        assert labels[8:20] == [False] * 12  # turns 2,3,4: cmd_idx=1 resolved=False
        assert labels[20:]  == [True] * 4    # turn 5: cmd_idx=4 resolved=True


class TestNoEdits:
    def test_empty_edits_returns_all_none(self):
        msgs, segs, mask, _ = _build(n_assistant=3, tokens_per_turn=5)
        labels = build_label_sequence(segs, msgs, [], mask, stride=1, probe="currently_compiles")
        assert all(l is None for l in labels)

    def test_no_baseline_returns_none_before_first_edit(self):
        msgs, segs, mask, _ = _build(n_assistant=4, tokens_per_turn=5)
        edits = [{"cmd_idx": 2, "compiles": True, "test_results": {"resolved": True}}]
        labels = build_label_sequence(segs, msgs, edits, mask, stride=1, probe="currently_compiles")
        # cmd_idx=2 → edit_turn=3; turns 0,1,2 get None, turn 3 gets True
        assert labels[:15] == [None] * 15   # turns 0,1,2: no label yet
        assert labels[15:] == [True] * 5    # turn 3: cmd_idx=2 effect visible


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

    TRAJ = Path("tests/resources/swebench/astropy__astropy-12907.json")
    LABELS = Path("tests/resources/swebench/astropy__astropy-12907_labels.json")

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
        # cmd_idx=9 effect is visible from turn 10 onwards
        segs = self.traj["tokenization"]["segments"]
        asst_segs = [s for s in segs if s["role"] == "assistant"]
        # turn 10's start token (= end token of turn 9)
        turn10_start = asst_segs[10]["start_token"]
        extracted = [p for p in range(len(mask)) if mask[p] == 1]  # stride=1
        after_indices = [i for i, p in enumerate(extracted) if p >= turn10_start]
        assert after_indices, "no extracted positions in turn 10+"
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


# ---------------------------------------------------------------------------
# map_positions_to_turns / edit_labels_by_turn
# ---------------------------------------------------------------------------

def test_positions_to_turns_uneven_turn_lengths():
    """Turn indices follow real segment boundaries, not a uniform split."""
    from src.tasks.swe_bench_label_map import map_positions_to_turns

    msgs = _make_messages(3)
    # system, user, a0 (20 tok), u1, a1 (2 tok), u2, a2 (8 tok), u3
    lengths = [5, 5, 20, 5, 2, 5, 8, 5]
    segs, pos = [], 0
    for i, (m, n) in enumerate(zip(msgs, lengths)):
        segs.append({"message_idx": i, "role": m["role"], "start_token": pos, "end_token": pos + n})
        pos += n
    mask = _make_mask(segs, pos)

    turns = map_positions_to_turns(segs, msgs, mask, stride=1)
    assert turns == [0] * 20 + [1] * 2 + [2] * 8


def test_edit_labels_by_turn_carry_forward():
    """Edit at cmd_idx N takes effect from turn N+1; baseline from turn 0."""
    from src.tasks.swe_bench_label_map import edit_labels_by_turn

    edits = [{"cmd_idx": -1}, {"cmd_idx": 1}, {"cmd_idx": 3}]
    assert edit_labels_by_turn(edits, [False, True, False], 6) == [False, False, True, True, False, False]


def test_edit_labels_by_turn_no_baseline():
    from src.tasks.swe_bench_label_map import edit_labels_by_turn

    assert edit_labels_by_turn([{"cmd_idx": 0}], [True], 3) == [None, True, True]
