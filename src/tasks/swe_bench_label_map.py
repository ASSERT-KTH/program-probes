"""Map per-edit labels from the labeler to per-extracted-position labels.

The mapping chain:
    cmd_idx (command_history index, 0-based; -1 = baseline before any edit)
      -> assistant turn index  (1:1: cmd_idx i corresponds to the i-th assistant message)
      -> message_idx           (from messages[] filtered to role='assistant')
      -> segment               (tokenization.segments entry with matching message_idx)
      -> token range           [start_token, end_token)
      -> extracted positions   (positions within range that survive the extraction mask + stride)

Each extracted position receives the label of the most recent edit at or before its
assistant turn, carried forward.  Positions before the first real edit use the
baseline label (cmd_idx=-1).  If no baseline edit exists, those positions get None.
"""
from __future__ import annotations


def map_edit_labels_to_positions(
    segments: list[dict],
    messages: list[dict],
    sorted_edits: list[dict],
    extraction_mask: list[int],
    stride: int,
    edit_labels: list[bool | None],
) -> list[bool | None]:
    """Map pre-computed per-edit labels to per-extracted-position labels.

    Parameters
    ----------
    segments:
        trajectory['tokenization']['segments'].
    messages:
        trajectory['messages'].
    sorted_edits:
        labels['edits'] sorted ascending by cmd_idx.
    extraction_mask:
        Binary vector (same length as token_ids), 1 at assistant-token positions.
    stride:
        Every stride-th masked position is extracted.
    edit_labels:
        One label per entry in sorted_edits.

    Returns
    -------
    list[bool | None]
        One entry per extracted position, carry-forwarded from the most recent
        edit whose effect is visible at that position.
    """
    turns = map_positions_to_turns(segments, messages, extraction_mask, stride)
    turn_labels = edit_labels_by_turn(sorted_edits, edit_labels, max(turns, default=-1) + 1)
    return [turn_labels[t] for t in turns]


def map_positions_to_turns(
    segments: list[dict],
    messages: list[dict],
    extraction_mask: list[int],
    stride: int,
) -> list[int]:
    """Return the assistant-turn index of every extracted position.

    Uses the same stride/mask logic as extraction, so the result is aligned
    1:1 with the activation rows.
    """
    asst_message_indices = [
        i for i, m in enumerate(messages) if m["role"] == "assistant"
    ]
    seg_by_message_idx = {s["message_idx"]: s for s in segments}
    turn_token_end: list[int] = [
        (seg_by_message_idx[idx]["end_token"] if idx in seg_by_message_idx else 0)
        for idx in asst_message_indices
    ]

    def _turn_for_pos(pos: int) -> int:
        for t, end in enumerate(turn_token_end):
            if pos < end:
                return t
        return len(turn_token_end) - 1

    extracted_positions: list[int] = []
    stride_counter = 0
    for pos, m in enumerate(extraction_mask):
        if m == 1:
            if stride_counter % stride == 0:
                extracted_positions.append(pos)
            stride_counter += 1

    return [_turn_for_pos(pos) for pos in extracted_positions]


def edit_labels_by_turn(
    sorted_edits: list[dict],
    edit_labels: list[bool | None],
    n_turns: int,
) -> list[bool | None]:
    """Return the label in effect at each assistant turn 0..n_turns-1.

    Turns before any applicable edit (and with no baseline) get None.
    """
    labels: list[bool | None] = []
    for turn in range(n_turns):
        current_label: bool | None = None
        found_any = False
        for i, edit in enumerate(sorted_edits):
            cidx = edit["cmd_idx"]
            # baseline applies from turn 0; a real edit at cmd_idx N is issued
            # during turn N, but its effect is only visible from turn N+1 onwards.
            edit_turn = 0 if cidx == -1 else cidx + 1
            if edit_turn <= turn:
                current_label = edit_labels[i]
                found_any = True
            else:
                break
        labels.append(current_label if found_any else None)

    return labels


def build_label_sequence(
    segments: list[dict],
    messages: list[dict],
    edits: list[dict],
    extraction_mask: list[int],
    stride: int,
    probe: str,
) -> list[bool | None]:
    """Return one label per extracted position for a named probe.

    Convenience wrapper around map_edit_labels_to_positions that derives
    per-edit labels from the raw edit dicts for 'currently_compiles' and
    'currently_correct'.
    """
    def _label_for_edit(edit: dict) -> bool | None:
        if edit.get("apply_error"):
            return None
        if probe == "currently_compiles":
            c = edit.get("compiles")
            return bool(c) if c is not None else None
        if probe == "currently_correct":
            tr = edit.get("test_results") or {}
            resolved = tr.get("resolved")
            return bool(resolved) if resolved is not None else None
        return None

    sorted_edits = sorted(edits, key=lambda e: e["cmd_idx"])
    edit_labels = [_label_for_edit(e) for e in sorted_edits]
    return map_edit_labels_to_positions(segments, messages, sorted_edits, extraction_mask, stride, edit_labels)
