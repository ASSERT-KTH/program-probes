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


def build_label_sequence(
    segments: list[dict],
    messages: list[dict],
    edits: list[dict],
    extraction_mask: list[int],
    stride: int,
    probe: str,
) -> list[bool | None]:
    """Return one label per extracted position.

    Parameters
    ----------
    segments:
        trajectory['tokenization']['segments'] — each has role, message_idx,
        start_token, end_token.
    messages:
        trajectory['messages'] — in order; assistant messages correspond 1:1
        with command_history entries.
    edits:
        labels['edits'] from the labeler — each has cmd_idx, compiles,
        test_results (with 'resolved' key).
    extraction_mask:
        Binary vector (same length as token_ids), 1 at assistant-token positions.
    stride:
        Every stride-th masked position is extracted.
    probe:
        Which label to extract: 'currently_compiles' or 'currently_correct'.

    Returns
    -------
    list[bool | None]
        One entry per extracted position.  None means no label is available yet
        (before the baseline edit, if no baseline exists).
    """
    # --- 1. Build assistant turn index -> token range -----------------------
    asst_message_indices = [
        i for i, m in enumerate(messages) if m["role"] == "assistant"
    ]
    seg_by_message_idx = {s["message_idx"]: s for s in segments}

    # turn_token_end[t] = last token index (exclusive) of the t-th assistant turn
    turn_token_end: list[int] = []
    for msg_idx in asst_message_indices:
        seg = seg_by_message_idx.get(msg_idx)
        turn_token_end.append(seg["end_token"] if seg else 0)

    # --- 2. Build cmd_idx -> label value ------------------------------------
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

    # Sort by cmd_idx; baseline (cmd_idx=-1) sorts first naturally
    sorted_edits = sorted(edits, key=lambda e: e["cmd_idx"])

    # --- 3. Compute extracted positions and their turn indices ---------------
    extracted_positions: list[int] = []
    stride_counter = 0
    for pos, m in enumerate(extraction_mask):
        if m == 1:
            if stride_counter % stride == 0:
                extracted_positions.append(pos)
            stride_counter += 1

    # For each extracted position, find which assistant turn it belongs to.
    # turn_token_end[t] is the end token of turn t; we find the first t
    # such that pos < turn_token_end[t].
    def _turn_for_pos(pos: int) -> int:
        for t, end in enumerate(turn_token_end):
            if pos < end:
                return t
        return len(turn_token_end) - 1

    # --- 4. Assign carry-forward labels -------------------------------------
    labels: list[bool | None] = []
    for pos in extracted_positions:
        turn = _turn_for_pos(pos)
        # Find the most recent edit at or before this turn
        current_label: bool | None = None
        found_any = False
        for edit in sorted_edits:
            cidx = edit["cmd_idx"]
            # baseline edit applies from turn 0 onwards
            edit_turn = 0 if cidx == -1 else cidx
            if edit_turn <= turn:
                current_label = _label_for_edit(edit)
                found_any = True
            else:
                break
        labels.append(current_label if found_any else None)

    return labels
