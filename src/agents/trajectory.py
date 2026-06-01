from __future__ import annotations

import bisect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_agent_trajectory(
    *,
    task: str,
    result: Any,
    messages: list[dict[str, Any]],
    command_history: list[dict[str, Any]],
    model_name: str,
    base_url: str,
    tokenizer_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_messages = [_clean_message(message) for message in messages]
    artifact = {
        "schema_version": "program-probes.agent_trajectory.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "result": _jsonable(result),
        "model": {
            "served_name": model_name,
            "base_url": base_url,
            "tokenizer_name": tokenizer_name,
        },
        "messages": clean_messages,
        "command_history": _jsonable(command_history),
        "metadata": metadata or {},
    }

    if tokenizer_name is not None:
        artifact["tokenization"] = _tokenize_trajectory(clean_messages, tokenizer_name)

    return artifact


def save_agent_trajectory(
    path: str | Path,
    *,
    task: str,
    result: Any,
    messages: list[dict[str, Any]],
    command_history: list[dict[str, Any]],
    model_name: str,
    base_url: str,
    tokenizer_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    artifact = build_agent_trajectory(
        task=task,
        result=result,
        messages=messages,
        command_history=command_history,
        model_name=model_name,
        base_url=base_url,
        tokenizer_name=tokenizer_name,
        metadata=metadata,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True))
    return out_path


def _tokenize_trajectory(messages: list[dict[str, Any]], tokenizer_name: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chat_messages = []
    for message in messages:
        msg: dict[str, Any] = {
            "role": message.get("role", "user"),
            "content": _content_to_text(message.get("content", "")),
        }
        if message.get("reasoning_content"):
            msg["reasoning_content"] = message["reasoning_content"]
        chat_messages.append(msg)

    try:
        text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_messages)

    try:
        enc = tokenizer(text, return_offsets_mapping=True)
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        segments = _build_segments(chat_messages, text, offsets)
    except Exception:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        segments = _build_segments_no_offsets(chat_messages)

    return {
        "text": text,
        "token_ids": token_ids,
        "segments": segments,
    }


def _build_segments(
    chat_messages: list[dict[str, str]],
    text: str,
    offsets: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    # Build a sorted list of token start positions for bisect lookups.
    # offsets[i] = (char_start, char_end) for token i.
    token_starts = [off[0] for off in offsets]
    token_ends = [off[1] for off in offsets]

    segments = []
    search_from = 0  # advance forward so repeated content is found in order

    for idx, message in enumerate(chat_messages):
        content = message["content"]
        if not content:
            continue

        char_start = text.find(content, search_from)
        if char_start == -1:
            # Content not found verbatim (e.g. template transformed it); skip.
            continue
        char_end = char_start + len(content)
        search_from = char_end

        # First token whose start is >= char_start
        start_token = bisect.bisect_left(token_starts, char_start)
        # First token whose end is > char_end — content ends before this token
        end_token = bisect.bisect_right(token_ends, char_end)

        segments.append({
            "message_idx": idx,
            "role": message["role"],
            "start_token": start_token,
            "end_token": end_token,
        })

    return segments


def _build_segments_no_offsets(chat_messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"message_idx": idx, "role": m["role"], "start_token": None, "end_token": None}
        for idx, m in enumerate(chat_messages)
        if m["content"]
    ]


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        "role": message.get("role"),
        "content": _jsonable(message.get("content", "")),
        "extra": _jsonable(message.get("extra", {})),
    }
    # Preserve reasoning_content so tokenization can faithfully reconstruct the
    # prompt that the model actually saw (some templates, e.g. Laguna, embed
    # thinking blocks from previous turns into the context).
    if message.get("reasoning_content"):
        cleaned["reasoning_content"] = _jsonable(message["reasoning_content"])
    return cleaned


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(_jsonable(content), ensure_ascii=True, sort_keys=True)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)
