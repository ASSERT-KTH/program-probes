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
    chat_messages = [
        {"role": message.get("role", "user"), "content": _content_to_text(message.get("content", ""))}
        for message in messages
    ]
    message_tokenization = []
    for idx, message in enumerate(chat_messages):
        token_ids = tokenizer.encode(message["content"], add_special_tokens=False)
        message_tokenization.append(
            {
                "message_idx": idx,
                "role": message["role"],
                "token_ids": token_ids,
                "n_tokens": len(token_ids),
            }
        )

    try:
        chat_text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        chat_text = "\n".join(f"{message['role']}: {message['content']}" for message in chat_messages)
    chat_token_ids = tokenizer.encode(chat_text, add_special_tokens=False)

    token_starts, token_ends = [], []
    spans = tokenizer(chat_text, return_offsets_mapping=True).offset_mapping
    for start, end in spans:
        token_starts.append(start)
        token_ends.append(end)

    message_spans = []
    pointer = 0
    for message in chat_messages:
        role = message["role"]
        content = message["content"]
        raw_offset = chat_text.find(content, pointer)
        if raw_offset == -1:
            pointer = 0
            raw_offset = chat_text.find(content, pointer)
        if raw_offset >= 0:
            start_token = bisect.bisect_left(token_starts, raw_offset)
            end_offset = raw_offset + len(content)
            end_token = bisect.bisect_right(token_ends, end_offset)
            if end_token >= len(token_ends):
                end_token = len(token_ends)
            pointer = raw_offset + len(content)
        else:
            start_token = 0
            end_token = 0
        message_spans.append({
            "message_idx": message_spans.__len__(),
            "role": role,
            "start_token": start_token,
            "end_token": end_token,
            "n_tokens": max(0, end_token - start_token),
        })

    return {
        "tokenizer_name": tokenizer_name,
        "chat_text": chat_text,
        "chat_token_ids": chat_token_ids,
        "n_chat_tokens": len(chat_token_ids),
        "messages": message_tokenization,
        "message_spans": message_spans,
    }


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": message.get("role"),
        "content": _jsonable(message.get("content", "")),
        "extra": _jsonable(message.get("extra", {})),
    }


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
