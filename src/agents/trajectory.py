from __future__ import annotations

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
    """Build a JSON-serializable mini-SWE trajectory artifact.

    The artifact stores the exact mini-SWE messages and bash command history.
    If tokenizer_name is provided, it also stores token ids for each message
    content and for the full chat template when the tokenizer supports it.
    """
    clean_messages = [_clean_message(message) for message in messages]
    artifact = {
        "schema_version": "program-probes.agent_trajectory.v1",
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
        artifact["flat_token_sequence"] = artifact["tokenization"]["flat"]

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

    return {
        "tokenizer_name": tokenizer_name,
        "chat_text": chat_text,
        "chat_token_ids": chat_token_ids,
        "n_chat_tokens": len(chat_token_ids),
        "flat": _build_flat_token_sequence(tokenizer, chat_messages),
        "messages": message_tokenization,
    }


def _build_flat_token_sequence(tokenizer: Any, chat_messages: list[dict[str, str]]) -> dict[str, Any]:
    """Build a simple, continuous token stream without sectioned message arrays."""
    text_parts = []
    tokens = []

    for idx, message in enumerate(chat_messages):
        role = message["role"]
        content = message["content"]
        segment_text = f"<|program_probes_message|>{role}\n{content}\n"
        segment_token_ids = tokenizer.encode(segment_text, add_special_tokens=False)
        start = len(tokens)
        tokens.extend(segment_token_ids)
        text_parts.append(
            {
                "message_idx": idx,
                "role": role,
                "start_token": start,
                "end_token": len(tokens),
                "n_tokens": len(segment_token_ids),
            }
        )

    return {
        "format": "program_probes_flat_messages_v1",
        "text": "".join(f"<|program_probes_message|>{m['role']}\n{m['content']}\n" for m in chat_messages),
        "token_ids": tokens,
        "n_tokens": len(tokens),
        "segments": text_parts,
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
