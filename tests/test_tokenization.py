"""Tests for trajectory tokenization.

Verifies that the tokenized representation in saved trajectories matches
exactly what the LLM sees, and that per-message segment boundaries are
correct and model-agnostic.

Requires the cached Qwen/Qwen3-8B tokenizer (no GPU, no network).
"""

import pytest
from transformers import AutoTokenizer

from src.agents.trajectory import _tokenize_trajectory


TOKENIZER_NAME = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _apply_template(tokenizer, messages: list[dict]) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return text, token_ids


def _clean_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"], "extra": m.get("extra", {})} for m in messages]


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

SIMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Fix the bug in foo()."},
    {"role": "assistant", "content": "Sure, here is the fix."},
]

MULTI_TURN_MESSAGES = [
    {"role": "system", "content": "You are an engineer."},
    {"role": "user", "content": "First question."},
    {"role": "assistant", "content": "First answer."},
    {"role": "user", "content": "Second question."},
    {"role": "assistant", "content": "Second answer."},
]

THINK_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Fix the bug."},
    {"role": "assistant", "content": "<think>\nLet me think.\n</think>\n\nHere is my fix."},
]

REPEATED_CONTENT_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Do the thing."},
    {"role": "assistant", "content": "Done."},
    {"role": "user", "content": "Do the thing."},   # same content as earlier user turn
    {"role": "assistant", "content": "Done again."},
]

UNKNOWN_ROLE_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Fix this."},
    {"role": "assistant", "content": "Fixed."},
    {"role": "exit", "content": ""},   # mini-swe-agent exit message, empty content
]


# ---------------------------------------------------------------------------
# token_ids must exactly match apply_chat_template
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("messages", [
    SIMPLE_MESSAGES,
    MULTI_TURN_MESSAGES,
    THINK_MESSAGES,
    REPEATED_CONTENT_MESSAGES,
])
def test_token_ids_match_apply_chat_template(tokenizer, messages):
    result = _tokenize_trajectory(_clean_messages(messages), TOKENIZER_NAME)
    _, expected_ids = _apply_template(tokenizer, messages)
    assert result["token_ids"] == expected_ids, (
        f"token_ids mismatch: got {len(result['token_ids'])} tokens, "
        f"expected {len(expected_ids)}"
    )


# ---------------------------------------------------------------------------
# Segment content must appear in the full text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("messages", [
    SIMPLE_MESSAGES,
    MULTI_TURN_MESSAGES,
    THINK_MESSAGES,
    REPEATED_CONTENT_MESSAGES,
])
def test_segment_content_matches_messages(tokenizer, messages):
    result = _tokenize_trajectory(_clean_messages(messages), TOKENIZER_NAME)
    text = result["text"]
    token_ids = result["token_ids"]

    for span in result["segments"]:
        idx = span["message_idx"]
        expected_content = messages[idx]["content"]
        span_tokens = token_ids[span["start_token"]:span["end_token"]]
        decoded = tokenizer.decode(span_tokens)
        assert expected_content in decoded, (
            f"Content of message {idx} ({messages[idx]['role']!r}) not found in decoded span.\n"
            f"Expected: {expected_content!r}\n"
            f"Decoded:  {decoded!r}"
        )


# ---------------------------------------------------------------------------
# Segments must not overlap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("messages", [
    SIMPLE_MESSAGES,
    MULTI_TURN_MESSAGES,
    THINK_MESSAGES,
    REPEATED_CONTENT_MESSAGES,
])
def test_segments_do_not_overlap(tokenizer, messages):
    result = _tokenize_trajectory(_clean_messages(messages), TOKENIZER_NAME)
    spans = result["segments"]
    for i in range(len(spans) - 1):
        assert spans[i]["end_token"] <= spans[i + 1]["start_token"], (
            f"Spans {i} and {i+1} overlap: "
            f"[{spans[i]['start_token']}, {spans[i]['end_token']}) vs "
            f"[{spans[i+1]['start_token']}, {spans[i+1]['end_token']})"
        )


# ---------------------------------------------------------------------------
# Repeated content: forward scan must find each occurrence in order
# ---------------------------------------------------------------------------

def test_repeated_roles_found_in_order(tokenizer):
    result = _tokenize_trajectory(_clean_messages(REPEATED_CONTENT_MESSAGES), TOKENIZER_NAME)
    spans = result["segments"]
    # Both "Do the thing." user turns must be at different token positions
    user_spans = [s for s in spans if s["role"] == "user"]
    assert len(user_spans) == 2
    assert user_spans[0]["start_token"] < user_spans[1]["start_token"]
    assert user_spans[0]["end_token"] <= user_spans[1]["start_token"]


# ---------------------------------------------------------------------------
# <think> block must land inside the assistant segment
# ---------------------------------------------------------------------------

def test_think_block_inside_assistant_segment(tokenizer):
    result = _tokenize_trajectory(_clean_messages(THINK_MESSAGES), TOKENIZER_NAME)
    token_ids = result["token_ids"]
    assistant_span = next(s for s in result["segments"] if s["role"] == "assistant")
    span_tokens = token_ids[assistant_span["start_token"]:assistant_span["end_token"]]
    decoded = tokenizer.decode(span_tokens)
    assert "<think>" in decoded
    assert "Let me think." in decoded
    assert "Here is my fix." in decoded


# ---------------------------------------------------------------------------
# Unknown / empty-content roles must not crash
# ---------------------------------------------------------------------------

def test_unknown_role_does_not_crash(tokenizer):
    result = _tokenize_trajectory(_clean_messages(UNKNOWN_ROLE_MESSAGES), TOKENIZER_NAME)
    # exit message is included in spans even with empty content (n_tokens=0)
    roles = [s["role"] for s in result["segments"]]
    assert "exit" in roles
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
