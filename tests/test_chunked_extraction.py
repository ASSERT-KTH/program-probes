"""Test that chunked KV-cache extraction is numerically equivalent to full-sequence extraction.

Property under test: hidden_states(x + y) == hidden_states(x) + hidden_states(y | kv_x)
i.e. chunking a sequence and passing KV cache across chunks gives identical hidden states
to processing the full sequence at once.
"""
import torch
import pytest
from src.models.base import hf_extract_hidden_states_chunked


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


def _full_extract(model, token_ids, layer_indices, mask, stride):
    """Baseline: single full-sequence forward pass, float32 output."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    positions = [i for i, m in enumerate(mask) if m][::stride]
    result = {}
    for li in layer_indices:
        hs = out.hidden_states[li + 1][0, positions]
        result[li] = hs.cpu()
    return result


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_chunked_matches_full(gpt2, chunk_size):
    torch.manual_seed(42)
    seq_len = 100
    vocab_size = gpt2.config.vocab_size
    token_ids = torch.randint(0, vocab_size, (seq_len,)).tolist()

    # GPT2 has 12 transformer blocks; hidden_states[0] is embedding, [1..12] are blocks
    layer_indices = [0, 5, 11]
    mask = [1] * seq_len
    stride = 3

    device = next(gpt2.parameters()).device
    full = _full_extract(gpt2, token_ids, layer_indices, mask, stride)
    chunked = hf_extract_hidden_states_chunked(
        gpt2, token_ids, layer_indices, mask, stride, chunk_size, device
    )

    for li in layer_indices:
        assert full[li].shape == chunked[li].shape, f"shape mismatch at layer {li}"
        assert torch.allclose(full[li], chunked[li], atol=1e-4), (
            f"layer {li}: max diff = {(full[li] - chunked[li]).abs().max().item():.2e}"
        )


def test_chunked_sparse_mask(gpt2):
    """Chunking works correctly when only a subset of positions are masked."""
    torch.manual_seed(7)
    seq_len = 80
    token_ids = torch.randint(0, gpt2.config.vocab_size, (seq_len,)).tolist()
    # Only extract from the second half (simulates assistant-turn mask)
    mask = [0] * 40 + [1] * 40
    layer_indices = [0, 11]
    stride = 1

    device = next(gpt2.parameters()).device
    full = _full_extract(gpt2, token_ids, layer_indices, mask, stride)
    chunked = hf_extract_hidden_states_chunked(
        gpt2, token_ids, layer_indices, mask, stride, chunk_size=32, device=device
    )

    for li in layer_indices:
        assert full[li].shape == chunked[li].shape
        assert torch.allclose(full[li], chunked[li], atol=1e-4)


def test_chunked_stride(gpt2):
    """Stride is applied consistently between full and chunked extraction."""
    torch.manual_seed(13)
    seq_len = 60
    token_ids = torch.randint(0, gpt2.config.vocab_size, (seq_len,)).tolist()
    mask = [1] * seq_len
    layer_indices = [6]
    stride = 5

    device = next(gpt2.parameters()).device
    full = _full_extract(gpt2, token_ids, layer_indices, mask, stride)
    chunked = hf_extract_hidden_states_chunked(
        gpt2, token_ids, layer_indices, mask, stride, chunk_size=20, device=device
    )

    for li in layer_indices:
        assert full[li].shape == chunked[li].shape
        assert torch.allclose(full[li], chunked[li], atol=1e-4)
