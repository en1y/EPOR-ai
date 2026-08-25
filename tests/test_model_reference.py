from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest

torch = pytest.importorskip("torch")

from epor.models import DebugByteTokenizer, EporDecoder, load_model_config  # noqa: E402
from epor.models.reference import DecoderBlock, GroupedQueryAttention, apply_rope  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def test_debug_tokenizer_round_trips_every_byte() -> None:
    tokenizer = DebugByteTokenizer()
    arbitrary_bytes = bytes(range(256))
    token_ids = tokenizer.encode(arbitrary_bytes, add_bos=True, add_eos=True)
    assert tokenizer.decode_bytes(token_ids) == arbitrary_bytes
    assert tokenizer.vocab_size == 260


def test_model_initialization_is_deterministic_and_preserves_global_rng() -> None:
    config = load_model_config(TINY_CONFIG)
    torch.manual_seed(9001)
    state_before = torch.get_rng_state().clone()
    first = EporDecoder(config, seed=42)
    assert torch.equal(torch.get_rng_state(), state_before)
    second = EporDecoder(config, seed=42)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert first.lm_head.weight is first.token_embeddings.weight
    assert first.token_embeddings.weight.dtype == torch.float32


def test_decoder_is_causal_and_computes_next_token_loss() -> None:
    config = load_model_config(TINY_CONFIG)
    model = EporDecoder(config, seed=7).eval()
    first = torch.tensor([[1, 10, 11, 12, 13]], dtype=torch.int64)
    changed_future = first.clone()
    changed_future[0, -1] = 99
    with torch.no_grad():
        first_output = model(first, torch.roll(first, -1, dims=1))
        changed_output = model(changed_future)
    assert first_output.logits.shape == (1, 5, config.vocab_size)
    assert first_output.loss is not None and torch.isfinite(first_output.loss)
    torch.testing.assert_close(
        first_output.logits[:, :-1], changed_output.logits[:, :-1], rtol=0, atol=0
    )


def test_greedy_generation_respects_context_limit() -> None:
    config = load_model_config(TINY_CONFIG)
    model = EporDecoder(config, seed=11)
    prompt = torch.tensor([[1, 20, 21]], dtype=torch.int64)
    generated = model.generate(prompt, max_new_tokens=4)
    assert generated.shape == (1, 7)


def test_grouped_sdpa_matches_explicit_causal_attention() -> None:
    config = load_model_config(TINY_CONFIG)
    block = cast(DecoderBlock, EporDecoder(config, seed=31).blocks[0])
    attention = cast(GroupedQueryAttention, block.attention).eval()
    generator = torch.Generator().manual_seed(5)
    inputs = torch.randn(2, 6, config.d_model, generator=generator)

    batch, sequence, _ = inputs.shape
    query = attention.query(inputs).view(batch, sequence, config.n_heads, attention.head_dimension)
    key = attention.key(inputs).view(batch, sequence, config.n_kv_heads, attention.head_dimension)
    value = attention.value(inputs).view(
        batch, sequence, config.n_kv_heads, attention.head_dimension
    )
    query, key = apply_rope(
        query.transpose(1, 2),
        key.transpose(1, 2),
        theta=config.rope_theta,
    )
    key = key.repeat_interleave(attention.groups, dim=1)
    value = value.transpose(1, 2).repeat_interleave(attention.groups, dim=1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dimension)
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    probabilities = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    expected = probabilities @ value
    expected = expected.transpose(1, 2).contiguous().view(batch, sequence, -1)
    expected = attention.output(expected)

    torch.testing.assert_close(attention(inputs), expected, rtol=1e-5, atol=1e-6)


def test_tiny_model_overfits_a_fixed_fixture_batch() -> None:
    config = load_model_config(TINY_CONFIG)
    model = EporDecoder(config, seed=101)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    inputs = torch.tensor([[1, 69, 70, 71, 69, 70, 71, 69]], dtype=torch.int64)
    targets = torch.tensor([[69, 70, 71, 69, 70, 71, 69, 2]], dtype=torch.int64)
    losses: list[float] = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, targets).loss
        assert loss is not None
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5
