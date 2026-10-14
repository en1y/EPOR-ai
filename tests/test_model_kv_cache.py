"""KV-cache inference: prefill + incremental decode match the full forward.

Tolerance: FP32 ``rtol=1e-5, atol=1e-5``.  The cached path computes the same
attention over the same keys in a different order of operations (one query
chunk at a time), so it agrees to accumulated FP32 rounding, not bitwise.
int8 storage is measured, not asserted equal: its error is bounded below.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from test_model_context import LOCAL, tiny, tokens  # noqa: E402

from epor.models import EporDecoder  # noqa: E402
from epor.models.cache import (  # noqa: E402
    DecoderCache,
    dequantize_int8,
    kv_cache_bytes,
    quantize_int8,
)
from epor.models.config import SparseAttention  # noqa: E402

CACHED = {"rtol": 1e-5, "atol": 1e-5}
VARIANTS: dict[str, dict[str, Any]] = {
    "full-gqa": {},
    "qk-norm-mtp": {"qk_norm": True, "mtp_depth": 1},
    "local-global": LOCAL,
    "split-base-yarn": {
        **LOCAL,
        "local_rope_theta": 5_000.0,
        "rope_scaling": {"type": "yarn", "factor": 2, "original_max_context": 64},
    },
    "mla": {"attention_variant": "mla", "mla_kv_rank": 16, "mla_rope_dim": 8},
    "sparse": {"sparse_attention": SparseAttention(top_k=6, select=True).model_dump()},
}


def _decode(model: EporDecoder, sequence: Any, prefill: int, **cache: Any) -> tuple[Any, Any]:
    state = DecoderCache.for_config(model.config, **cache)
    pieces = [model.forward_cached(sequence[:, :prefill], state)]
    for index in range(prefill, sequence.shape[1]):
        pieces.append(model.forward_cached(sequence[:, index : index + 1], state))
    return torch.cat(pieces, dim=1), state


@pytest.mark.parametrize("name", list(VARIANTS))
def test_cached_decoding_matches_the_full_forward(name: str) -> None:
    model = EporDecoder(tiny(**VARIANTS[name]), seed=21).eval()
    sequence = tokens(2, 30)
    with torch.no_grad():
        full = model(sequence).logits
    cached, state = _decode(model, sequence, 12)
    torch.testing.assert_close(cached, full, **CACHED)
    assert state.length == 30


def test_local_layers_keep_a_bounded_ring_and_bytes_match_the_formula() -> None:
    config = tiny(**LOCAL)
    model = EporDecoder(config, seed=23).eval()
    _, state = _decode(model, tokens(1, 40), 5)
    local, global_ = state.layers[0], state.layers[1]
    assert local.positions is not None and len(local.positions) == config.sliding_window
    assert global_.positions is not None and len(global_.positions) == 40
    assert state.nbytes() == kv_cache_bytes(config, 40, dtype_bytes=4)
    full = tiny()
    assert kv_cache_bytes(config, 4096) < kv_cache_bytes(full, 4096)


def test_mla_caches_only_the_latent_and_positional_key() -> None:
    config = tiny(**VARIANTS["mla"])
    _, state = _decode(EporDecoder(config, seed=25).eval(), tokens(1, 20), 4)
    assert set(state.layers[0].streams) == {"latent", "rope_key"}
    assert state.nbytes() == kv_cache_bytes(config, 20, dtype_bytes=4)
    assert kv_cache_bytes(config, 4096) < kv_cache_bytes(tiny(), 4096)


def test_int8_cache_error_is_small_and_bytes_shrink() -> None:
    values = torch.randn(2, 3, 7, 16)
    quantized, scale = quantize_int8(values)
    error = (dequantize_int8(quantized, scale, torch.float32) - values).abs().max()
    assert float(error) <= float(values.abs().max()) / 127 / 2 + 1e-6
    config = tiny(**LOCAL)
    model = EporDecoder(config, seed=27).eval()
    sequence = tokens(1, 30)
    with torch.no_grad():
        full = model(sequence).logits
    cached, state = _decode(model, sequence, 10, quantize="int8")
    assert state.nbytes() == kv_cache_bytes(config, 30, quantize="int8")
    assert state.nbytes() < kv_cache_bytes(config, 30, dtype_bytes=4) / 2
    drift = (cached - full).abs().max()
    assert 0 < float(drift) < 0.05  # measured, documented in CONTEXT.md


def test_cached_decoding_refuses_to_pass_the_configured_context() -> None:
    model = EporDecoder(tiny(), seed=29).eval()
    state = DecoderCache.for_config(model.config)
    model.forward_cached(tokens(1, 120), state)
    with pytest.raises(ValueError, match="exceeds configured context"):
        model.forward_cached(tokens(1, 9), state)
