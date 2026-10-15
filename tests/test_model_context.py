"""v0.0.8 positional extension, local/global attention, MLA, and the sparse retrofit.

Agreement tolerances follow ``test_model_attention_agreement.py``: FP32
``rtol=1.3e-6, atol=1e-5`` (PyTorch's FP32 defaults; SDPA and the explicit
reference differ only in reduction order).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402
from epor.models.config import (  # noqa: E402
    HASH_NEUTRAL_DEFAULTS,
    RopeScaling,
    SparseAttention,
    structural_parameter_count,
)
from epor.models.reference import (  # noqa: E402
    DecoderBlock,
    GroupedQueryAttention,
    attention_backend,
    local_attention_mask,
    rope_frequencies,
)

TINY = Path("configs/models/epor-tiny.yaml")
FP32 = {"rtol": 1.3e-6, "atol": 1e-5}
LOCAL = {"sliding_window": 8, "local_layers_per_global": 1}


def tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(TINY)
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


def tokens(batch: int, sequence: int, seed: int = 2) -> Any:
    return torch.randint(3, 260, (batch, sequence), generator=torch.Generator().manual_seed(seed))


# --- configuration ------------------------------------------------------------------


def test_new_fields_are_hash_neutral_at_their_defaults() -> None:
    config = tiny()
    payload = config.canonical_json()
    for name in ("rope_scaling", "sliding_window", "attention_variant", "sparse_attention"):
        assert name in HASH_NEUTRAL_DEFAULTS and f'"{name}"' not in payload
    assert config.sha256 == "fd9a3df88eb3700dbf449afebcb77f2f9dba6278fec1f0631ea0750ce51a63c9"
    assert tiny(**LOCAL).sha256 != config.sha256
    scaled = tiny(rope_scaling={"type": "yarn", "factor": 4, "original_max_context": 64})
    assert '"rope_scaling"' in scaled.canonical_json()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sliding_window": 8}, "set together"),
        ({"local_layers_per_global": 2, "sliding_window": 8}, "no global attention layer"),
        ({"local_rope_theta": 1e4}, "needs local"),
        ({**LOCAL, "sliding_window": 4096}, "cannot exceed configured_max_context"),
        ({"attention_variant": "mla", "mla_kv_rank": 8, "mla_rope_dim": 3}, "even mla_rope_dim"),
        ({"mla_kv_rank": 8}, "apply to attention_variant 'mla'"),
        (
            {"attention_variant": "mla", "mla_kv_rank": 8, "mla_rope_dim": 4, "qk_norm": True},
            "does not implement QK-norm",
        ),
        (
            {"rope_scaling": {"type": "linear", "factor": 2, "original_max_context": 4096}},
            "original_max_context exceeds",
        ),
        (
            {
                "rope_scaling": {
                    "type": "ntk",
                    "factor": 2,
                    "original_max_context": 64,
                    "beta_fast": 8,
                }
            },
            "yarn only",
        ),
    ],
)
def test_invalid_context_configurations_are_refused(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tiny(**overrides)


def test_layer_pattern_and_split_rope_base() -> None:
    config = load_model_config(TINY).model_copy()
    config = ModelConfig.model_validate(
        {
            **config.model_dump(mode="json"),
            "n_layers": 6,
            "sliding_window": 16,
            "local_layers_per_global": 5,
            "rope_theta": 1_000_000.0,
            "local_rope_theta": 10_000.0,
            "rope_scaling": {"type": "linear", "factor": 2, "original_max_context": 64},
        }
    )
    assert [config.layer_is_local(i) for i in range(6)] == [True] * 5 + [False]
    model = EporDecoder(config, seed=1)
    local = cast(GroupedQueryAttention, cast(DecoderBlock, model.blocks[0]).attention)
    global_ = cast(GroupedQueryAttention, cast(DecoderBlock, model.blocks[5]).attention)
    assert (local.rope_theta, local.window, local.rope_scaling) == (10_000.0, 16, None)
    assert global_.rope_theta == 1_000_000.0 and global_.window is None
    assert global_.rope_scaling is not None


# --- RoPE scaling -------------------------------------------------------------------


def test_rope_scaling_methods() -> None:
    plain, factor = rope_frequencies(16, theta=10_000.0)
    reference = 10_000.0 ** (-torch.arange(0, 16, 2, dtype=torch.float32) / 16)
    assert factor == 1.0 and torch.equal(plain, reference)
    linear, _ = rope_frequencies(
        16, theta=1e4, scaling=RopeScaling(type="linear", factor=4, original_max_context=64)
    )
    torch.testing.assert_close(linear, plain / 4)
    ntk, _ = rope_frequencies(
        16, theta=1e4, scaling=RopeScaling(type="ntk", factor=4, original_max_context=64)
    )
    torch.testing.assert_close(ntk, (1e4 * 4 ** (16 / 14)) ** (-torch.arange(0, 16, 2) / 16))
    assert float(ntk[0]) == 1.0 and float(ntk[-1]) < float(plain[-1])
    yarn, temperature = rope_frequencies(
        16, theta=1e4, scaling=RopeScaling(type="yarn", factor=4, original_max_context=64)
    )
    assert temperature == pytest.approx(0.1 * math.log(4) + 1)
    rotations = 64 * plain / (2 * math.pi)
    assert torch.equal(yarn[rotations > 32], plain[rotations > 32])  # high frequencies kept
    torch.testing.assert_close(yarn[rotations < 1], plain[rotations < 1] / 4)  # low interpolated


def test_linear_scaling_equals_dividing_positions() -> None:
    scaled = EporDecoder(
        tiny(rope_scaling={"type": "linear", "factor": 2, "original_max_context": 64}), seed=3
    ).eval()
    plain = EporDecoder(tiny(), seed=3).eval()
    attention = cast(GroupedQueryAttention, cast(DecoderBlock, plain.blocks[0]).attention)
    hidden = torch.randn(1, 10, 64)
    positions = torch.arange(10)[None]
    with torch.no_grad():
        attention_scaled = cast(
            GroupedQueryAttention, cast(DecoderBlock, scaled.blocks[0]).attention
        )
        attention_scaled.load_state_dict(attention.state_dict())
        # Doubling the positions under 2x interpolation recovers the plain rotation.
        torch.testing.assert_close(
            attention_scaled(hidden, positions * 2), attention(hidden, positions), **FP32
        )


# --- local/global attention ---------------------------------------------------------


def test_local_mask_is_causal_windowed_and_segment_bounded() -> None:
    mask = local_attention_mask(5, 2)[0, 0]
    assert mask.int().tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]
    packed = local_attention_mask(5, 3, segment_ids=torch.tensor([[0, 0, 0, 1, 1]]))[0, 0]
    assert not packed[3, :3].any() and packed[4, 3]


@pytest.mark.parametrize("qk_norm", [False, True], ids=["control", "qk-norm"])
def test_local_sdpa_matches_the_math_reference_with_packing(qk_norm: bool) -> None:
    model = EporDecoder(tiny(qk_norm=qk_norm, mtp_depth=1, **LOCAL), seed=5).eval()
    batch = tokens(2, 40)
    segments = torch.tensor([[0] * 15 + [1] * 25, [0] * 40])
    targets = torch.roll(batch, -1, dims=1)
    with torch.no_grad():
        sdpa = model(batch, targets, segment_ids=segments)
        with attention_backend(model, "math"):
            math_out = model(batch, targets, segment_ids=segments)
    torch.testing.assert_close(sdpa.logits, math_out.logits, **FP32)
    assert len(sdpa.mtp_losses) == 1 and torch.isfinite(sdpa.loss)


def test_a_packed_document_matches_running_it_alone_under_a_window() -> None:
    model = EporDecoder(tiny(**LOCAL), seed=7).eval()
    first, second = tokens(1, 13, 1), tokens(1, 21, 2)
    packed = torch.cat((first, second), dim=1)
    segments = torch.tensor([[0] * 13 + [1] * 21])
    with torch.no_grad():
        logits = model(packed, segment_ids=segments).logits
        torch.testing.assert_close(logits[:, 13:], model(second).logits, **FP32)


def test_a_window_covering_the_sequence_equals_full_attention() -> None:
    wide = EporDecoder(tiny(sliding_window=128, local_layers_per_global=1), seed=9).eval()
    full = EporDecoder(tiny(), seed=9).eval()
    batch = tokens(1, 30)
    with torch.no_grad():
        torch.testing.assert_close(wide(batch).logits, full(batch).logits, **FP32)


# --- MLA and sparse attention -------------------------------------------------------


def test_mla_counts_match_and_sdpa_matches_math() -> None:
    config = tiny(attention_variant="mla", mla_kv_rank=16, mla_rope_dim=8)
    model = EporDecoder(config, seed=11).eval()
    assert sum(p.numel() for p in model.parameters()) == structural_parameter_count(config)
    batch = tokens(1, 24)
    with torch.no_grad():
        sdpa = model(batch).logits
        with attention_backend(model, "math"):
            torch.testing.assert_close(model(batch).logits, sdpa, **FP32)


def test_sparse_retrofit_counts_selection_and_indexer_loss() -> None:
    dense = tiny(sparse_attention=SparseAttention(top_k=128).model_dump())
    model = EporDecoder(dense, seed=13)
    assert sum(p.numel() for p in model.parameters()) == structural_parameter_count(dense)
    batch = tokens(1, 32)
    targets = torch.roll(batch, -1, dims=1)
    output = model(batch, targets)
    assert output.aux_loss is not None and float(output.aux_loss) >= 0
    torch.testing.assert_close(output.loss, output.main_loss + output.aux_loss)
    plain = EporDecoder(tiny(), seed=13)
    plain.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        # Without selection (warm-up) attention stays dense.
        torch.testing.assert_close(model(batch).logits, plain(batch).logits, **FP32)
        selective = EporDecoder(
            tiny(sparse_attention=SparseAttention(top_k=4, select=True).model_dump()), seed=13
        )
        selective.load_state_dict(model.state_dict())
        assert not torch.allclose(selective(batch).logits, plain(batch).logits, atol=1e-3)
        everything = EporDecoder(
            tiny(sparse_attention=SparseAttention(top_k=1024, select=True).model_dump()), seed=13
        )
        everything.load_state_dict(model.state_dict())
        torch.testing.assert_close(everything(batch).logits, plain(batch).logits, **FP32)
