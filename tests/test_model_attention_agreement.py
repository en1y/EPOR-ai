"""SDPA versus the explicit math attention reference (gate G3).

Tolerances
----------
FP32: ``rtol=1.3e-6, atol=1e-5`` are PyTorch's own FP32 ``assert_close``
defaults.  SDPA and the math path differ only in reduction order, which
observed at about 4e-7 absolute on these logits.

BF16: ``rtol=1.6e-2, atol=3e-2``.  BF16 keeps 8 significant bits, so one unit
in the last place is 2**-7 (about 7.8e-3) for values in [1, 2).  SDPA rounds
intermediates to BF16 while the math reference accumulates in FP32; the
observed gap on these logits is one ULP, and ``atol`` allows a few ULPs of
drift across layers without hiding a masking or scaling bug (those produce
O(1) differences).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402
from epor.models.reference import (  # noqa: E402
    DecoderBlock,
    GroupedQueryAttention,
    attention_backend,
    segment_attention_mask,
    segment_positions,
)

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")

FP32_RTOL = 1.3e-6
FP32_ATOL = 1e-5
BF16_RTOL = 1.6e-2
BF16_ATOL = 3e-2
TOLERANCES = {
    torch.float32: {"rtol": FP32_RTOL, "atol": FP32_ATOL},
    torch.bfloat16: {"rtol": BF16_RTOL, "atol": BF16_ATOL},
}

# Two packed rows: the second holds three documents.
SEGMENTS = torch.tensor([[0] * 20 + [1] * 28, [0] * 5 + [1] * 30 + [2] * 13])


def _tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


def _tokens(batch: int, sequence: int, seed: int = 2) -> Any:
    return torch.randint(3, 260, (batch, sequence), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("qk_norm", [False, True], ids=["control", "qk-norm"])
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "document-causal"])
def test_sdpa_and_math_attention_agree(dtype: Any, qk_norm: bool, packed: bool) -> None:
    model = EporDecoder(_tiny(qk_norm=qk_norm), seed=5).to(dtype).eval()
    tokens = _tokens(*SEGMENTS.shape)
    segment_ids = SEGMENTS if packed else None
    with torch.no_grad():
        optimized = model(tokens, segment_ids=segment_ids).logits
        with attention_backend(model, "math"):
            reference = model(tokens, segment_ids=segment_ids).logits
    assert optimized.dtype == dtype
    torch.testing.assert_close(optimized, reference, **TOLERANCES[dtype])
    assert all(
        module.backend == "sdpa"
        for module in model.modules()
        if isinstance(module, GroupedQueryAttention)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_attention_module_agrees_on_unit_scale_inputs(dtype: Any) -> None:
    model = EporDecoder(_tiny(qk_norm=True), seed=13).to(dtype).eval()
    attention = cast(GroupedQueryAttention, cast(DecoderBlock, model.blocks[0]).attention)
    inputs = torch.randn(2, 48, 64, generator=torch.Generator().manual_seed(3)).to(dtype)
    positions = segment_positions(SEGMENTS)
    mask = segment_attention_mask(SEGMENTS)
    with torch.no_grad():
        optimized = attention(inputs, positions, mask)
        with attention_backend(model, "math"):
            reference = attention(inputs, positions, mask)
    torch.testing.assert_close(optimized, reference, **TOLERANCES[dtype])


def test_segment_positions_restart_per_document() -> None:
    segment_ids = torch.tensor([[7, 7, 7, 3, 3, 9], [0, 0, 0, 0, 0, 0]])
    assert segment_positions(segment_ids).tolist() == [[0, 1, 2, 0, 1, 0], [0, 1, 2, 3, 4, 5]]
    mask = segment_attention_mask(segment_ids)[0, 0]
    assert mask.tolist() == [
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, False, False, False],
        [False, False, False, True, False, False],
        [False, False, False, True, True, False],
        [False, False, False, False, False, True],
    ]


@pytest.mark.parametrize("qk_norm", [False, True], ids=["control", "qk-norm"])
def test_packed_documents_match_running_each_document_alone(qk_norm: bool) -> None:
    model = EporDecoder(_tiny(qk_norm=qk_norm), seed=17).eval()
    documents = [_tokens(1, length, seed) for length, seed in ((11, 1), (7, 2), (14, 3))]
    packed = torch.cat(documents, dim=1)
    segment_ids = torch.cat(
        [torch.full_like(document, index) for index, document in enumerate(documents)], dim=1
    )
    with torch.no_grad():
        packed_logits = model(packed, segment_ids=segment_ids).logits
        start = 0
        for document in documents:
            alone = model(document).logits
            end = start + document.shape[1]
            torch.testing.assert_close(
                packed_logits[:, start:end], alone, rtol=FP32_RTOL, atol=FP32_ATOL
            )
            start = end


def test_one_segment_matches_the_unpacked_causal_path() -> None:
    model = EporDecoder(_tiny(), seed=19).eval()
    tokens = _tokens(2, 24)
    with torch.no_grad():
        plain = model(tokens).logits
        single = model(tokens, segment_ids=torch.zeros_like(tokens)).logits
    torch.testing.assert_close(single, plain, rtol=FP32_RTOL, atol=FP32_ATOL)


def test_other_documents_cannot_influence_a_document() -> None:
    model = EporDecoder(_tiny(), seed=23).eval()
    tokens = _tokens(1, 20)
    segment_ids = torch.tensor([[0] * 8 + [1] * 12])
    changed = tokens.clone()
    changed[0, :8] = _tokens(1, 8, seed=99)
    with torch.no_grad():
        first = model(tokens, segment_ids=segment_ids).logits
        second = model(changed, segment_ids=segment_ids).logits
    # Masked keys get exactly zero probability, so the second document is
    # bit-for-bit unaffected by the first.
    torch.testing.assert_close(first[:, 8:], second[:, 8:], rtol=0, atol=0)


def test_segment_ids_are_validated() -> None:
    model = EporDecoder(_tiny(), seed=1)
    tokens = _tokens(1, 6)
    with pytest.raises(ValueError, match="segment_ids must have the same shape"):
        model(tokens, segment_ids=torch.zeros(1, 5, dtype=torch.int64))
    with pytest.raises(TypeError, match="segment_ids must contain integer"):
        model(tokens, segment_ids=torch.zeros(1, 6))
