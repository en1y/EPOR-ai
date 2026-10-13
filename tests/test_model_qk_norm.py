from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402
from epor.models.config import structural_parameter_count  # noqa: E402
from epor.models.reference import (  # noqa: E402
    DecoderBlock,
    GroupedQueryAttention,
    parameter_report,
)

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def _tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


def _attention(model: EporDecoder, layer: int = 0) -> GroupedQueryAttention:
    return cast(GroupedQueryAttention, cast(DecoderBlock, model.blocks[layer]).attention)


def test_control_block_has_no_qk_norm_state() -> None:
    model = EporDecoder(_tiny(), seed=3)
    assert _attention(model).query_norm is None
    assert not any("query_norm" in name or "key_norm" in name for name in model.state_dict())


def test_qk_norm_adds_one_head_dimension_weight_per_projection() -> None:
    config = _tiny(qk_norm=True)
    model = EporDecoder(config, seed=3)
    head_dimension = config.d_model // config.n_heads
    attention = _attention(model)
    assert attention.query_norm is not None and attention.key_norm is not None
    assert attention.query_norm.weight.shape == (head_dimension,)
    assert torch.equal(attention.key_norm.weight, torch.ones(head_dimension))

    measured = parameter_report(config).measured_total_parameters
    assert measured == structural_parameter_count(config)
    assert measured - parameter_report(_tiny()).measured_total_parameters == (
        config.n_layers * 2 * head_dimension
    )


def test_qk_norm_init_shares_the_control_weights_and_preserves_global_rng() -> None:
    torch.manual_seed(77)
    state_before = torch.get_rng_state().clone()
    control = EporDecoder(_tiny(), seed=42).state_dict()
    variant = EporDecoder(_tiny(qk_norm=True), seed=42).state_dict()
    assert torch.equal(torch.get_rng_state(), state_before)
    # QK-norm weights are 1-D and consume no initializer draws, so every shared
    # tensor matches the control exactly: an ablation differs only by QK-norm.
    for name, tensor in control.items():
        assert torch.equal(variant[name], tensor)


def test_qk_norm_makes_attention_invariant_to_query_and_key_scale() -> None:
    inputs = torch.randn(1, 10, 64, generator=torch.Generator().manual_seed(4))
    outputs: dict[bool, list[Any]] = {}
    for qk_norm in (False, True):
        attention = _attention(EporDecoder(_tiny(qk_norm=qk_norm), seed=9)).eval()
        with torch.no_grad():
            before = attention(inputs)
            attention.query.weight.mul_(50.0)
            attention.key.weight.mul_(50.0)
            after = attention(inputs)
        outputs[qk_norm] = [before, after]
    torch.testing.assert_close(outputs[True][0], outputs[True][1], rtol=2e-3, atol=1e-4)
    assert not torch.allclose(outputs[False][0], outputs[False][1], rtol=1e-2, atol=1e-3)
