"""AdamW parameter groups, warmup/cosine schedule, and global-norm clipping (G3)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402
from epor.training.optim import (  # noqa: E402
    OptimizerSettings,
    build_optimizer,
    clip_gradients,
    learning_rate_at,
    parameter_groups,
)

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def test_roadmap_optimizer_defaults() -> None:
    settings = OptimizerSettings()
    assert (settings.beta1, settings.beta2) == (0.9, 0.95)
    assert settings.weight_decay == 0.1
    assert settings.gradient_clip == 1.0
    assert settings.eps == 1e-8
    with pytest.raises(ValueError, match="warmup_steps"):
        OptimizerSettings(warmup_steps=11, total_steps=10)


def test_warmup_then_cosine_to_floor() -> None:
    settings = OptimizerSettings(
        learning_rate=1e-3, warmup_steps=4, total_steps=12, min_lr_ratio=0.1
    )
    rates = [learning_rate_at(step, settings) for step in range(1, 16)]
    assert rates[:4] == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3])
    decay = rates[3:12]
    assert all(left > right for left, right in itertools.pairwise(decay))
    midpoint = learning_rate_at(8, settings)
    assert midpoint == pytest.approx(1e-4 + 0.9e-3 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert rates[11:] == pytest.approx([1e-4] * 4)
    with pytest.raises(ValueError, match="1-based"):
        learning_rate_at(0, settings)
    flat = OptimizerSettings(learning_rate=1e-3, warmup_steps=0, total_steps=1, min_lr_ratio=1)
    assert learning_rate_at(1, flat) == learning_rate_at(50, flat) == pytest.approx(1e-3)


def test_no_decay_on_norms_and_tied_parameters_once() -> None:
    base = load_model_config(TINY_CONFIG)
    config = ModelConfig.model_validate(
        {**base.model_dump(mode="json"), "qk_norm": True, "mtp_depth": 1}
    )
    model = EporDecoder(config, seed=3)
    groups = parameter_groups(model, 0.1)
    decay, no_decay = groups[0]["params"], groups[1]["params"]
    assert isinstance(decay, list) and isinstance(no_decay, list)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert all(parameter.ndim >= 2 for parameter in decay)
    assert all(parameter.ndim == 1 for parameter in no_decay)
    unique = {id(parameter) for parameter in model.parameters()}
    assert len(decay) + len(no_decay) == len(unique)
    # The tied embedding/output matrix is decayed exactly once.
    assert sum(parameter is model.token_embeddings.weight for parameter in decay) == 1
    norm_ids = {
        id(module.weight) for module in model.modules() if type(module).__name__ == "RMSNorm"
    }
    assert norm_ids and norm_ids <= {id(parameter) for parameter in no_decay}

    optimizer = build_optimizer(model, OptimizerSettings())
    for group in optimizer.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8
        assert group["foreach"] is False


def test_clip_matches_global_norm_and_is_noop_below_threshold() -> None:
    generator = torch.Generator().manual_seed(0)
    parameters = [torch.nn.Parameter(torch.zeros(5, 3)), torch.nn.Parameter(torch.zeros(4))]
    for parameter in parameters:
        parameter.grad = torch.randn(parameter.shape, generator=generator) * 3
    reference = [parameter.grad.clone() for parameter in parameters]  # type: ignore[union-attr]
    expected = torch.linalg.vector_norm(torch.cat([g.flatten() for g in reference]))
    total = clip_gradients(parameters, 1.0)
    assert total == pytest.approx(float(expected), rel=1e-6)
    clipped = torch.linalg.vector_norm(
        torch.cat([p.grad.flatten() for p in parameters])  # type: ignore[union-attr]
    )
    assert float(clipped) == pytest.approx(1.0, rel=1e-5)

    for parameter, grad in zip(parameters, reference, strict=True):
        parameter.grad = grad.clone() * 1e-3
    small = [p.grad.clone() for p in parameters]  # type: ignore[union-attr]
    clip_gradients(parameters, 1.0)
    assert all(torch.equal(p.grad, s) for p, s in zip(parameters, small, strict=True))
