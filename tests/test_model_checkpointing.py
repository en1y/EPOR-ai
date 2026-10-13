from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def _tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


def _gradients(model: EporDecoder, **inputs: Any) -> tuple[Any, dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    output = model(**inputs)
    output.loss.backward()
    return output.loss.detach(), {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.parametrize(
    "overrides",
    [{}, {"qk_norm": True, "mtp_depth": 2}],
    ids=["control", "qk-norm-mtp"],
)
def test_activation_checkpointing_preserves_loss_and_gradients(overrides: dict[str, Any]) -> None:
    config = _tiny(**overrides)
    plain = EporDecoder(config, seed=31)
    checkpointed = EporDecoder(config, seed=31, activation_checkpointing=True)
    generator = torch.Generator().manual_seed(8)
    tokens = torch.randint(3, 260, (2, 24), generator=generator)
    inputs = {
        "input_ids": tokens,
        "targets": torch.roll(tokens, -1, dims=1),
        "segment_ids": torch.tensor([[0] * 10 + [1] * 14, [0] * 24]),
    }
    plain_loss, plain_grads = _gradients(plain.train(), **inputs)
    checkpoint_loss, checkpoint_grads = _gradients(checkpointed.train(), **inputs)
    torch.testing.assert_close(checkpoint_loss, plain_loss, rtol=0, atol=0)
    assert checkpoint_grads.keys() == plain_grads.keys()
    for name, gradient in plain_grads.items():
        # Recomputation repeats the identical CPU forward, so FP32 gradients
        # agree to well below the FP32 assert_close defaults.
        torch.testing.assert_close(checkpoint_grads[name], gradient, msg=name)


def test_activation_checkpointing_replays_dropout_rng() -> None:
    config = _tiny(dropout=0.1)
    tokens = torch.randint(3, 260, (1, 16), generator=torch.Generator().manual_seed(9))
    inputs = {"input_ids": tokens, "targets": torch.roll(tokens, -1, dims=1)}
    results = []
    for flag in (False, True):
        model = EporDecoder(config, seed=4, activation_checkpointing=flag).train()
        torch.manual_seed(1234)
        results.append(_gradients(model, **inputs))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(results[1][1][name], gradient, msg=name)


def test_activation_checkpointing_is_a_runtime_flag_outside_the_config() -> None:
    config = _tiny()
    model = EporDecoder(config, seed=2, activation_checkpointing=True)
    assert "activation_checkpointing" not in config.model_dump()
    assert model.config.sha256 == config.sha256
    # Inference and no-grad evaluation take the plain path.
    tokens = torch.randint(3, 260, (1, 8), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        torch.testing.assert_close(
            model.eval()(tokens).logits, EporDecoder(config, seed=2).eval()(tokens).logits
        )
