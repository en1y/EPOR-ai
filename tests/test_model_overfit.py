"""Gate G3 smoke check: every model variant can overfit one fixed batch.

A model that cannot memorize 2 x 16 tokens has a wiring bug (masking,
targets, gradient flow), whatever its architecture.  The 0.1-nat threshold is
far below the ~5.56-nat (ln 260) loss at initialization.  Every variant reached
it in under 40 AdamW steps when this test was written; the 100-step budget
leaves headroom while keeping each variant well under a CPU second.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")
OVERFIT_LOSS_THRESHOLD = 0.1
OVERFIT_MAX_STEPS = 100


def _tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


@pytest.mark.parametrize(
    "overrides",
    [{}, {"qk_norm": True}, {"mtp_depth": 1}, {"qk_norm": True, "mtp_depth": 1}],
    ids=["control", "qk-norm", "mtp-1", "qk-norm-mtp-1"],
)
def test_tiny_variant_overfits_a_fixed_packed_batch(overrides: dict[str, Any]) -> None:
    model = EporDecoder(_tiny(**overrides), seed=101, activation_checkpointing=True).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0, foreach=False)
    generator = torch.Generator().manual_seed(12)
    tokens = torch.randint(3, 260, (2, 17), generator=generator)
    inputs, targets = tokens[:, :-1], tokens[:, 1:].clone()
    segment_ids = torch.tensor([[0] * 7 + [1] * 9, [0] * 16])
    targets[0, 6] = 2  # first packed document ends with EOS
    losses: list[float] = []
    mtp_losses: list[float] = []
    for _ in range(OVERFIT_MAX_STEPS):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, targets, segment_ids=segment_ids)
        assert output.loss is not None and output.main_loss is not None
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.main_loss.detach()))
        mtp_losses.extend(float(loss.detach()) for loss in output.mtp_losses[:1])
        if losses[-1] < OVERFIT_LOSS_THRESHOLD:
            break
    assert losses[-1] < OVERFIT_LOSS_THRESHOLD, losses[-5:]
    if overrides.get("mtp_depth"):
        assert mtp_losses[-1] < mtp_losses[0]
