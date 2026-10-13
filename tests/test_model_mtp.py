from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F  # noqa: E402

from epor.models import EporDecoder, ModelConfig, load_model_config  # noqa: E402
from epor.models.config import (  # noqa: E402
    mtp_logit_bytes_per_token,
    mtp_parameter_count,
    structural_parameter_count,
)
from epor.models.reference import (  # noqa: E402
    IGNORE_INDEX,
    MultiTokenPredictionModule,
    parameter_report,
)

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")
EOS = 2


def _tiny(**overrides: Any) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})


def _document(length: int, seed: int) -> tuple[Any, Any]:
    tokens = torch.randint(3, 260, (1, length), generator=torch.Generator().manual_seed(seed))
    targets = torch.cat((tokens[:, 1:], torch.tensor([[EOS]])), dim=1)
    return tokens, targets


def test_mtp_disabled_keeps_the_single_token_objective() -> None:
    model = EporDecoder(_tiny(), seed=3)
    tokens, targets = _document(12, 1)
    output = model(tokens, targets)
    assert len(model.mtp_modules) == 0
    assert output.mtp_losses == ()
    assert output.loss is output.main_loss
    assert model(tokens).loss is None


def test_mtp_loss_combines_main_and_mean_depth_losses() -> None:
    config = _tiny(mtp_depth=2, mtp_loss_weight=0.25)
    model = EporDecoder(config, seed=3)
    tokens, targets = _document(16, 2)
    output = model(tokens, targets)
    assert output.main_loss is not None and output.loss is not None
    assert len(output.mtp_losses) == 2
    expected = output.main_loss + 0.25 * (output.mtp_losses[0] + output.mtp_losses[1]) / 2
    torch.testing.assert_close(output.loss, expected)
    assert output.logits.shape == (1, 16, config.vocab_size)


def test_depth_one_follows_the_sequential_deepseek_v3_chain() -> None:
    model = EporDecoder(_tiny(mtp_depth=1), seed=5).eval()
    tokens, targets = _document(10, 3)
    with torch.no_grad():
        output = model(tokens, targets)
        hidden = model.token_embeddings(tokens)
        for block in model.blocks:
            hidden = block(hidden)
        module = cast(MultiTokenPredictionModule, model.mtp_modules[0])
        # Position i combines main hidden h_i with Emb(t_{i+1}) and predicts t_{i+2}.
        combined = module.projection(
            torch.cat(
                (
                    module.hidden_norm(hidden[:, :-1]),
                    module.embedding_norm(model.token_embeddings(tokens[:, 1:])),
                ),
                dim=-1,
            )
        )
        logits = model.lm_head(module.output_norm(module.block(combined)))
        expected = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1))
    torch.testing.assert_close(output.mtp_losses[0], expected)


def test_mtp_shares_embedding_and_head_and_trains_the_main_trunk() -> None:
    config = _tiny(mtp_depth=1)
    model = EporDecoder(config, seed=7)
    assert model.lm_head.weight is model.token_embeddings.weight
    assert not any(
        parameter is model.token_embeddings.weight for parameter in model.mtp_modules.parameters()
    )
    tokens, targets = _document(12, 4)
    model(tokens, targets).mtp_losses[0].backward()
    first_block_grad = next(model.blocks[0].parameters()).grad
    assert first_block_grad is not None and bool(first_block_grad.abs().sum() > 0)


@pytest.mark.parametrize("depth", [1, 2])
def test_packed_mtp_never_predicts_across_documents(depth: int) -> None:
    model = EporDecoder(_tiny(mtp_depth=depth), seed=11).eval()
    documents = [_document(9, 5), _document(13, 6)]
    tokens = torch.cat([document[0] for document in documents], dim=1)
    targets = torch.cat([document[1] for document in documents], dim=1)
    segment_ids = torch.tensor([[0] * 9 + [1] * 13])
    with torch.no_grad():
        packed = model(tokens, targets, segment_ids=segment_ids)
        alone = [model(*document) for document in documents]
    for index in range(depth):
        counts = [9 - (index + 1), 13 - (index + 1)]
        expected = sum(
            output.mtp_losses[index] * count for output, count in zip(alone, counts, strict=True)
        ) / sum(counts)
        torch.testing.assert_close(packed.mtp_losses[index], expected, rtol=1e-5, atol=1e-6)


def test_fully_ignored_depth_is_zero_not_nan() -> None:
    model = EporDecoder(_tiny(mtp_depth=1), seed=13)
    tokens, _ = _document(6, 7)
    targets = torch.full_like(tokens, IGNORE_INDEX)
    targets[0, 0] = 5
    output = model(tokens, targets)
    assert float(output.mtp_losses[0].detach()) == 0.0
    assert torch.isfinite(output.loss)


def test_mtp_requires_a_sequence_longer_than_its_depth() -> None:
    model = EporDecoder(_tiny(mtp_depth=2), seed=1)
    tokens, targets = _document(2, 8)
    with pytest.raises(ValueError, match="must exceed mtp_depth"):
        model(tokens, targets)
    assert model(tokens).logits.shape[1] == 2


def test_mtp_parameter_and_memory_overhead_is_reported() -> None:
    control = _tiny()
    config = _tiny(mtp_depth=2, qk_norm=True)
    report = parameter_report(config)
    assert report.measured_total_parameters == structural_parameter_count(config)
    assert report.mtp_parameters == mtp_parameter_count(config) > 0
    assert report.mtp_logit_bytes_per_token == mtp_logit_bytes_per_token(config)
    assert parameter_report(control).mtp_parameters == 0

    # Parameters of the control trunk are unchanged and initialized identically.
    control_state = EporDecoder(control, seed=21).state_dict()
    mtp_state = EporDecoder(_tiny(mtp_depth=2), seed=21).state_dict()
    for name, tensor in control_state.items():
        assert torch.equal(mtp_state[name], tensor)
