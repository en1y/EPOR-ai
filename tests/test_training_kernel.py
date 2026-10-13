"""Kernel objective normalization, overfitting, manifests, and evaluation (gate G3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings  # noqa: E402

from epor.models import EporDecoder  # noqa: E402
from epor.training import (  # noqa: E402
    OptimizerSettings,
    PackedBatchStream,
    ShardCorpus,
    evaluate_packed_checkpoint,
    pretrain_packed,
)
from epor.training.kernel import micro_objective, mtp_target_counts  # noqa: E402


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("kernel"))


def test_mtp_objective_is_exact_under_accumulation(shards: TokenShardFixture) -> None:
    """Accumulated micro-objectives give the full-batch MTP-model gradient."""

    config = kernel_config(shards.vocab_size, mtp_depth=2, mtp_loss_weight=0.3)
    corpus = ShardCorpus.open(shards.data_root, shards.tokenized_id, "train", model_config=config)
    stream = PackedBatchStream(corpus, seq_len=16, tokens_per_batch=64, seed=5, micro_batch_size=1)
    # Find a step whose rows hold different numbers of valid MTP targets (a
    # document boundary in some rows), where naive per-micro-batch averaging
    # weights the rows wrongly.
    for _ in range(200):
        step = stream.next_step()
        micro = step.micro_batches
        counts = [mtp_target_counts(batch, 2) for batch in micro]
        if len({tuple(c) for c in counts}) > 1:
            break
    else:
        pytest.fail("fixture produced no step with a document boundary")
    totals = [sum(c[d] for c in counts) for d in range(2)]

    full_model = EporDecoder(config, seed=4)
    whole = (
        torch.cat([b.input_ids for b in micro]),
        torch.cat([b.targets for b in micro]),
        torch.cat([b.segment_ids for b in micro]),
    )
    output = full_model(whole[0], whole[1], segment_ids=whole[2])
    assert output.loss is not None
    output.loss.backward()
    full_grads = {n: p.grad.clone() for n, p in full_model.named_parameters()}  # type: ignore[union-attr]

    accumulated = EporDecoder(config, seed=4)
    naive = EporDecoder(config, seed=4)
    for batch, batch_counts in zip(micro, counts, strict=True):
        out = accumulated(batch.input_ids, batch.targets, segment_ids=batch.segment_ids)
        micro_objective(
            out,
            target_tokens=batch.target_tokens,
            mtp_counts=batch_counts,
            global_target_tokens=step.global_target_tokens,
            global_mtp_counts=totals,
            mtp_loss_weight=config.mtp_loss_weight,
        ).backward()
        plain = naive(batch.input_ids, batch.targets, segment_ids=batch.segment_ids)
        assert plain.loss is not None
        (plain.loss / len(micro)).backward()
    worst = 0.0
    naive_worst = 0.0
    for name, parameter in accumulated.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, full_grads[name], rtol=1e-5, atol=1e-7)
        worst = max(worst, float((parameter.grad - full_grads[name]).abs().max()))
        naive_grad = dict(naive.named_parameters())[name].grad
        assert naive_grad is not None
        naive_worst = max(naive_worst, float((naive_grad - full_grads[name]).abs().max()))
    assert naive_worst > 100 * max(worst, 1e-9)


@pytest.mark.parametrize(
    ("precision", "steps", "threshold"),
    [("fp32", 300, 1.0), ("bf16", 150, 2.5)],
)
def test_tiny_model_overfits_packed_fixture(
    shards: TokenShardFixture, tmp_path: Path, precision: str, steps: int, threshold: float
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(
        shards.tokenized_id,
        seq_len=32,
        tokens_per_batch=256,
        micro_batch_size=None,
        precision=precision,
        checkpoint_every=steps,
        optimizer=OptimizerSettings(
            learning_rate=1e-2, warmup_steps=4, total_steps=steps, weight_decay=0.0
        ),
    )
    result = pretrain_packed(
        config, shards.data_root, tmp_path / precision, settings, max_steps=steps
    )
    start = result.losses[0]
    assert start == pytest.approx(math.log(shards.vocab_size), abs=0.1)
    tail = sum(result.losses[-8:]) / 8
    assert tail < threshold < start


def test_manifest_records_kernel_provenance_and_evaluation(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    config = kernel_config(shards.vocab_size, qk_norm=True, mtp_depth=1)
    settings = kernel_settings(
        shards.tokenized_id,
        precision="bf16",
        activation_checkpointing=True,
        validation_split="validation",
        eval_max_steps=4,
    )
    run = tmp_path / "run"
    result = pretrain_packed(config, shards.data_root, run, settings, max_steps=3)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["command"] == "kernel_train"
    assert manifest["status"] == "succeeded"
    assert manifest["precision"] == "bf16"
    assert manifest["optimizer"] == settings.optimizer.model_dump(mode="json")
    assert manifest["distributed"] == {"strategy": "none", "world_size": 1}
    assert manifest["data_pipeline"]["tokenized_id"] == shards.tokenized_id
    assert len(manifest["data_pipeline"]["fingerprint"]) == 64
    assert manifest["tokenizer_id"] == shards.tokenizer_id
    assert manifest["model_features"]["qk_norm"] is True
    assert manifest["model_features"]["mtp_depth"] == 1
    assert manifest["activation_checkpointing"] is True
    assert manifest["counters"]["tokens_seen"] == 3 * 32
    assert manifest["throughput"]["tokens_per_second"] > 0
    assert manifest["actual_resource_use"]["maximum_rss_bytes"] > 0
    assert manifest["code_revision"]
    assert manifest["checkpoints"][-1]["step"] == 3

    evaluation = manifest["evaluation"]
    assert result.evaluation is not None
    assert evaluation["split"] == "validation"
    assert evaluation["perplexity"] == pytest.approx(math.exp(evaluation["main_loss"]))
    assert len(evaluation["mtp_losses"]) == 1
    assert evaluation["main_loss"] != pytest.approx(evaluation["mtp_losses"][0])

    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [metric["step"] for metric in metrics] == [1, 2, 3]
    first = metrics[0]
    assert first["training_loss"] == pytest.approx(
        first["main_loss"] + config.mtp_loss_weight * first["mtp_losses"][0]
    )
    assert any(metric["grad_norm"] > settings.optimizer.gradient_clip for metric in metrics)

    # Standalone evaluation of the checkpoint uses the main loss for perplexity.
    report = evaluate_packed_checkpoint(
        run, shards.data_root, max_steps=4, report_dir=tmp_path / "reports"
    )
    assert report.evaluation.main_loss == pytest.approx(evaluation["main_loss"], rel=1e-2)
    assert report.evaluation.perplexity == pytest.approx(math.exp(report.evaluation.main_loss))
    assert report.report_path is not None and report.report_path.is_file()
