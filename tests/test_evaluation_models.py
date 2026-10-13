"""Decoder scoring adapters: the debug byte path and a real tiny kernel checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from evaluation_support import TEST_HARNESS  # noqa: E402
from kernel_support import kernel_config, kernel_settings  # noqa: E402

from epor.data.store import DataStore  # noqa: E402
from epor.evaluation import fixture_suite, load_report, run_suite, store_report  # noqa: E402
from epor.evaluation.decoder import (  # noqa: E402
    debug_byte_scoring_model,
    load_kernel_scoring_model,
)
from epor.models import EporDecoder  # noqa: E402
from epor.models.tokenizer import DebugByteTokenizer  # noqa: E402
from epor.training import pretrain_packed  # noqa: E402


@pytest.fixture(scope="module")
def debug_model() -> EporDecoder:
    return EporDecoder(kernel_config(DebugByteTokenizer.VOCAB_SIZE), seed=3)


def test_loglikelihood_matches_a_manual_forward(debug_model: EporDecoder) -> None:
    scoring = debug_byte_scoring_model(debug_model, batch_size=4)
    tokenizer = DebugByteTokenizer()
    context, continuation = "Question: 2+2?\nAnswer:", " 4"
    ids = [tokenizer.BOS_ID, *tokenizer.encode(context), *tokenizer.encode(continuation)]
    with torch.no_grad():
        logits = debug_model(torch.tensor([ids[:-1]])).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    count = len(tokenizer.encode(continuation))
    expected = sum(
        float(logprobs[position, ids[position + 1]])
        for position in range(len(ids) - 1 - count, len(ids) - 1)
    )
    result = scoring.loglikelihood(context, continuation)
    assert result.sum_logprob == pytest.approx(expected, abs=1e-5)
    greedy = all(
        int(logprobs[position].argmax()) == ids[position + 1]
        for position in range(len(ids) - 1 - count, len(ids) - 1)
    )
    assert result.is_greedy is greedy


def test_batched_scoring_agrees_with_single_requests(debug_model: EporDecoder) -> None:
    requests = [("short", " a"), ("a much longer context here", " bc"), ("x", " yes please")]
    batched = debug_byte_scoring_model(debug_model, batch_size=8).loglikelihood_batch(requests)
    single = debug_byte_scoring_model(debug_model, batch_size=1).loglikelihood_batch(requests)
    for left, right in zip(batched, single, strict=True):
        assert left.sum_logprob == pytest.approx(right.sum_logprob, abs=1e-4)
    with pytest.raises(ValueError, match="continuation"):
        debug_byte_scoring_model(debug_model).loglikelihood("context", "")


def test_long_contexts_are_left_truncated(debug_model: EporDecoder) -> None:
    scoring = debug_byte_scoring_model(debug_model)
    window = scoring.max_length
    result = scoring.loglikelihood("z" * (window * 2), " ok")
    assert math.isfinite(result.sum_logprob)
    with pytest.raises(ValueError, match="does not fit"):
        scoring.loglikelihood("c", "y" * window)


def test_greedy_generation_is_deterministic_and_stops(debug_model: EporDecoder) -> None:
    scoring = debug_byte_scoring_model(debug_model)
    first = scoring.generate("Question: hi\nAnswer:", max_new_tokens=12)
    assert first == scoring.generate("Question: hi\nAnswer:", max_new_tokens=12)
    assert scoring.generate("Question: hi\nAnswer:", max_new_tokens=0) == ""
    if first:
        marker = first[len(first) // 2]
        stopped = scoring.generate("Question: hi\nAnswer:", max_new_tokens=12, stop=[marker])
        assert marker not in stopped and first.startswith(stopped)


def test_debug_model_fixture_suite_report_is_deterministic(debug_model: EporDecoder) -> None:
    scoring = debug_byte_scoring_model(debug_model, model_id="debug-tiny")
    first = run_suite(fixture_suite(), scoring, harness=TEST_HARNESS)
    second = run_suite(fixture_suite(), scoring, harness=TEST_HARNESS)
    assert first.report_id == second.report_id
    assert first.model.kind == "in-memory-decoder"
    assert first.model.config_sha256 == debug_model.config.sha256
    assert first.model.scoring["batch_size"] == 8


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("evaluation"))


def test_kernel_checkpoint_smoke(shards: TokenShardFixture, tmp_path: Path) -> None:
    config = kernel_config(shards.vocab_size, mtp_depth=1)
    settings = kernel_settings(shards.tokenized_id)
    run = tmp_path / "run"
    pretrain_packed(config, shards.data_root, run, settings, max_steps=2)
    scoring = load_kernel_scoring_model(run, shards.data_root)
    identity = scoring.identity
    assert identity.kind == "kernel-checkpoint"
    assert identity.config_sha256 == config.sha256
    assert identity.checkpoint_step == 2
    assert identity.checkpoint_sha256 is not None
    assert identity.tokenizer_id == shards.tokenizer_id

    report = run_suite(fixture_suite(), scoring, harness=TEST_HARNESS)
    assert [s.task_id for s in report.skipped_tasks] == ["fixture-code"]
    loss = report.task("fixture-validation-loss")
    assert loss.metric("perplexity").value == pytest.approx(
        math.exp(loss.metric("main_loss").value or 0.0)
    )
    assert loss.metric("mtp_loss_d1").value is not None
    assert report.seeds == {"evaluation": 0, "model": settings.model_seed, "data": 23}
    build = [source for source in report.data if source.kind == "tokenized-build"]
    assert build[0].source_id == shards.tokenized_id
    assert build[0].task_ids == ["fixture-validation-loss"]
    mc = report.task("fixture-multiple-choice")
    assert mc.calibration is not None and mc.calibration.items == 6

    store = DataStore(shards.data_root)
    store_report(store, report)
    assert load_report(store, report.report_id) == report
    again = run_suite(fixture_suite(), scoring, harness=TEST_HARNESS)
    assert again.report_id == report.report_id
