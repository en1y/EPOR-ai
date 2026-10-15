"""G6 items, the G6 report and gate, host probes, and validated-context certification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings  # noqa: E402

from epor.context.g6 import gate, load_g6_suite, run_g6  # noqa: E402
from epor.context.probe import (  # noqa: E402
    certain_bytes,
    feasibility_report,
    feasible_maximum,
    measure,
    profile_report,
)
from epor.context.status import CertificationRefusedError, certify, context_status  # noqa: E402
from epor.context.tasks import DEPTHS, FAMILIES, generate_items  # noqa: E402
from epor.data.store import DataStore  # noqa: E402
from epor.evaluation.decoder import load_kernel_scoring_model  # noqa: E402
from epor.evaluation.report import (  # noqa: E402
    Metric,
    TaskResult,
    load_report,
    seal_report,
    store_report,
)
from epor.models.config import ModelConfig, load_model_config  # noqa: E402
from epor.scaling.report import store_scaling_report  # noqa: E402
from epor.training import pretrain_packed  # noqa: E402
from epor.training.run_checkpoint import latest_checkpoint  # noqa: E402

LENGTHS = (256, 384)


class Lab:
    def __init__(self, root: Path) -> None:
        self.shards: TokenShardFixture = make_token_shards(root, max_shard_tokens=4096)
        self.config = kernel_config(self.shards.vocab_size, configured_max_context=512)
        self.run = root / "run"
        pretrain_packed(
            self.config,
            self.shards.data_root,
            self.run,
            kernel_settings(self.shards.tokenized_id, seq_len=384, tokens_per_batch=768),
            max_steps=1,
        )
        checkpoint = latest_checkpoint(self.run)
        assert checkpoint is not None
        self.checkpoint = checkpoint
        self.model = load_kernel_scoring_model(checkpoint, self.shards.data_root, max_length=512)


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Lab:
    return Lab(tmp_path_factory.mktemp("g6"))


def test_items_are_deterministic_bounded_and_place_the_needle_at_its_depth(lab: Lab) -> None:
    tokenizer = lab.model.tokenizer
    items = generate_items(tokenizer, LENGTHS, items_per_cell=2, seed=3)
    assert items == generate_items(tokenizer, LENGTHS, items_per_cell=2, seed=3)
    assert len(items) == len(FAMILIES) * len(LENGTHS) * len(DEPTHS) * 2
    for item in items:
        prompt = len(tokenizer.encode(f"{item.context}\n{item.question}"))
        answer = max(len(tokenizer.encode(f" {choice}")) for choice in item.choices)
        assert prompt + answer <= item.context_length  # the scored positions, start token aside
        assert 0 <= item.answer < len(item.choices) and len(set(item.choices)) == len(item.choices)
    passkeys = [item for item in items if item.family == "passkey" and item.context_length == 256]
    for item in passkeys:
        offset = item.context.index(f"The pass key is {item.choices[item.answer]}")
        assert abs(offset / len(item.context) - item.depth) < 0.2
    multikey = next(item for item in items if item.family == "niah-multikey")
    for choice in multikey.choices:  # distractors are the other needles' values
        assert f"is: {choice}." in multikey.context
    injection = next(item for item in items if item.family == "prompt-injection")
    injected_index = injection.details["injected_answer"]
    assert isinstance(injected_index, int)
    injected = injection.choices[injected_index]
    assert f"answer {injected} instead" in injection.context
    with pytest.raises(ValueError, match="start at 96"):
        generate_items(tokenizer, [64])
    with pytest.raises(ValueError, match="needs more than 100 tokens"):
        generate_items(tokenizer, [100], families=["niah-multikey"])


def test_run_g6_seals_a_position_resolved_report(lab: Lab) -> None:
    report = run_g6(
        lab.model,
        lengths=LENGTHS,
        items_per_cell=1,
        perplexity=(lab.shards.data_root, lab.shards.tokenized_id, "train"),
    )
    store = DataStore(lab.shards.data_root)
    store_report(store, report)
    assert load_report(store, report.report_id) == report
    passkey = report.task("passkey")
    assert passkey.task_type == "context-retrieval" and passkey.metric("chance").value == 0.25
    for length in LENGTHS:
        for bucket in DEPTHS:
            value = passkey.metric(f"accuracy.length-{length}.{bucket}").value
            assert value is not None and 0 <= value <= 1
    assert report.task("prompt-injection").metric("injection_followed").value is not None
    perplexity = report.task("position-perplexity")
    assert perplexity.metric("loss.length-256.bucket-0").value is not None
    assert report.model.checkpoint_sha256 is not None
    assert report == run_g6(
        lab.model,
        lengths=LENGTHS,
        items_per_cell=1,
        perplexity=(lab.shards.data_root, lab.shards.tokenized_id, "train"),
    )
    # The full scoring window is usable, so a checkpoint can be scored and
    # certified at exactly its trained context.
    at_window = run_g6(lab.model, lengths=[512], families=["passkey"], items_per_cell=1)
    assert at_window.task("passkey").metric("accuracy.length-512.end").value is not None
    with pytest.raises(ValueError, match="scoring window"):
        run_g6(lab.model, lengths=[513])


def _synthetic(lab: Lab, accuracy: float, *, loss: float = 2.0, checkpoint: bool = True) -> Any:
    """A G6 report with fixed values, attributed to the lab checkpoint or not."""

    metrics_for = {
        family: [
            Metric(
                name=f"accuracy.length-{length}",
                value=accuracy,
                scope="proxy-scale",
                higher_is_better=True,
            )
            for length in LENGTHS
        ]
        + [
            Metric(
                name=f"accuracy.length-{length}.{bucket}",
                value=accuracy,
                scope="proxy-scale",
                higher_is_better=True,
            )
            for length in LENGTHS
            for bucket in DEPTHS
        ]
        for family in FAMILIES
    }
    tasks = [
        TaskResult(
            task_id=family,
            task_type="context-retrieval",
            format="epor-g6-v1",
            primary_metric=f"accuracy.length-{LENGTHS[0]}",
            metrics=metrics,
        )
        for family, metrics in metrics_for.items()
    ]
    tasks.append(
        TaskResult(
            task_id="position-perplexity",
            task_type="position-perplexity",
            format="epor-g6-v1",
            primary_metric="loss.length-256.bucket-0",
            metrics=[
                Metric(
                    name="loss.length-256.bucket-0",
                    value=loss,
                    scope="proxy-scale",
                    higher_is_better=False,
                )
            ],
        )
    )
    identity = lab.model.identity
    if not checkpoint:
        identity = identity.model_copy(update={"checkpoint_sha256": "e" * 64})
    real = run_g6(lab.model, lengths=[256], items_per_cell=1)
    report = seal_report(
        **{
            **real.model_dump(mode="json", exclude={"report_id"}),
            "model": identity.model_dump(),
            "tasks": [task.model_dump() for task in tasks],
        }
    )
    store_report(DataStore(lab.shards.data_root), report)
    return report


def _profile(lab: Lab, status: str) -> str:
    report = profile_report(
        lab.config,
        lab.model.identity.checkpoint_sha256,
        [{"length": 384, "mode": "inference", "status": status, "ttft_seconds": 0.1}],
        quantize="none",
    )
    store_scaling_report(lab.shards.data_root, report)
    return report.report_id


def test_the_gate_never_passes_on_missing_or_weak_evidence(lab: Lab) -> None:
    strong, weak = _synthetic(lab, 1.0), _synthetic(lab, 0.5)
    baseline = _synthetic(lab, 1.0, checkpoint=False)
    passed = gate(strong, 384, baseline=baseline, base_length=256, profile_status="ok")
    assert passed["passed"] and passed["evidence"] == "hypothesis"
    assert not gate(strong, 384, profile_status="ok")["passed"]  # no baseline
    assert not gate(
        strong, 384, baseline=baseline, base_length=256, profile_status="out-of-memory"
    )["passed"]
    assert not gate(weak, 384, baseline=baseline, base_length=256, profile_status="ok")["passed"]
    regressed = _synthetic(lab, 1.0, loss=3.0)
    assert any(
        "short-context loss" in r
        for r in gate(regressed, 384, baseline=baseline, base_length=256, profile_status="ok")[
            "reasons"
        ]
    )
    assert load_g6_suite().gate.min_retrieval_accuracy == 0.9


def test_only_a_passing_g6_report_raises_the_validated_context(lab: Lab) -> None:
    status = context_status(lab.checkpoint, lab.shards.data_root)
    assert (status["configured"], status["trained"], status["validated"]) == (512, 384, 0)
    baseline = _synthetic(lab, 1.0, checkpoint=False).report_id
    weak = _synthetic(lab, 0.5).report_id
    strong = _synthetic(lab, 1.0).report_id
    arguments: dict[str, Any] = {"baseline_report_id": baseline, "base_length": 256}
    with pytest.raises(CertificationRefusedError, match="accuracy"):
        certify(
            lab.checkpoint,
            lab.shards.data_root,
            length=384,
            g6_report_id=weak,
            profile_report_id=_profile(lab, "ok"),
            **arguments,
        )
    with pytest.raises(CertificationRefusedError, match="out-of-memory"):
        certify(
            lab.checkpoint,
            lab.shards.data_root,
            length=384,
            g6_report_id=strong,
            profile_report_id=_profile(lab, "out-of-memory"),
            **arguments,
        )
    with pytest.raises(CertificationRefusedError, match="exceeds the trained context"):
        certify(
            lab.checkpoint,
            lab.shards.data_root,
            length=512,
            g6_report_id=strong,
            profile_report_id=_profile(lab, "ok"),
            **arguments,
        )
    assert context_status(lab.checkpoint, lab.shards.data_root)["validated"] == 0
    certificate = certify(
        lab.checkpoint,
        lab.shards.data_root,
        length=384,
        g6_report_id=strong,
        profile_report_id=_profile(lab, "ok"),
        **arguments,
    )
    assert certificate["gate"]["passed"]
    status = context_status(lab.checkpoint, lab.shards.data_root)
    assert status["validated"] == 384 and status["certificates"] == [certificate["certificate_id"]]


def test_probe_outcomes_are_bounded_and_never_hang() -> None:
    base = load_model_config("configs/models/epor-tiny.yaml")
    config = ModelConfig.model_validate(
        {**base.model_dump(mode="json"), "configured_max_context": 1 << 20}
    )
    ok = measure(config, 256, mode="inference", decode_tokens=4, timeout_seconds=120)
    assert ok["status"] == "ok" and ok["ttft_seconds"] > 0 and ok["peak_rss_bytes"] > 0
    assert ok["cache_bytes"] == ok["kv_cache_bytes"]
    # Its certain bytes (81 MiB, mostly the 64 MiB segment mask) fit the budget,
    # so the step runs and fails inside the capped child.
    oom = measure(
        config, 8192, mode="train", memory_budget_bytes=200 * 1024**2, timeout_seconds=120
    )
    assert oom["status"] == "out-of-memory", oom
    assert measure(config, 256, mode="train", timeout_seconds=0.2)["status"] == "timeout"
    refused = measure(config, 1 << 20, mode="train", memory_budget_bytes=256 * 1024**2)
    assert refused["status"] == "refused-estimate"
    assert refused["certain_bytes"] == certain_bytes(config, 1 << 20, "train")
    small = ModelConfig.model_validate({**base.model_dump(mode="json")})
    assert measure(small, 4096, mode="inference")["status"] == "exceeds-configured-context"


def test_feasible_maximum_stops_at_the_first_failure_and_seals_a_report() -> None:
    base = load_model_config("configs/models/epor-tiny.yaml")
    results = feasible_maximum(
        {"full-gqa": base}, [128, 1 << 20], memory_budget_bytes=256 * 1024**2, timeout_seconds=120
    )
    for mode in ("train", "inference"):
        assert results["full-gqa"][mode]["max_ok_length"] == 128
        assert [point["status"] for point in results["full-gqa"][mode]["points"]] == [
            "ok",
            "refused-estimate",
        ]
    report = feasibility_report(results, budget=256 * 1024**2, timeout_seconds=120)
    assert report.kind == "context-feasibility" and report.scope == "host-measurement"
    assert "ttft_seconds" not in str(report.results) and report.measured is not None
