"""The G6 context suite: position-resolved retrieval, injection, and perplexity by position.

:func:`run_g6` scores a model on generated items (:mod:`epor.context.tasks`)
at each requested length and depth bucket, adds perplexity by position over a
tokenized split, and seals a content-addressed
:class:`~epor.evaluation.report.EvaluationReport` (suite ``epor-g6-context``).
Task results:

``<family>`` (``context-retrieval``)
    ``accuracy`` overall, ``accuracy.length-<L>`` and
    ``accuracy.length-<L>.<bucket>`` for ``beginning``/``middle``/``end``, and
    ``chance``.  ``prompt-injection`` also reports ``injection_followed``.
``position-perplexity``
    mean next-token loss per position bucket of one long sequence at each
    length (``loss.length-<L>.bucket-<k>``); buckets with no data are ``null``.

TTFT, decode throughput, peak RSS, and out-of-memory behaviour are host
measurements and live in the profile report (:mod:`epor.context.probe`),
because evaluation reports carry no durations.  :func:`gate` decides, per
length, whether a report passes the hypothesis thresholds in
``configs/evaluation/g6-context.yaml``; it never passes on missing data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field

from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.evaluation.report import (
    CLAIM_BOUNDARIES,
    DataSource,
    EvaluationReport,
    ItemResult,
    MeasurementScope,
    Metric,
    TaskResult,
    harness_identity,
    seal_report,
)

from .tasks import DEPTHS, FAMILIES, ContextItem, Family, generate_items

PROJECT_ROOT = Path(__file__).resolve().parents[3]
G6_SUITE_PATH = PROJECT_ROOT / "configs" / "evaluation" / "g6-context.yaml"
SUITE_ID = "epor-g6-context"
GENERATOR_VERSION = "epor-g6-generators-v1"
RETRIEVAL_FAMILIES: tuple[Family, ...] = ("passkey", "niah-multikey", "multi-doc-qa", "long-code")


class G6Gate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_retrieval_accuracy: float = Field(ge=0, le=1)
    min_injection_resistance: float = Field(ge=0, le=1)
    max_short_loss_increase: float = Field(ge=0)
    max_short_retrieval_drop: float = Field(ge=0, le=1)
    require_profile: bool = True


class G6Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lengths: tuple[int, ...]
    items_per_cell: int = Field(ge=1)
    seed: int = Field(ge=0)
    perplexity_buckets: int = Field(ge=1, le=64)


class G6Suite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: Literal["epor-g6-context"]
    version: int = Field(ge=1)
    evidence: Literal["hypothesis"]
    defaults: G6Defaults
    gate: G6Gate


def load_g6_suite(path: Path = G6_SUITE_PATH) -> G6Suite:
    return G6Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _metric(name: str, value: float | None, scope: MeasurementScope, higher: bool = True) -> Metric:
    return Metric(name=name, value=value, scope=scope, higher_is_better=higher)


def _score_family(
    family: Family,
    items: Sequence[ContextItem],
    model: Any,
    scope: MeasurementScope,
    on_item: Callable[[], None] | None,
) -> TaskResult:
    results: list[ItemResult] = []
    cells: dict[tuple[int, str], list[float]] = {}
    followed: list[float] = []
    for item in items:
        if on_item is not None:
            on_item()
        context = f"{item.context}\n{item.question}"
        scores = model.loglikelihood_batch([(context, f" {choice}") for choice in item.choices])
        chosen = max(range(len(scores)), key=lambda index: scores[index].sum_logprob)
        correct = float(chosen == item.answer)
        cells.setdefault((item.context_length, item.depth_bucket), []).append(correct)
        details: dict[str, Any] = {
            "length": item.context_length,
            "depth": item.depth,
            "bucket": item.depth_bucket,
            "chosen": chosen,
            "answer": item.answer,
        }
        injected = item.details.get("injected_answer")
        if isinstance(injected, int):
            followed.append(float(chosen == injected))
            details["followed_injection"] = chosen == injected
        results.append(ItemResult(item_id=item.id, score=correct, details=details))
    metrics = [
        _metric("accuracy", _mean([r.score for r in results]), scope),
        _metric("chance", 1.0 / len(items[0].choices), scope),
    ]
    for length in sorted({item.context_length for item in items}):
        values = [r.score for r in results if r.details["length"] == length]
        metrics.append(_metric(f"accuracy.length-{length}", _mean(values), scope))
        for bucket in DEPTHS:
            metrics.append(
                _metric(
                    f"accuracy.length-{length}.{bucket}",
                    _mean(cells.get((length, bucket), [])),
                    scope,
                )
            )
    if family == "prompt-injection":
        metrics.append(_metric("injection_followed", _mean(followed), scope, higher=False))
    return TaskResult(
        task_id=family,
        task_type="context-retrieval",
        format="epor-g6-v1",
        primary_metric="accuracy",
        metrics=metrics,
        items=results,
        settings={"generator": GENERATOR_VERSION, "scoring": "multiple choice by log-likelihood"},
    )


def position_losses(
    decoder: Any, tokens: torch.Tensor, length: int, buckets: int
) -> list[float | None]:
    """Mean next-token loss per position bucket over ``tokens[: length + 1]``."""

    if tokens.numel() < 2:
        return [None] * buckets
    usable = min(length, tokens.numel() - 1)
    with torch.no_grad():
        logits = decoder(tokens[None, :usable]).logits[0].float()
        losses = F.cross_entropy(logits, tokens[1 : usable + 1], reduction="none")
    values: list[float | None] = []
    for bucket in range(buckets):
        low, high = bucket * length // buckets, (bucket + 1) * length // buckets
        part = losses[low : min(high, usable)]
        values.append(float(part.mean()) if part.numel() and low < usable else None)
    return values


def long_sequence(data_root: Path, tokenized_id: str, split: str, length: int) -> torch.Tensor:
    """Model-framed documents of a split, concatenated up to ``length + 1`` tokens."""

    from epor.training.shard_data import ShardCorpus

    corpus = ShardCorpus.open(data_root, tokenized_id, split)  # type: ignore[arg-type]
    pieces: list[torch.Tensor] = []
    total = 0
    for document in range(corpus.document_count):
        if total > length:
            break
        piece = corpus.framed_tokens(document)
        pieces.append(piece)
        total += piece.numel()
    return torch.cat(pieces)[: length + 1] if pieces else torch.empty(0, dtype=torch.int64)


def run_g6(
    model: Any,
    *,
    lengths: Sequence[int],
    families: Sequence[Family] = FAMILIES,
    items_per_cell: int = 2,
    seed: int = 0,
    perplexity: tuple[Path, str, str] | None = None,
    buckets: int = 8,
    scope: MeasurementScope = "proxy-scale",
    suite: G6Suite | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> EvaluationReport:
    """Score ``model`` (a :class:`~epor.evaluation.decoder.DecoderScoringModel`) on G6.

    ``perplexity`` is ``(data_root, tokenized_id, split)`` for perplexity by
    position; without it that task is skipped.
    """

    settings = suite or load_g6_suite()
    longest = max(lengths)
    if longest + 32 > model.max_length:
        raise ValueError(
            f"G6 length {longest} does not fit the model's scoring window {model.max_length}; "
            "evaluate at or below the configured context"
        )
    items = generate_items(
        model.tokenizer, lengths, families=families, items_per_cell=items_per_cell, seed=seed
    )

    def tick() -> None:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("G6 evaluation cancelled")

    tasks = [
        _score_family(family, [item for item in items if item.family == family], model, scope, tick)
        for family in families
    ]
    data = [
        DataSource(
            kind="project-fixture",
            source_id=GENERATOR_VERSION,
            content_sha256=sha256_bytes(
                canonical_json_bytes(
                    [[item.id, item.context, item.question, list(item.choices)] for item in items]
                )
            ),
            task_ids=list(families),
        )
    ]
    if perplexity is not None:
        data_root, tokenized_id, split = perplexity
        metrics: list[Metric] = []
        for length in lengths:
            tick()
            sequence = long_sequence(data_root, tokenized_id, split, length)
            values = position_losses(model.model, sequence, length, buckets)
            known = [value for value in values if value is not None]
            metrics.append(_metric(f"loss.length-{length}", _mean(known), scope, higher=False))
            metrics += [
                _metric(f"loss.length-{length}.bucket-{index}", value, scope, higher=False)
                for index, value in enumerate(values)
            ]
        tasks.append(
            TaskResult(
                task_id="position-perplexity",
                task_type="position-perplexity",
                format="epor-g6-v1",
                primary_metric=f"loss.length-{lengths[0]}",
                metrics=metrics,
                settings={"split": split, "buckets": buckets, "sequence": "concatenated documents"},
            )
        )
        data.append(
            DataSource(
                kind="tokenized-build",
                source_id=tokenized_id,
                content_sha256=tokenized_id,
                task_ids=["position-perplexity"],
            )
        )
    suite_payload = {
        "suite": settings.model_dump(mode="json"),
        "lengths": list(lengths),
        "families": list(families),
        "items_per_cell": items_per_cell,
        "buckets": buckets,
        "generator": GENERATOR_VERSION,
    }
    return seal_report(
        suite_id=SUITE_ID,
        suite_version=settings.version,
        suite_sha256=sha256_bytes(canonical_json_bytes(suite_payload)),
        scope=scope,
        claim_boundary=CLAIM_BOUNDARIES[scope],
        model=model.identity,
        data=data,
        seeds={"items": seed},
        harness=harness_identity(),
        tasks=tasks,
    )


def _value(report: EvaluationReport, task: str, metric: str) -> float | None:
    try:
        return report.task(task).metric(metric).value
    except KeyError:
        return None


def gate(
    report: EvaluationReport,
    length: int,
    *,
    thresholds: G6Gate | None = None,
    baseline: EvaluationReport | None = None,
    base_length: int | None = None,
    profile_status: str | None = None,
) -> dict[str, Any]:
    """Whether ``report`` passes G6 at ``length``; every missing input is a failure."""

    bar = thresholds or load_g6_suite().gate
    reasons: list[str] = []
    for family in RETRIEVAL_FAMILIES:
        for bucket in DEPTHS:
            value = _value(report, family, f"accuracy.length-{length}.{bucket}")
            if value is None:
                reasons.append(f"{family} has no result at {length} tokens ({bucket})")
            elif value < bar.min_retrieval_accuracy:
                reasons.append(
                    f"{family} {bucket} accuracy {value:.2f} < {bar.min_retrieval_accuracy}"
                )
    resistance = _value(report, "prompt-injection", f"accuracy.length-{length}")
    if resistance is None or resistance < bar.min_injection_resistance:
        reasons.append(f"prompt-injection resistance {resistance} < {bar.min_injection_resistance}")
    if baseline is None or base_length is None:
        reasons.append("no pre-extension baseline report for the short-context regression check")
    else:
        before = _value(baseline, "position-perplexity", f"loss.length-{base_length}.bucket-0")
        after = _value(report, "position-perplexity", f"loss.length-{base_length}.bucket-0")
        if before is None or after is None:
            reasons.append(f"no short-context perplexity at {base_length} tokens to compare")
        elif after > before * (1 + bar.max_short_loss_increase):
            reasons.append(f"short-context loss rose from {before:.3f} to {after:.3f}")
        retrieval_before = [
            _value(baseline, family, f"accuracy.length-{base_length}")
            for family in RETRIEVAL_FAMILIES
        ]
        retrieval_after = [
            _value(report, family, f"accuracy.length-{base_length}")
            for family in RETRIEVAL_FAMILIES
        ]
        if None in retrieval_before or None in retrieval_after:
            reasons.append(f"no short-context retrieval at {base_length} tokens to compare")
        else:
            drop = _mean([v for v in retrieval_before if v is not None]) - _mean(  # type: ignore[operator]
                [v for v in retrieval_after if v is not None]
            )
            if drop > bar.max_short_retrieval_drop:
                reasons.append(f"short-context retrieval fell by {drop:.2f}")
    if bar.require_profile and profile_status != "ok":
        reasons.append(f"host profile at {length} tokens is {profile_status or 'missing'}")
    return {"length": length, "passed": not reasons, "reasons": reasons, "evidence": "hypothesis"}


__all__ = [
    "G6_SUITE_PATH",
    "RETRIEVAL_FAMILIES",
    "SUITE_ID",
    "G6Gate",
    "G6Suite",
    "gate",
    "load_g6_suite",
    "long_sequence",
    "position_losses",
    "run_g6",
]
