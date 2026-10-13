"""Reward-overoptimization monitors for a preference-trained policy.

Three signals, each compared with explicit alert/abort thresholds from
``configs/alignment/monitors.yaml`` (every threshold is a ``hypothesis``):

``kl_per_token``
    Mean per-token ``log pi - log ref`` on the policy's own greedy replies to
    held-out prompts: a sample estimate of KL(pi || ref) along the policy's
    trajectory.  Measured, not a bound.
``proxy_minus_audited``
    Proxy reward accuracy — the fraction of held-out preference pairs whose
    DPO implicit reward ``beta * (log-ratio(chosen) - log-ratio(rejected))``
    favours the chosen reply — minus the policy's agreement with
    *human-audited* judge decisions (the fraction of audited pairs where the
    policy prefers the reply the human endorsed).  ``unknown`` until audited
    pairs exist; never filled in by the judge alone.
``length_ratio``
    Mean policy reply length over the reference's on the same prompts; drift
    in either direction (ratio or its inverse) is checked.

The report is content-addressed and stored under
``<data-root>/alignment/reports/overoptimization/``.  Its status is the worst
of its monitors (``ok`` < ``unknown`` < ``alert`` < ``abort``); ``abort``
blocks the policy from being treated as a promotion candidate until a human
reviews it.  Tiny proxy models make every number here a pipeline
measurement, not an alignment result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from epor.data.store import canonical_json_bytes, sha256_bytes

from .chat import ChatMessage
from .records import AlignmentStore, Sampling

MONITORS_PATH = Path(__file__).resolve().parents[3] / "configs" / "alignment" / "monitors.yaml"
Status = Literal["ok", "unknown", "alert", "abort"]
_ORDER = {"ok": 0, "unknown": 1, "alert": 2, "abort": 3}


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alert: float = Field(gt=0.0)
    abort: float = Field(gt=0.0)


class JudgeAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_disagreement: float = Field(ge=0.0, le=1.0)
    min_per_stratum: int = Field(ge=1)


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str
    version: int
    evidence: Literal["hypothesis"]
    kl_per_token: Threshold
    proxy_minus_audited: Threshold
    length_ratio: Threshold
    dpo_step_logratio: Threshold
    judge_admission: JudgeAdmission

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


def load_monitor_config(path: Path = MONITORS_PATH) -> MonitorConfig:
    return MonitorConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def classify(value: float | None, threshold: Threshold) -> Status:
    if value is None:
        return "unknown"
    if value >= threshold.abort:
        return "abort"
    return "alert" if value >= threshold.alert else "ok"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def overoptimization_report(
    store: AlignmentStore,
    policy: Any,
    reference: Any,
    *,
    prompts: Sequence[Sequence[ChatMessage]],
    held_out_pairs: Sequence[tuple[Sequence[ChatMessage], str, str]],
    audited_pairs: Sequence[tuple[Sequence[ChatMessage], str, str]],
    beta: float,
    config: MonitorConfig | None = None,
    max_new_tokens: int = 32,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Measure the three monitors and store the report; returns ``(report_id, report)``.

    ``held_out_pairs`` are ``(prompt, chosen, rejected)`` from a preference
    dataset's validation split; ``audited_pairs`` are ``(prompt, endorsed,
    other)`` where a human verdict decided which reply is endorsed.
    """

    settings = config or load_monitor_config()
    greedy = Sampling(decoding="greedy", max_new_tokens=max_new_tokens)
    logratios: list[float] = []
    policy_lengths: list[int] = []
    reference_lengths: list[int] = []
    for messages in prompts:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("monitor cancelled")
        text, ids = policy.generate(messages, greedy)
        _, reference_ids = reference.generate(messages, greedy)
        policy_lengths.append(len(ids))
        reference_lengths.append(len(reference_ids))
        ((pol, count),) = policy.response_logprobs([(messages, text)])
        ((ref, _),) = reference.response_logprobs([(messages, text)])
        logratios.append((pol - ref) / count)

    def preferences(pairs: Sequence[tuple[Sequence[ChatMessage], str, str]]) -> list[float]:
        wins: list[float] = []
        for messages, first, second in pairs:
            pol = policy.response_logprobs([(messages, first), (messages, second)])
            ref = reference.response_logprobs([(messages, first), (messages, second)])
            margin = beta * ((pol[0][0] - ref[0][0]) - (pol[1][0] - ref[1][0]))
            wins.append(float(margin > 0))
        return wins

    proxy = _mean(preferences(held_out_pairs))
    audited = _mean(preferences(audited_pairs))
    kl = _mean(logratios)
    policy_mean, reference_mean = _mean(policy_lengths), _mean(reference_lengths)
    ratio = None
    if policy_mean is not None and reference_mean:
        ratio = policy_mean / reference_mean
    drift = None if ratio is None or ratio == 0 else max(ratio, 1 / ratio)
    gap = None if proxy is None or audited is None else proxy - audited
    monitors = {
        "kl_per_token": {
            "value": kl,
            "status": classify(kl, settings.kl_per_token),
            "threshold": settings.kl_per_token.model_dump(),
            "prompts": len(prompts),
        },
        "proxy_minus_audited": {
            "value": gap,
            "proxy_reward_accuracy": proxy,
            "audited_agreement": audited,
            "held_out_pairs": len(held_out_pairs),
            "audited_pairs": len(audited_pairs),
            "status": classify(gap, settings.proxy_minus_audited),
            "threshold": settings.proxy_minus_audited.model_dump(),
        },
        "length_ratio": {
            "value": ratio,
            "drift": drift,
            "policy_mean_reply_ids": policy_mean,
            "reference_mean_reply_ids": reference_mean,
            "status": classify(drift, settings.length_ratio),
            "threshold": settings.length_ratio.model_dump(),
        },
    }
    status = max((item["status"] for item in monitors.values()), key=lambda s: _ORDER[s])
    report = {
        "schema_version": 1,
        "kind": "overoptimization",
        "policy": policy.generator.model_dump(mode="json"),
        "reference": reference.generator.model_dump(mode="json"),
        "beta": beta,
        "monitors": monitors,
        "status": status,
        "thresholds": {"config_sha256": settings.sha256, "evidence": settings.evidence},
        "created_at": datetime.now(UTC).isoformat(),
        "claim": (
            "proxy-scale pipeline measurement; thresholds are hypotheses and no value here is "
            "evidence of alignment or of its absence"
        ),
    }
    return store_report(store, "overoptimization", report), report


Pair = tuple[tuple[ChatMessage, ...], str, str]


def held_out_pairs(store: AlignmentStore, dataset_id: str) -> list[Pair]:
    """``(prompt, chosen, rejected)`` texts of a preference dataset's validation split."""

    from epor.training.chat_data import load_chat_manifest

    manifest = load_chat_manifest(store.data, dataset_id)
    if manifest.kind != "preference":
        raise ValueError("overoptimization monitors read a preference dataset")
    split = manifest.splits.get("validation")
    records = [store.get(item) for item in (split.record_ids if split else ())]
    return [(tuple(item.messages), item.chosen, item.rejected) for item in records]


def audited_pairs(store: AlignmentStore, ledger: Any) -> list[Pair]:
    """``(prompt, endorsed, other)`` for every human-audited pairwise judge decision.

    The human verdict decides which reply is endorsed: ``accept`` endorses the
    judge's preferred reply, ``reject`` the other one.
    """

    pairs: list[Pair] = []
    for item in ledger.items():
        if item.verdict is None or item.subject not in {"preference", "revision"}:
            continue
        judgement = store.get(item.record)
        first, second = (store.get(value) for value in judgement.inputs)
        preferred, other = (
            (second, first) if judgement.preferred == judgement.inputs[1] else (first, second)
        )
        endorsed, rest = (preferred, other) if item.verdict == "accept" else (other, preferred)
        if endorsed.output != rest.output:
            pairs.append((tuple(first.messages), endorsed.output, rest.output))
    return pairs


def store_report(store: AlignmentStore, kind: str, report: dict[str, Any]) -> str:
    payload = canonical_json_bytes(report)
    report_id = sha256_bytes(payload)
    store.data.write_alignment_file(f"reports/{kind}/{report_id}.json", payload)
    return report_id


def load_report(store: AlignmentStore, kind: str, report_id: str) -> dict[str, Any]:
    import json

    if len(report_id) != 64 or any(c not in "0123456789abcdef" for c in report_id):
        raise ValueError(f"not a report ID: {report_id!r}")
    path = store.data.alignment_path(f"reports/{kind}/{report_id}.json")
    payload = path.read_bytes()
    if sha256_bytes(payload) != report_id:
        raise ValueError(f"alignment report {report_id} was altered")
    return json.loads(payload)


def report_ids(store: AlignmentStore, kind: str) -> list[str]:
    directory = store.data.alignment_path(f"reports/{kind}")
    return sorted(path.stem for path in directory.glob("*.json")) if directory.is_dir() else []


__all__ = [
    "MONITORS_PATH",
    "MonitorConfig",
    "Threshold",
    "audited_pairs",
    "classify",
    "held_out_pairs",
    "load_monitor_config",
    "load_report",
    "overoptimization_report",
    "report_ids",
    "store_report",
]
