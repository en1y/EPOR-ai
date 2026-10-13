"""Human audits of AI judge decisions: queue, verdicts, calibration, and admission.

The audit queue is control-plane truth: one append-only, hash-chained log
(``<safety-root>/alignment-audit.jsonl``, the same primitive as the identity
and escalation log) holding two kinds of record.

``audit.item``
    A judge decision (or a G10 failure needing review) placed in the queue by
    stratified sampling.  An item starts *pending*.
``audit.verdict``
    A human's verdict — ``accept`` or ``reject`` — on one item, with the
    verified actor who gave it and an optional note.  A later verdict on the
    same item supersedes an earlier one; neither is ever deleted.

**Nothing is ever auto-marked audited.**  The only writer of a verdict is
:meth:`AuditLedger.record_verdict`, which requires an authenticated human
actor; no job handler, judge, sampler, or report calls it, and an item with no
verdict counts as unaudited everywhere.  Surfaces gate the call through the
covenant's ``align_audit`` action, a scope separate from training.

Calibration and admission
-------------------------
For one judging run, :func:`calibration` compares the judge with the humans on
the audited sample: an agreement matrix, the judge-human **disagreement
rate** overall and per stratum, and reliability bins of judge confidence
against human agreement.  :func:`admission_gate` admits a run's candidates
only when every sampled item has a verdict, at least ``min_per_stratum``
accepted *and* rejected decisions were audited, and the disagreement rate is
at most ``max_disagreement`` — a ``hypothesis`` threshold from
``configs/alignment/monitors.yaml``.  Within an admitted run a candidate still
needs the judge's acceptance, and a human ``reject`` on its own item excludes
it.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from epor.control.truth import ChainedLog

from .datasets import Admission
from .records import AlignmentStore, ExampleRecord, JudgementRecord, PreferenceRecord

AUDIT_LOG = "alignment-audit.jsonl"
Verdict = Literal["accept", "reject"]
MAX_NOTE = 2_000


class AuditError(ValueError):
    """An audit request is malformed, unauthenticated, or names an unknown item."""


@dataclass(frozen=True, slots=True)
class AuditItem:
    id: str
    subject: str
    judging_run: str
    record: str
    stratum: str
    judge_accepted: bool | None
    confidence: float | None
    created_at: str
    created_by: str
    detail: dict[str, Any]
    verdicts: tuple[dict[str, Any], ...]

    @property
    def verdict(self) -> str | None:
        return self.verdicts[-1]["verdict"] if self.verdicts else None

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "judging_run": self.judging_run,
            "record": self.record,
            "stratum": self.stratum,
            "judge_accepted": self.judge_accepted,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "detail": self.detail,
            "state": "pending" if self.verdict is None else "audited",
            "verdict": self.verdict,
            "verdicts": list(self.verdicts),
        }


class AuditLedger:
    def __init__(self, safety_root: str | Path) -> None:
        self.log = ChainedLog(Path(safety_root) / AUDIT_LOG)

    def items(self) -> list[AuditItem]:
        records = self.log.load()
        verdicts: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if record.get("kind") == "audit.verdict":
                verdicts.setdefault(str(record["item_id"]), []).append(
                    {
                        key: record[key]
                        for key in ("verdict", "actor_id", "actor_role", "note", "created_at")
                    }
                )
        return [
            AuditItem(
                id=str(record["item_id"]),
                subject=str(record["subject"]),
                judging_run=str(record["judging_run"]),
                record=str(record["record"]),
                stratum=str(record["stratum"]),
                judge_accepted=record.get("judge_accepted"),
                confidence=record.get("confidence"),
                created_at=str(record["created_at"]),
                created_by=str(record["created_by"]),
                detail=dict(record.get("detail") or {}),
                verdicts=tuple(verdicts.get(str(record["item_id"]), ())),
            )
            for record in records
            if record.get("kind") == "audit.item"
        ]

    def item(self, item_id: str) -> AuditItem:
        for item in self.items():
            if item.id == item_id:
                return item
        raise AuditError(f"audit item not found: {item_id}")

    def enqueue(
        self,
        *,
        subject: str,
        judging_run: str,
        record: str,
        stratum: str,
        actor_id: str,
        judge_accepted: bool | None = None,
        confidence: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> str | None:
        """Queue one item unless the same record is already queued for the run."""

        item_id = str(uuid.uuid4())

        def build(committed: list[dict[str, Any]]) -> dict[str, Any] | None:
            for existing in committed:
                if (
                    existing.get("kind") == "audit.item"
                    and existing.get("judging_run") == judging_run
                    and existing.get("record") == record
                ):
                    return None
            return {
                "kind": "audit.item",
                "item_id": item_id,
                "subject": subject,
                "judging_run": judging_run,
                "record": record,
                "stratum": stratum,
                "judge_accepted": judge_accepted,
                "confidence": confidence,
                "detail": detail or {},
                "created_by": actor_id,
                "created_at": datetime.now(UTC).isoformat(),
            }

        return item_id if self.log.transact(build) is not None else None

    def record_verdict(
        self, item_id: str, verdict: Verdict, *, actor: Any, note: str = ""
    ) -> dict[str, Any]:
        """Append a human verdict.  The only way an item ever becomes audited."""

        if verdict not in ("accept", "reject"):
            raise AuditError("a verdict is 'accept' or 'reject'")
        if not getattr(actor, "is_authenticated", False):
            raise AuditError("a verdict needs a verified human identity")
        if len(note) > MAX_NOTE:
            raise AuditError(f"a verdict note is at most {MAX_NOTE} characters")
        self.item(item_id)
        return self.log.append(
            {
                "kind": "audit.verdict",
                "item_id": item_id,
                "verdict": verdict,
                "actor_id": str(actor.id),
                "actor_role": str(getattr(getattr(actor, "role", None), "value", "unknown")),
                "note": note,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )


def sample_run(
    ledger: AuditLedger,
    store: AlignmentStore,
    judging_run: str,
    *,
    per_stratum: int,
    seed: int,
    actor_id: str,
) -> list[str]:
    """Queue a seeded sample of up to ``per_stratum`` accepted and rejected decisions."""

    if not 1 <= per_stratum <= 500:
        raise AuditError("per_stratum must be between 1 and 500")
    from .judge import judging_runs

    decisions = judging_runs(store).get(judging_run)
    if not decisions:
        raise AuditError(f"no judgements belong to judging run {judging_run}")
    queued: list[str] = []
    for stratum, accepted in (("accepted", True), ("rejected", False)):
        members = sorted(item for item in decisions if item[1].accepted is accepted)
        chosen = random.Random(f"{seed}:{judging_run}:{stratum}").sample(
            members, min(per_stratum, len(members))
        )
        for identifier, record in chosen:
            created = ledger.enqueue(
                subject=record.subject,
                judging_run=judging_run,
                record=identifier,
                stratum=stratum,
                actor_id=actor_id,
                judge_accepted=record.accepted,
                confidence=record.confidence,
                detail=_detail(store, record),
            )
            if created is not None:
                queued.append(created)
    return queued


def _detail(store: AlignmentStore, record: JudgementRecord) -> dict[str, Any]:
    """What a human needs to decide: the prompt, the replies, and the judge's pick."""

    inputs = [store.get(item) for item in record.inputs]
    if record.subject == "eval-item":
        item = inputs[0]
        return {
            "category": item.category,
            "obligation": item.obligation,
            "context": item.context,
            "safe": item.safe,
            "unsafe": item.unsafe,
            "question": "Is this a well-formed, harmless G10 item whose safe reply is safe?",
        }
    first, second = inputs
    return {
        "prompt": [message.model_dump() for message in first.messages],
        "replies": [first.output, second.output],
        "judge_preferred": 1 if record.preferred == record.inputs[1] else 0,
        "margin": record.margin,
        "question": (
            "Is the revision (second reply) an improvement that should be trained on?"
            if record.subject == "revision"
            else "Is the judge's preferred reply the better one?"
        ),
    }


def _agreement(item: AuditItem) -> bool:
    return (item.verdict == "accept") == bool(item.judge_accepted)


def calibration(
    ledger: AuditLedger, judging_run: str, *, bins: int = 5, store: AlignmentStore | None = None
) -> dict[str, Any]:
    """Judge-human comparison on a run's audited sample.

    An item *agrees* when the human accepts a decision the judge accepted, or
    rejects one it rejected.
    """

    items = [item for item in ledger.items() if item.judging_run == judging_run]
    audited = [item for item in items if item.verdict is not None]
    strata: dict[str, dict[str, Any]] = {}
    for stratum in ("accepted", "rejected"):
        members = [item for item in items if item.stratum == stratum]
        done = [item for item in members if item.verdict is not None]
        strata[stratum] = {
            "sampled": len(members),
            "audited": len(done),
            "disagreement_rate": (
                sum(not _agreement(item) for item in done) / len(done) if done else None
            ),
        }
    matrix = {
        f"judge_{'accept' if judge else 'reject'}__human_{human}": sum(
            1 for item in audited if bool(item.judge_accepted) is judge and item.verdict == human
        )
        for judge in (True, False)
        for human in ("accept", "reject")
    }
    reliability = []
    for index in range(bins):
        low, high = 0.5 + 0.5 * index / bins, 0.5 + 0.5 * (index + 1) / bins
        members = [
            item
            for item in audited
            if item.confidence is not None
            and (low <= item.confidence < high or (index == bins - 1 and item.confidence == 1.0))
        ]
        reliability.append(
            {
                "confidence_low": low,
                "confidence_high": high,
                "items": len(members),
                "human_agreement": (
                    sum(_agreement(item) for item in members) / len(members) if members else None
                ),
            }
        )
    total_decisions = None
    if store is not None:
        from .judge import judging_runs

        total_decisions = len(judging_runs(store).get(judging_run, []))
    return {
        "judging_run": judging_run,
        "decisions": total_decisions,
        "sampled": len(items),
        "audited": len(audited),
        "pending": len(items) - len(audited),
        "strata": strata,
        "agreement_matrix": matrix,
        "disagreement_rate": (
            sum(not _agreement(item) for item in audited) / len(audited) if audited else None
        ),
        "reliability": reliability,
        "evidence": "reported from human verdicts on a seeded stratified sample",
    }


def admission_gate(
    ledger: AuditLedger,
    store: AlignmentStore,
    *,
    max_disagreement: float,
    min_per_stratum: int = 1,
) -> Any:
    """A :data:`~epor.alignment.datasets.CandidateGate` over audited judging runs."""

    judgements = dict(store.records("judgement"))
    items = ledger.items()
    by_record = {item.record: item for item in items}
    run_status: dict[str, Admission] = {}

    def run_admission(run: str) -> Admission:
        if run in run_status:
            return run_status[run]
        report = calibration(ledger, run)
        strata = report["strata"]
        reasons: list[str] = []
        if report["pending"]:
            reasons.append(f"{report['pending']} sampled decision(s) still await a human verdict")
        for stratum in ("accepted", "rejected"):
            if strata[stratum]["audited"] < min_per_stratum:
                reasons.append(f"fewer than {min_per_stratum} {stratum} decision(s) audited")
        rate = report["disagreement_rate"]
        if rate is not None and rate > max_disagreement:
            reasons.append(
                f"judge-human disagreement {rate:.3f} exceeds {max_disagreement} (hypothesis)"
            )
        evidence = {
            "judging_run": run,
            "audited": report["audited"],
            "disagreement_rate": rate,
            "max_disagreement": max_disagreement,
        }
        decision = Admission(not reasons, "; ".join(reasons) or "audited run admitted", evidence)
        run_status[run] = decision
        return decision

    def judged(record_id: str, record: Any) -> tuple[str, JudgementRecord] | None:
        if isinstance(record, PreferenceRecord) and record.judgement is not None:
            judgement = judgements.get(record.judgement)
            return (record.judgement, judgement) if judgement is not None else None
        if isinstance(record, ExampleRecord) and record.purpose == "revision":
            for identifier, judgement in judgements.items():
                if (
                    judgement.subject == "revision"
                    and judgement.inputs[-1] == record_id
                    and judgement.preferred == record_id
                ):
                    return identifier, judgement
        return None

    def gate(record_id: str, record: Any) -> Admission:
        found = judged(record_id, record)
        if found is None:
            return Admission(False, "no judge decision covers this candidate")
        judgement_id, judgement = found
        run = run_admission(judgement.judging_run)
        if not run.admitted:
            return run
        if not judgement.accepted:
            return Admission(False, "the judge did not accept this candidate", run.evidence)
        own = by_record.get(judgement_id)
        if own is not None and own.verdict == "reject":
            return Admission(False, "a human rejected this candidate's decision", run.evidence)
        return Admission(True, "judge-accepted in an audited run", run.evidence)

    return gate


def pending(ledger: AuditLedger, subjects: Sequence[str] | None = None) -> list[AuditItem]:
    return [
        item
        for item in ledger.items()
        if item.verdict is None and (subjects is None or item.subject in subjects)
    ]


__all__ = [
    "AUDIT_LOG",
    "AuditError",
    "AuditItem",
    "AuditLedger",
    "Verdict",
    "admission_gate",
    "calibration",
    "pending",
    "sample_run",
]
