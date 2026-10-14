"""Read-only, verified views of alignment state for the CLI and the console.

Every record is read through :class:`~epor.alignment.records.AlignmentStore`
(which re-checks its content address), every dataset manifest through its ID,
and every report through :func:`~epor.alignment.monitors.load_report`; a
failure is listed as an integrity error rather than shown as healthy.  The
audit queue comes from the hash-chained audit log.  Keys avoid the control
plane's credential pattern, and model text appears only inside audit items,
where a human needs it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .audit import AuditLedger, calibration, default_thresholds, run_admission
from .records import RECORD_KINDS, AlignmentStore

MAX_LISTED = 200


def _reports(store: AlignmentStore, kind: str, errors: list[str]) -> list[dict[str, Any]]:
    from .monitors import load_report, report_ids

    loaded: list[dict[str, Any]] = []
    for report_id in report_ids(store, kind):
        try:
            loaded.append({"report_id": report_id, **load_report(store, kind, report_id)})
        except (OSError, ValueError) as exc:
            errors.append(f"{kind} report {report_id[:12]}: {exc}")
    loaded.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return loaded[:MAX_LISTED]


def _overoptimization_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "status": report["status"],
        "created_at": report["created_at"],
        "policy": {"model": report["policy"]["model"], "version": report["policy"]["version"]},
        "reference": {
            "model": report["reference"]["model"],
            "version": report["reference"]["version"],
        },
        "monitors": report["monitors"],
        "evidence": report["thresholds"]["evidence"],
        "claim": report["claim"],
    }


def _g10_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "gate": report["gate"],
        "created_at": report["created_at"],
        "before": {"model": report["before"]["model"], "version": report["before"]["version"]},
        "after": {"model": report["after"]["model"], "version": report["after"]["version"]},
        "categories": report["categories"],
        "overall": report["overall"],
        "regressed_categories": report["regressed_categories"],
        "failing_items": report["failing_items"],
        "claim": report["claim"],
    }


def _datasets(store: AlignmentStore, errors: list[str]) -> list[dict[str, Any]]:
    from epor.training.chat_data import ChatDatasetError, load_chat_manifest

    from .datasets import describe_dataset

    directory = store.data.alignment_path("datasets")
    views: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*/manifest.json")) if directory.is_dir() else []:
        dataset_id = path.parent.name
        try:
            views.append(describe_dataset(dataset_id, load_chat_manifest(store.data, dataset_id)))
        except (OSError, ValueError, ChatDatasetError) as exc:
            errors.append(f"dataset {dataset_id[:12]}: {exc}")
    return views[:MAX_LISTED]


def judging_run_views(store: AlignmentStore, ledger: AuditLedger) -> list[dict[str, Any]]:
    from .judge import judging_runs

    thresholds = default_thresholds()
    views: list[dict[str, Any]] = []
    for run, decisions in sorted(judging_runs(store).items()):
        report = calibration(ledger, run, store=store)
        accepted = [record for _, record in decisions if record.accepted]
        admission = run_admission(ledger, run, **thresholds)
        views.append(
            {
                "judging_run": run,
                "subject": decisions[0][1].subject,
                "judge": decisions[0][1].generator.model,
                "decisions": len(decisions),
                "accepted": len(accepted),
                "rejected": len(decisions) - len(accepted),
                "sampled": report["sampled"],
                "audited": report["audited"],
                "pending": report["pending"],
                "disagreement_rate": report["disagreement_rate"],
                "strata": report["strata"],
                "agreement_matrix": report["agreement_matrix"],
                "reliability": report["reliability"],
                "admitted": admission.admitted,
                "admission": admission.reason,
                "max_disagreement": thresholds["max_disagreement"],
            }
        )
    return views


def alignment_summary(data_root: str | Path, ledger: AuditLedger) -> dict[str, Any]:
    store = AlignmentStore(data_root)
    errors: list[str] = []
    counts: dict[str, int] = {}
    examples: Counter[str] = Counter()
    candidates = 0
    for kind in RECORD_KINDS:
        identifiers = store.ids(kind)
        counts[kind] = len(identifiers)
        if kind == "example":
            for identifier in identifiers:
                try:
                    record = store.get(identifier)
                except (OSError, ValueError) as exc:
                    errors.append(f"record {identifier[:12]}: {exc}")
                    continue
                examples[f"{record.generator.kind}/{record.purpose}"] += 1
                candidates += int(record.candidate)
    items = ledger.items()
    return {
        "schema_version": 1,
        "scope": (
            "Proxy-scale alignment pipeline state on a local machine; no record, run, or report "
            "here is evidence of alignment or safety."
        ),
        "records": counts,
        "examples_by_origin": dict(sorted(examples.items())),
        "candidates": candidates,
        "datasets": _datasets(store, errors),
        "judging_runs": judging_run_views(store, ledger),
        "audit_queue": {
            "pending": sum(1 for item in items if item.verdict is None),
            "audited": sum(1 for item in items if item.verdict is not None),
        },
        "overoptimization": [
            _overoptimization_view(item) for item in _reports(store, "overoptimization", errors)
        ],
        "g10": [_g10_view(item) for item in _reports(store, "g10", errors)],
        "integrity_errors": sorted(errors),
    }


__all__ = ["alignment_summary", "judging_run_views"]
