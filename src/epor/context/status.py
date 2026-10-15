"""Configured, trained, and validated context — three numbers, never conflated.

``configured``  the checkpoint's model config: the longest input the code accepts.
``trained``     the longest ``seq_len`` its lineage trained at (``trainer.json``).
``validated``   the longest length a :func:`certify` certificate exists for.

Only :func:`certify` creates a certificate, and only when every condition
holds for that exact checkpoint (by the SHA-256 of its ``checkpoint.json``):

* the length does not exceed the trained context (validated <= trained <=
  configured, the invariant model configs already enforce);
* a G6 report of this checkpoint passes :func:`epor.context.g6.gate` at the
  length — retrieval at every depth, injection resistance, and the
  short-context regression against a pre-extension baseline report;
* a host profile report of this checkpoint shows the length completing with
  no out-of-memory, timeout, or refusal.

Accepting a long input, training at a length, or a passing report alone never
raises the validated context.  Certificates are immutable records under
``<data-root>/context-certificates/``; nothing edits a config or checkpoint.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes


class CertificationRefusedError(PermissionError):
    """The evidence does not support raising the validated context to this length."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def _checkpoint(checkpoint: str | Path) -> tuple[Path, dict[str, Any], str]:
    from epor.alignment.model import checkpoint_identity
    from epor.training.run_checkpoint import read_kernel_checkpoint

    trainer = read_kernel_checkpoint(checkpoint)
    directory = Path(trainer["checkpoint_dir"])
    return directory, trainer, checkpoint_identity(directory)


def certificates(store: DataStore, checkpoint_sha256: str | None = None) -> list[dict[str, Any]]:
    found = []
    for certificate_id in store.certificate_ids():
        payload = store.certificate_path(certificate_id).read_bytes()
        if sha256_bytes(payload) != certificate_id:
            raise ValueError(f"context certificate {certificate_id} was altered")
        record = json.loads(payload)
        if checkpoint_sha256 is None or record["checkpoint_sha256"] == checkpoint_sha256:
            found.append({"certificate_id": certificate_id, **record})
    return found


def context_status(checkpoint: str | Path, data_root: str | Path) -> dict[str, Any]:
    from epor.models.config import model_config_from_mapping
    from epor.training.kernel import trained_context

    _, trainer, identity = _checkpoint(checkpoint)
    config = model_config_from_mapping(trainer["model_config"])
    held = certificates(DataStore(Path(data_root)), identity)
    return {
        "checkpoint_sha256": identity,
        "model_id": config.slug,
        "configured": config.configured_max_context,
        "trained": trained_context(trainer),
        "validated": max((record["length"] for record in held), default=0),
        "certificates": [record["certificate_id"] for record in held],
        "rope_scaling": None if config.rope_scaling is None else config.rope_scaling.model_dump(),
        "attention_variant": config.attention_variant,
        "sliding_window": config.sliding_window,
        "local_layers_per_global": config.local_layers_per_global,
    }


def certify(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    length: int,
    g6_report_id: str,
    baseline_report_id: str,
    base_length: int,
    profile_report_id: str,
) -> dict[str, Any]:
    """Record a validated context of ``length`` for ``checkpoint``, or refuse with reasons."""

    from epor.evaluation.report import load_report
    from epor.scaling.report import load_scaling_reports
    from epor.training.kernel import trained_context

    from .g6 import G6_SUITE_PATH, SUITE_ID, gate, load_g6_suite

    store = DataStore(Path(data_root))
    _, trainer, identity = _checkpoint(checkpoint)
    reasons: list[str] = []
    trained = trained_context(trainer)
    if length > trained:
        reasons.append(f"{length} exceeds the trained context {trained}")
    report = load_report(store, g6_report_id)
    baseline = load_report(store, baseline_report_id)
    if report.suite_id != SUITE_ID or baseline.suite_id != SUITE_ID:
        reasons.append("the reports are not G6 context reports")
    if report.model.checkpoint_sha256 != identity:
        reasons.append("the G6 report evaluated a different checkpoint")
    if baseline.model.checkpoint_sha256 == identity:
        reasons.append("the baseline must be the pre-extension checkpoint, not this one")
    [profile] = load_scaling_reports(store, profile_report_id)[-1:]
    profile_status = None
    if profile.kind != "context-profile" or profile.inputs.get("checkpoint_sha256") != identity:
        reasons.append("the profile report does not measure this checkpoint")
    else:
        for record in profile.results.get("lengths", []):
            if record.get("length") == length:
                profile_status = record.get("status")
    decision = gate(
        report,
        length,
        thresholds=load_g6_suite().gate,
        baseline=baseline,
        base_length=base_length,
        profile_status=profile_status,
    )
    reasons += decision["reasons"]
    if reasons:
        raise CertificationRefusedError(reasons)
    record = {
        "schema_version": 1,
        "kind": "validated-context",
        "checkpoint_sha256": identity,
        "length": length,
        "g6_report_id": g6_report_id,
        "baseline_report_id": baseline_report_id,
        "base_length": base_length,
        "profile_report_id": profile_report_id,
        "gate": decision,
        "thresholds_sha256": sha256_bytes(G6_SUITE_PATH.read_bytes()),
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = canonical_json_bytes(record)
    certificate_id = sha256_bytes(payload)
    store.write_certificate(certificate_id, payload)
    return {"certificate_id": certificate_id, **record}


__all__ = ["CertificationRefusedError", "certificates", "certify", "context_status"]
