"""Read-only, torch-free summaries of training-kernel runs and evaluation reports.

Runs are found in two fixed places only: control-job outputs
(``<artifact_root>/<job id>/<name>/manifest.json``) and direct CLI runs under
``<project_root>/runs/<name>/manifest.json``.  Only manifests whose
``command`` is ``kernel_train`` are listed.

Verification is deliberately cheap enough to run on every console poll and
strict about what it does check: the latest-checkpoint pointer must name a
complete checkpoint whose ``checkpoint.json`` hashes to the pointer's digest,
``trainer.json`` must hash to its recorded digest, every recorded file must
exist with its recorded size, and a finished run's live manifest must equal
its immutable copy.  Full content hashes of the tensor files are re-verified
whenever a checkpoint is loaded (resume or evaluation), not here.  Anything
that fails is reported as an integrity error rather than shown as healthy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MAX_RUNS = 100
MAX_REPORTS = 200
MAX_JSON_BYTES = 8 * 1024 * 1024
_CHECKPOINT_SCHEMA = "epor-kernel-checkpoint-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("file exceeds the summary size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _verify_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    """Return the latest checkpoint's step and digest, or ``None`` before the first one."""

    root = run_dir / "checkpoints"
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = _read_json(pointer_path)
    name = str(pointer.get("name", ""))
    if not name.startswith("step-") or "/" in name or name in {".", ".."}:
        raise ValueError("latest-checkpoint pointer names an invalid directory")
    directory = root / name
    manifest_path = directory / "checkpoint.json"
    if not manifest_path.is_file():
        raise ValueError(f"latest checkpoint {name} is incomplete")
    digest = _sha256(manifest_path)
    if digest != pointer.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint {name} does not match its latest-checkpoint digest")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint {name} has an unsupported schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"checkpoint {name} records no files")
    trainer_digest = None
    for record in files:
        relative = str(record.get("path", ""))
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()) or not candidate.is_file():
            raise ValueError(f"checkpoint {name} is missing {relative}")
        if candidate.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"checkpoint {name} file {relative} changed size")
        if relative == "trainer.json":
            trainer_digest = record.get("sha256")
            if _sha256(candidate) != trainer_digest:
                raise ValueError(f"checkpoint {name} trainer state does not match its digest")
    if trainer_digest is None:
        raise ValueError(f"checkpoint {name} records no trainer state")
    step = manifest.get("step")
    if step != pointer.get("step"):
        raise ValueError(f"checkpoint {name} step disagrees with its pointer")
    return {
        "name": name,
        "step": step,
        "world_size": manifest.get("world_size"),
        "checkpoint_sha256": digest,
    }


def _run_record(manifest_path: Path, *, source: str, location: str) -> dict[str, Any] | None:
    manifest = _read_json(manifest_path)
    if manifest.get("command") != "kernel_train":
        return None
    run_dir = manifest_path.parent
    status = str(manifest.get("status", "unknown"))
    if status != "running":
        immutable = run_dir / "manifests" / f"{manifest.get('run_id')}.json"
        if not immutable.is_file() or _read_json(immutable) != manifest:
            raise ValueError("finished run manifest differs from its immutable copy")
    settings = manifest.get("training_settings") or {}
    metrics = manifest.get("metrics") or {}
    throughput = manifest.get("throughput") or {}
    features = manifest.get("model_features") or {}
    return {
        "run_id": manifest.get("run_id"),
        "lineage_id": manifest.get("lineage_id"),
        "parent_run_id": manifest.get("parent_run_id"),
        "source": source,
        "location": location,
        "status": status,
        "termination_reason": manifest.get("termination_reason"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "model_id": manifest.get("model_id"),
        "config_sha256": manifest.get("config_sha256"),
        "qk_norm": bool(features.get("qk_norm", False)),
        "mtp_depth": int(features.get("mtp_depth", 0)),
        "precision": manifest.get("precision"),
        "strategy": (manifest.get("distributed") or {}).get("strategy"),
        "world_size": (manifest.get("distributed") or {}).get("world_size"),
        "tokenized_id": manifest.get("tokenized_id"),
        "seq_len": settings.get("seq_len"),
        "tokens_per_batch": settings.get("tokens_per_batch"),
        "start_step": manifest.get("start_step"),
        "target_steps": manifest.get("target_steps"),
        "step": metrics.get("step"),
        "training_loss": metrics.get("training_loss"),
        "main_loss": metrics.get("main_loss"),
        "mtp_losses": list(metrics.get("mtp_losses") or []),
        "learning_rate": metrics.get("learning_rate"),
        "tokens_seen": (manifest.get("counters") or {}).get("tokens_seen"),
        "tokens_per_second": throughput.get("tokens_per_second"),
        "evaluation": manifest.get("evaluation"),
        "checkpoint": _verify_checkpoint(run_dir),
        "manifest_sha256": _sha256(manifest_path),
    }


def _report_record(path: Path, *, location: str) -> dict[str, Any] | None:
    report = _read_json(path)
    if report.get("command") != "kernel_eval":
        return None
    return {
        "location": location,
        "report_sha256": _sha256(path),
        "checkpoint_step": report.get("checkpoint_step"),
        "model_id": report.get("model_id"),
        "tokenized_id": report.get("tokenized_id"),
        "split": report.get("split"),
        "main_loss": report.get("main_loss"),
        "perplexity": report.get("perplexity"),
        "mtp_losses": list(report.get("mtp_losses") or []),
        "target_tokens": report.get("target_tokens"),
        "steps": report.get("steps"),
        "created_at": report.get("created_at"),
    }


def training_summary(artifact_root: Path, project_root: Path) -> dict[str, Any]:
    """Kernel runs and evaluation reports, newest first, with integrity errors."""

    runs: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    cli_root = project_root / "runs"
    candidates = [
        (path, "job", path.relative_to(artifact_root).parent.as_posix())
        for path in artifact_root.glob("*/*/manifest.json")
    ] + [
        (path, "cli", path.relative_to(project_root).parent.as_posix())
        for path in cli_root.glob("*/manifest.json")
    ]
    for path, source, location in candidates:
        try:
            record = _run_record(path, source=source, location=location)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{source} run {location}: {exc}")
            continue
        if record is not None:
            runs.append(record)
    report_paths = [
        (path, path.relative_to(artifact_root).as_posix())
        for path in artifact_root.glob("*/kernel-eval/*.json")
    ] + [
        (path, path.relative_to(project_root).as_posix())
        for path in cli_root.glob("*/evaluations/*.json")
    ]
    for path, location in report_paths:
        try:
            report = _report_record(path, location=location)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"evaluation report {location}: {exc}")
            continue
        if report is not None:
            reports.append(report)
    runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    reports.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {
        "schema_version": 1,
        "scope": (
            "Proxy-scale local measurements on named tokenized builds; none is a trained or "
            "validated model result."
        ),
        "runs": runs[:MAX_RUNS],
        "evaluations": reports[:MAX_REPORTS],
        "integrity_errors": sorted(errors),
    }
