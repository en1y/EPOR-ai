"""Read-only, torch-free summaries of training-kernel runs and evaluation reports.

Runs are found beneath two roots only: control-job outputs under the artifact
root and direct CLI runs under ``<project_root>/runs``.  A run is any
``manifest.json`` at most :data:`MAX_DEPTH` directories below its root, so
nested layouts are listed too: ``<job id>/<name>``, scaling sweep and mixture
attempts (``<output>/runs/<point_id>/attempt-<n>``), and any ``--output
runs/<a>/<b>``.  Evaluation reports are the ``*.json`` files of any
``evaluations`` or ``kernel-eval`` directory found by the same walk.  The walk
never follows a symbolic link, keeps every path inside its root, does not enter
``checkpoints`` or ``manifests`` directories (a run's checkpoints and immutable
manifest copies, never another run), and stops at :data:`MAX_DIRECTORIES`
directories and :data:`MAX_MANIFESTS` files of each kind per root; a bound that
cuts the walk short is reported in ``notices``.  Only manifests whose
``command`` is ``kernel_train`` (pretraining, SFT, distillation) or
``alignment_dpo`` (the v0.0.7 DPO baseline) are listed; ``objective`` says
which.

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
import os
from pathlib import Path
from typing import Any

from .security import PathOutsideRootError, contained_path

MAX_RUNS = 100
MAX_REPORTS = 200
MAX_JSON_BYTES = 8 * 1024 * 1024
# Discovery bounds, per root.  A job sweep attempt sits five levels down
# (``<job>/<name>/runs/<point>/attempt-<n>``) and its reports six.
MAX_DEPTH = 8
MAX_DIRECTORIES = 5_000
MAX_MANIFESTS = 1_000
_NOT_ENTERED = frozenset({"checkpoints", "manifests"})
_REPORT_DIRECTORIES = frozenset({"evaluations", "kernel-eval"})
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

    pointer_path = contained_path(run_dir, "checkpoints/latest.json")
    if not pointer_path.is_file():
        return None
    pointer = _read_json(pointer_path)
    name = str(pointer.get("name", ""))
    if not name.startswith("step-") or "/" in name or name in {".", ".."}:
        raise ValueError("latest-checkpoint pointer names an invalid directory")
    directory = contained_path(run_dir, f"checkpoints/{name}")
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
    if manifest.get("command") not in {"kernel_train", "alignment_dpo"}:
        return None
    run_dir = manifest_path.parent
    status = str(manifest.get("status", "unknown"))
    if status != "running":
        run_id = str(manifest.get("run_id"))
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run manifest has an invalid run_id")
        immutable = contained_path(run_dir, f"manifests/{run_id}.json")
        if not immutable.is_file() or _read_json(immutable) != manifest:
            raise ValueError("finished run manifest differs from its immutable copy")
    settings = manifest.get("training_settings") or manifest.get("preference_settings") or {}
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
        "evaluation": None
        if manifest.get("command") == "alignment_dpo"
        else manifest.get("evaluation"),
        "checkpoint": _verify_checkpoint(run_dir),
        "manifest_sha256": _sha256(manifest_path),
        **_alignment_fields(manifest),
    }


def _objective(manifest: dict[str, Any]) -> str:
    if manifest.get("command") == "alignment_dpo":
        return "dpo"
    if manifest.get("distillation"):
        return "logit-distillation"
    if manifest.get("chat_dataset_id"):
        return "sft"
    return "pretrain"


def _alignment_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    """What a v0.0.7 run adds: its objective, dataset, parent, and DPO signals."""

    metrics = manifest.get("metrics") or {}
    parent = manifest.get("init_checkpoint") or {}
    preference = manifest.get("preference_settings") or {}
    reference = preference.get("reference") or {}
    return {
        "objective": _objective(manifest),
        "chat_dataset_id": manifest.get("chat_dataset_id") or manifest.get("dataset_id"),
        "parent_checkpoint_sha256": parent.get("checkpoint_sha256")
        or (preference.get("policy_init") or {}).get("checkpoint_sha256"),
        "reference_checkpoint_sha256": reference.get("checkpoint_sha256"),
        "distill_teachers": len((manifest.get("distillation") or {}).get("teachers") or []),
        "distill_kl": metrics.get("distill_kl"),
        "reward_margin": metrics.get("reward_margin"),
        "preference_accuracy": metrics.get("preference_accuracy"),
        "policy_logratio": metrics.get("policy_logratio"),
        "alerts": list(manifest.get("alerts") or []),
        "preference_evaluation": (
            manifest.get("evaluation") if manifest.get("command") == "alignment_dpo" else None
        ),
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


def _discover(root: Path) -> tuple[list[Path], list[Path], list[str]]:
    """Run manifests and evaluation reports beneath ``root``, within the bounds.

    Directories are walked in sorted order without following symbolic links;
    a symlinked file is skipped and every returned path is resolved and inside
    the root.  Returns ``(manifests, reports, notices)``.
    """

    if not root.is_dir():
        return [], [], []
    base = root.resolve()
    found: dict[str, list[Path]] = {"run manifests": [], "evaluation reports": []}
    notices: set[str] = set()
    for walked, (current, directories, files) in enumerate(os.walk(base), start=1):
        if walked > MAX_DIRECTORIES:
            notices.add(f"the search stopped after {MAX_DIRECTORIES} directories")
            break
        here = Path(current)
        directories[:] = sorted(name for name in directories if name not in _NOT_ENTERED)
        if directories and len(here.relative_to(base).parts) >= MAX_DEPTH:
            notices.add(f"directories more than {MAX_DEPTH} levels deep were not searched")
            directories.clear()
        candidates = [("run manifests", "manifest.json")] if "manifest.json" in files else []
        if here.name in _REPORT_DIRECTORIES:
            candidates += [
                ("evaluation reports", name)
                for name in sorted(files)
                if name.endswith(".json") and name != "manifest.json"
            ]
        for kind, name in candidates:
            path = here / name
            if path.is_symlink():
                continue
            if len(found[kind]) >= MAX_MANIFESTS:
                notices.add(f"only the first {MAX_MANIFESTS} {kind} were read")
                continue
            try:
                found[kind].append(contained_path(base, path, must_exist=True))
            except (OSError, PathOutsideRootError):
                continue
    return found["run manifests"], found["evaluation reports"], sorted(notices)


def training_summary(artifact_root: Path, project_root: Path) -> dict[str, Any]:
    """Kernel runs and evaluation reports, newest first, with integrity errors."""

    runs: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    notices: list[str] = []
    seen: set[Path] = set()
    for source, root, prefix in (
        ("job", artifact_root, ""),
        ("cli", project_root / "runs", "runs/"),
    ):
        manifests, report_paths, cut = _discover(root)
        notices += [
            f"{source} discovery: {reason}; the listing may be incomplete" for reason in cut
        ]
        base = root.resolve()
        for path in manifests:
            if path in seen:
                continue
            seen.add(path)
            location = prefix + path.parent.relative_to(base).as_posix()
            try:
                record = _run_record(path, source=source, location=location)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                errors.append(f"{source} run {location}: {exc}")
                continue
            if record is not None:
                runs.append(record)
        for path in report_paths:
            if path in seen:
                continue
            seen.add(path)
            location = prefix + path.relative_to(base).as_posix()
            try:
                report = _report_record(path, location=location)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                errors.append(f"evaluation report {location}: {exc}")
                continue
            if report is not None:
                reports.append(report)
    runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    reports.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    if len(runs) > MAX_RUNS:
        notices.append(f"showing the newest {MAX_RUNS} of {len(runs)} runs")
    if len(reports) > MAX_REPORTS:
        notices.append(f"showing the newest {MAX_REPORTS} of {len(reports)} evaluation reports")
    return {
        "schema_version": 1,
        "scope": (
            "Proxy-scale local measurements on named tokenized builds; none is a trained or "
            "validated model result."
        ),
        "runs": runs[:MAX_RUNS],
        "evaluations": reports[:MAX_REPORTS],
        "integrity_errors": sorted(errors),
        "notices": notices,
    }
