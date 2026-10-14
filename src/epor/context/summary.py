"""Read-only, verified views of context-laboratory state for the CLI and the console.

Context runs are listed here, separately from the Training page's listing:
any kernel run whose manifest carries a ``context`` section (every run since
v0.0.8), found under the artifact root's job outputs or anywhere up to three
levels below the project's ``runs/``.  Each shows configured, trained, and
validated context side by side; trained and validated come only from the
run's latest checkpoint, whose ``checkpoint.json`` hash is re-derived, so a
run with no checkpoint yet shows neither.

G6 reports are evaluation reports of suite ``epor-g6-context``; profiles,
feasibility searches, sweeps, and projections are scaling reports of the four
``context-*`` kinds.  Everything is re-verified against its content address
when read, and anything that fails is an integrity error, not a row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from epor.data.store import DataStore

from .tasks import DEPTHS

MAX_RUNS = 100


def _runs(artifact_root: Path, project_root: Path) -> list[tuple[Path, str]]:
    found = [(path, "job") for path in artifact_root.glob("*/*/manifest.json")]
    runs = project_root / "runs"
    for depth in ("*", "*/*", "*/*/*"):
        found += [(path, "cli") for path in runs.glob(f"{depth}/manifest.json")]
    return found


def _latest_checkpoint(run_dir: Path) -> tuple[str, int] | None:
    """The verified latest checkpoint's hash and the context its lineage trained at.

    Trained context comes from the checkpoint, never the manifest: a run that
    stopped before its first checkpoint has trained nothing at its length.
    """

    from epor.training.kernel import trained_context

    pointer = run_dir / "checkpoints" / "latest.json"
    if not pointer.is_file():
        return None
    latest = json.loads(pointer.read_text(encoding="utf-8"))
    name = str(latest.get("name", ""))
    if not name.startswith("step-") or "/" in name:
        raise ValueError("latest-checkpoint pointer names an invalid directory")
    digest = hashlib.sha256((run_dir / "checkpoints" / name / "checkpoint.json").read_bytes())
    if digest.hexdigest() != latest.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint {name} does not match its pointer")
    trainer = json.loads((run_dir / "checkpoints" / name / "trainer.json").read_text("utf-8"))
    return digest.hexdigest(), trained_context(trainer)


def g6_view(report: Any) -> dict[str, Any]:
    heatmap: dict[str, dict[str, dict[str, float | None]]] = {}
    lengths: set[int] = set()
    for task in report.tasks:
        if task.task_type != "context-retrieval":
            continue
        cells: dict[str, dict[str, float | None]] = {}
        for metric in task.metrics:
            parts = metric.name.split(".")
            if len(parts) == 3 and parts[1].startswith("length-") and parts[2] in DEPTHS:
                length = parts[1].removeprefix("length-")
                lengths.add(int(length))
                cells.setdefault(length, {})[parts[2]] = metric.value
        heatmap[task.task_id] = cells
    perplexity: dict[str, list[float | None]] = {}
    for task in report.tasks:
        if task.task_type == "position-perplexity":
            for metric in task.metrics:
                parts = metric.name.split(".")
                if len(parts) == 3 and parts[2].startswith("bucket-"):
                    perplexity.setdefault(parts[1].removeprefix("length-"), []).append(metric.value)
    injection = next((task for task in report.tasks if task.task_id == "prompt-injection"), None)
    followed = None
    if injection is not None:
        followed = next(
            (m.value for m in injection.metrics if m.name == "injection_followed"), None
        )
    return {
        "report_id": report.report_id,
        "model_id": report.model.model_id,
        "checkpoint_sha256": report.model.checkpoint_sha256,
        "checkpoint_step": report.model.checkpoint_step,
        "scope": report.scope,
        "lengths": sorted(lengths),
        "retrieval": heatmap,
        "position_loss": perplexity,
        "injection_followed": followed,
        "claim_boundary": report.claim_boundary,
    }


def context_summary(data_root: Path, artifact_root: Path, project_root: Path) -> dict[str, Any]:
    from epor.evaluation.summary import load_verified_report
    from epor.scaling.summary import load_verified_scaling_reports

    from .status import certificates

    store = DataStore(data_root)
    errors: list[str] = []
    held: list[dict[str, Any]] = []
    try:
        held = certificates(store)
    except (OSError, ValueError) as exc:
        errors.append(f"certificates: {exc}")
    validated: dict[str, int] = {}
    for record in held:
        sha = record["checkpoint_sha256"]
        validated[sha] = max(validated.get(sha, 0), int(record["length"]))
    runs: list[dict[str, Any]] = []
    for path, source in _runs(artifact_root, project_root):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            context = manifest.get("context")
            if manifest.get("command") != "kernel_train" or not isinstance(context, dict):
                continue
            latest = _latest_checkpoint(path.parent)
        except (OSError, ValueError) as exc:
            errors.append(f"{source} run {path.parent.name}: {exc}")
            continue
        root = artifact_root if source == "job" else project_root
        runs.append(
            {
                "location": path.parent.relative_to(root).as_posix(),
                "source": source,
                "run_id": manifest.get("run_id"),
                "status": manifest.get("status"),
                "model_id": manifest.get("model_id"),
                "started_at": manifest.get("started_at"),
                "configured": context.get("configured"),
                "trained": None if latest is None else latest[1],
                "validated": 0 if latest is None else validated.get(latest[0], 0),
                "rope_scaling": context.get("rope_scaling"),
                "sliding_window": context.get("sliding_window"),
                "local_layers_per_global": context.get("local_layers_per_global"),
                "attention_variant": context.get("attention_variant"),
                "sparse_attention": context.get("sparse_attention"),
                "extension_of": context.get("extension_of"),
                "checkpoint_sha256": None if latest is None else latest[0],
            }
        )
    runs.sort(key=lambda row: str(row.get("started_at") or ""), reverse=True)
    g6: list[dict[str, Any]] = []
    for report_id in store.evaluation_report_ids():
        try:
            report = load_verified_report(data_root, report_id)
        except (OSError, ValueError) as exc:
            errors.append(f"evaluation report {report_id[:12]}: {exc}")
            continue
        if report.suite_id == "epor-g6-context":
            g6.append(g6_view(report))
    reports: dict[str, list[dict[str, Any]]] = {
        "context-profile": [],
        "context-feasibility": [],
        "context-sweep": [],
        "context-projection": [],
    }
    for report_id in store.scaling_report_ids():
        try:
            measured = load_verified_scaling_reports(data_root, report_id)
        except (OSError, ValueError) as exc:
            errors.append(f"scaling report {report_id[:12]}: {exc}")
            continue
        for report in measured:
            if report.kind in reports:
                reports[report.kind].append(
                    {
                        "report_id": report.report_id,
                        "content_sha256": report.content_sha256,
                        "scope": report.scope,
                        "inputs": report.inputs,
                        "results": report.results,
                        "measured": report.measured,
                        "warnings": report.warnings,
                    }
                )
    return {
        "schema_version": 1,
        "scope": (
            "Proxy-scale and host measurements on the CPU reference machine and projections "
            "from operator inputs. Configured, trained, and validated context are different "
            "numbers; only a G6 certificate raises validated context."
        ),
        "runs": runs[:MAX_RUNS],
        "g6": g6,
        "certificates": held,
        **{kind.removeprefix("context-"): items for kind, items in reports.items()},
        "integrity_errors": sorted(errors),
    }


__all__ = ["context_summary", "g6_view"]
