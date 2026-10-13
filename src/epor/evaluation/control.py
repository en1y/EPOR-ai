"""Adapter between allowlisted evaluation control jobs and the evaluation harness.

Two job types run here: ``eval_suite`` (score a kernel checkpoint on a suite)
and ``eval_contamination`` (scan a suite against a training build without a
model).  Code tasks are always skipped in jobs: running model-generated code
is the separate ``eval_code_execution`` action, which only the owner's CLI
can request, so this module passes ``allow_code_execution=False`` explicitly
and accepts no option that could change it.

Job results carry the report ID and a small fixed summary
(:func:`epor.evaluation.summary.report_summary`); the full report stays in
``<data-root>/evaluation-reports/`` and is served, verified, by the read-only
console endpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from epor.control.security import contained_path
from epor.data.store import DataStore

from .suites import (
    EvaluationCancelledError,
    SuiteDefinition,
    TrainingData,
    fixture_suite,
    load_suite,
    run_contamination,
    run_suite,
    safety_suite,
)
from .summary import report_summary

_BUILTIN_SUITES: dict[str, Callable[[], SuiteDefinition]] = {
    "fixtures": fixture_suite,
    "safety": safety_suite,
}
CODE_EXECUTION_NOTE = (
    "code tasks are skipped in control jobs: running model-generated code is a separate "
    "action available only from the owner's CLI"
)


def _suite(value: str, project_root: Path) -> SuiteDefinition:
    if value in _BUILTIN_SUITES:
        return _BUILTIN_SUITES[value]()
    path = contained_path(project_root, value, must_exist=True)
    if not path.is_file():
        raise FileNotFoundError(f"suite file not found: {value}")
    return load_suite(path)


def _sources(spec: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for source in spec.get("sources", []):
        path = contained_path(project_root, str(source["path"]), must_exist=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError("benchmark sources must be regular non-symlink files")
        paths[str(source["source_id"])] = path
    return paths


def _task_progress(context: Any, total: int) -> Callable[[str], None]:
    seen: list[str] = []

    def on_task(task_id: str) -> None:
        seen.append(task_id)
        # Leave cancellation to the runner, which raises between tasks.
        if context.cancelled:
            return
        context.progress(
            0.05 + 0.9 * (len(seen) - 1) / max(total, 1),
            message=f"Running task {task_id}",
            details={"task": task_id, "index": len(seen), "tasks": total},
        )

    return on_task


def _roots(context: Any) -> tuple[Path, Path, Path]:
    settings = context.service.settings
    return (
        Path(settings.project_root).resolve(),
        Path(settings.artifact_root).resolve(),
        Path(settings.data_root).resolve(),
    )


def _finish(context: Any, report: Any, store: DataStore) -> dict[str, Any]:
    from .report import store_report

    store_report(store, report)
    if any(item.reason == "code execution not enabled" for item in report.skipped_tasks):
        context.log(CODE_EXECUTION_NOTE, level="warning")
    context.progress(1.0, message="Evaluation report stored")
    return {
        **report_summary(report),
        "skipped_tasks": [item.model_dump(mode="json") for item in report.skipped_tasks],
    }


def _run_suite(spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    from epor.training.run_checkpoint import resolve_checkpoint

    from .decoder import load_kernel_scoring_model

    project_root, artifact_root, data_root = _roots(context)
    directory = contained_path(artifact_root, str(spec["checkpoint_path"]), must_exist=True)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a run or checkpoint directory: {spec['checkpoint_path']}")
    checkpoint = resolve_checkpoint(directory)
    contained_path(artifact_root, checkpoint, must_exist=True)
    definition = _suite(str(spec.get("suite", "fixtures")), project_root)
    paths = _sources(spec, project_root)
    context.raise_if_cancelled()
    context.progress(0.02, message=f"Loading {checkpoint.name} for suite {definition.id}")
    model = load_kernel_scoring_model(
        checkpoint,
        data_root,
        batch_size=int(spec.get("batch_size", 8)),
        max_length=spec.get("max_length"),
    )
    training = spec.get("training_data")
    store = DataStore(data_root)
    try:
        report = run_suite(
            definition,
            model,
            store=store,
            source_paths=paths,
            allow_code_execution=False,
            allow_unisolated_network=False,
            training_data=(
                None
                if training is None
                else TrainingData(data_root=data_root, tokenized_id=str(training))
            ),
            should_cancel=lambda: bool(context.cancelled),
            on_task=_task_progress(context, len(definition.tasks)),
        )
    except EvaluationCancelledError:
        context.raise_if_cancelled()
        raise
    return _finish(context, report, store)


def _run_contamination(spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    project_root, _, data_root = _roots(context)
    definition = _suite(str(spec.get("suite", "fixtures")), project_root)
    paths = _sources(spec, project_root)
    context.raise_if_cancelled()
    context.progress(0.02, message=f"Scanning suite {definition.id} against the training build")
    store = DataStore(data_root)
    try:
        report = run_contamination(
            definition,
            TrainingData(data_root=data_root, tokenized_id=str(spec["training_data"])),
            store=store,
            source_paths=paths,
            should_cancel=lambda: bool(context.cancelled),
            on_task=_task_progress(context, len(definition.tasks)),
        )
    except EvaluationCancelledError:
        context.raise_if_cancelled()
        raise
    return _finish(context, report, store)


def run_evaluation_job(name: str, spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Execute one allowlisted evaluation job; never a shell command or generated code."""

    if name == "eval_suite":
        return _run_suite(spec, context)
    if name == "eval_contamination":
        return _run_contamination(spec, context)
    raise ValueError(f"unsupported evaluation control job: {name}")


__all__ = ["CODE_EXECUTION_NOTE", "run_evaluation_job"]
