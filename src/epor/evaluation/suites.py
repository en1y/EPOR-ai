"""Suite definitions (YAML under ``configs/evaluation/``) and the suite runner.

A suite is versioned data::

    schema_version: 1
    id: epor-harness-fixtures
    version: 1
    scope: harness-fixture          # or proxy-scale
    title: ...
    seed: 0
    tasks:
      - id: arithmetic-mc
        type: multiple-choice        # exact-match | math | code | validation-loss
        items: [...]                 # project-authored items, inline
      - id: operator-mmlu
        type: multiple-choice
        source:                      # an operator-registered local file instead
          source_id: mmlu-reviewed
          format: multiple-choice-jsonl
          sha256: <optional pin>
          limit: <optional first-N>

:func:`run_suite` produces a sealed :class:`~epor.evaluation.report.EvaluationReport`.
Tasks that cannot run in this invocation are listed in the report's
``skipped_tasks`` with a reason instead of being silently dropped: code tasks
without ``allow_code_execution=True``, and validation-loss tasks without a
kernel checkpoint.  A suite containing *only* skipped tasks is an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from epor.data.models import Identifier, Sha256
from epor.data.store import DataStore, sha256_model

from .calibration import DEFAULT_CALIBRATION_BINS
from .report import (
    CLAIM_BOUNDARIES,
    PROJECT_ROOT,
    DataSource,
    EvaluationReport,
    HarnessIdentity,
    MeasurementScope,
    TaskResult,
    TaskType,
    harness_identity,
    seal_report,
)
from .sandbox import CodeSandbox
from .scoring import ScoringModel
from .sources import FORMAT_TASK_TYPES, ExternalFormat, load_registered_items
from .tasks import (
    CODE_STOP_SEQUENCES,
    DEFAULT_NORMALIZATION,
    CodeItem,
    ExactMatchItem,
    MathItem,
    MultipleChoiceItem,
    Normalization,
    TaskItem,
    run_code,
    run_exact_match,
    run_math,
    run_multiple_choice,
    run_validation_loss,
)

SUITE_DIRECTORY = PROJECT_ROOT / "configs" / "evaluation"
FIXTURE_SUITE_PATH = SUITE_DIRECTORY / "harness-fixtures.yaml"

_ITEM_MODELS: dict[str, type[TaskItem]] = {
    "multiple-choice": MultipleChoiceItem,
    "exact-match": ExactMatchItem,
    "math": MathItem,
    "code": CodeItem,
}


class EvaluationCancelledError(RuntimeError):
    """The suite run was cancelled between tasks."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExternalSourceRef(_Frozen):
    source_id: Identifier
    format: ExternalFormat
    sha256: Sha256 | None = None
    limit: int | None = Field(default=None, ge=1)


class TaskSpec(_Frozen):
    id: Identifier
    type: TaskType
    items: list[dict[str, Any]] | None = None
    source: ExternalSourceRef | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=4_096)
    stop: list[str] | None = None
    normalization: list[Normalization] | None = None
    calibration_bins: int = Field(default=DEFAULT_CALIBRATION_BINS, ge=1, le=100)
    split: Literal["train", "validation", "test"] = "validation"
    max_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def one_item_origin(self) -> TaskSpec:
        if self.type == "validation-loss":
            if self.items is not None or self.source is not None:
                raise ValueError("validation-loss tasks read the checkpoint's own build")
            return self
        if (self.items is None) == (self.source is None):
            raise ValueError(f"task {self.id} needs exactly one of items or source")
        if self.items is not None:
            model = _ITEM_MODELS[self.type]
            for item in self.items:
                model.model_validate(item)
        if self.source is not None and FORMAT_TASK_TYPES[self.source.format] != self.type:
            raise ValueError(f"format {self.source.format} does not produce {self.type} items")
        return self


class SuiteDefinition(_Frozen):
    schema_version: Literal[1] = 1
    id: Identifier
    version: int = Field(ge=1)
    scope: MeasurementScope
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4_000)
    seed: int = Field(default=0, ge=0)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tasks(self) -> SuiteDefinition:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("suite task IDs must be unique")
        if self.scope == "harness-fixture" and any(t.source is not None for t in self.tasks):
            raise ValueError("harness-fixture suites hold only project-authored inline items")
        return self

    @property
    def sha256(self) -> str:
        return sha256_model(self)


def load_suite(path: str | Path) -> SuiteDefinition:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SuiteDefinition.model_validate(document)


def fixture_suite() -> SuiteDefinition:
    """The project-authored harness-fixture suite (not a benchmark)."""

    return load_suite(FIXTURE_SUITE_PATH)


def _merge_lineage(sources: list[DataSource]) -> list[DataSource]:
    merged: dict[tuple[str, str, str], DataSource] = {}
    for source in sources:
        key = (source.kind, source.source_id, source.content_sha256)
        if key in merged:
            previous = merged[key]
            merged[key] = previous.model_copy(
                update={"task_ids": sorted({*previous.task_ids, *source.task_ids})}
            )
        else:
            merged[key] = source.model_copy(update={"task_ids": sorted(source.task_ids)})
    return [merged[key] for key in sorted(merged)]


def run_suite(
    suite: SuiteDefinition,
    model: ScoringModel,
    *,
    store: DataStore | None = None,
    source_paths: Mapping[str, Path] | None = None,
    allow_code_execution: bool = False,
    allow_unisolated_network: bool = False,
    sandbox: CodeSandbox | None = None,
    harness: HarnessIdentity | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_task: Callable[[str], None] | None = None,
) -> EvaluationReport:
    """Run every task of ``suite`` against ``model`` and return the sealed report.

    External sources need ``store`` (for registrations) and a local path per
    source ID in ``source_paths``.  ``should_cancel`` is polled between tasks.
    """

    paths = dict(source_paths or {})
    checkpoint: Path | None = getattr(model, "checkpoint_dir", None)
    data_root: Path | None = getattr(model, "data_root", None)
    lineage: list[DataSource] = []
    results: list[TaskResult] = []
    skipped: list[dict[str, str]] = []
    for spec in suite.tasks:
        if should_cancel is not None and should_cancel():
            raise EvaluationCancelledError("evaluation cancellation requested")
        if on_task is not None:
            on_task(spec.id)
        if spec.type == "code" and not allow_code_execution:
            skipped.append({"task_id": spec.id, "reason": "code execution not enabled"})
            continue
        if spec.type == "validation-loss":
            if checkpoint is None or data_root is None:
                skipped.append({"task_id": spec.id, "reason": "model is not a kernel checkpoint"})
                continue
            results.append(
                run_validation_loss(
                    spec.id,
                    checkpoint,
                    data_root,
                    scope=suite.scope,
                    split=spec.split,
                    max_steps=spec.max_steps,
                    should_cancel=should_cancel,
                )
            )
            continue
        items = _task_items(suite, spec, store, paths, lineage)
        results.append(
            _run_task(
                spec,
                items,
                model,
                suite.scope,
                allow_code_execution=allow_code_execution,
                allow_unisolated_network=allow_unisolated_network,
                sandbox=sandbox,
            )
        )
    if not results:
        raise ValueError("no task in the suite could run with the given model and options")
    tokenized_id: str | None = getattr(model, "tokenized_id", None)
    if tokenized_id is not None:
        lineage.append(
            DataSource(
                kind="tokenized-build",
                source_id=tokenized_id,
                content_sha256=tokenized_id,
                task_ids=[r.task_id for r in results if r.task_type == "validation-loss"],
            )
        )
    seeds: dict[str, int] = {"evaluation": suite.seed, **getattr(model, "seeds", {})}
    return seal_report(
        suite_id=suite.id,
        suite_version=suite.version,
        suite_sha256=suite.sha256,
        scope=suite.scope,
        claim_boundary=CLAIM_BOUNDARIES[suite.scope],
        model=model.identity,
        data=_merge_lineage(lineage),
        seeds=seeds,
        harness=harness or harness_identity(),
        tasks=results,
        skipped_tasks=skipped,
    )


def _task_items(
    suite: SuiteDefinition,
    spec: TaskSpec,
    store: DataStore | None,
    paths: Mapping[str, Path],
    lineage: list[DataSource],
) -> list[TaskItem]:
    if spec.items is not None:
        model = _ITEM_MODELS[spec.type]
        items = [model.model_validate(item) for item in spec.items]
        lineage.append(
            DataSource(
                kind="project-fixture",
                source_id=f"{suite.id}/{spec.id}",
                content_sha256=sha256_model([item.model_dump(mode="json") for item in items]),
                task_ids=[spec.id],
            )
        )
        return items
    assert spec.source is not None
    if store is None:
        raise ValueError("external benchmark sources need a data store for their registrations")
    path = paths.get(spec.source.source_id)
    if path is None:
        raise ValueError(f"no local file was supplied for source {spec.source.source_id!r}")
    loaded = load_registered_items(
        store,
        spec.source.source_id,
        path,
        spec.source.format,
        expected_sha256=spec.source.sha256,
        limit=spec.source.limit,
        task_ids=[spec.id],
    )
    lineage.append(loaded.lineage)
    return loaded.items


def _run_task(
    spec: TaskSpec,
    items: list[TaskItem],
    model: ScoringModel,
    scope: MeasurementScope,
    *,
    allow_code_execution: bool,
    allow_unisolated_network: bool,
    sandbox: CodeSandbox | None,
) -> TaskResult:
    if spec.type == "multiple-choice":
        return run_multiple_choice(
            spec.id,
            [item for item in items if isinstance(item, MultipleChoiceItem)],
            model,
            scope=scope,
            calibration_bins=spec.calibration_bins,
        )
    if spec.type == "exact-match":
        return run_exact_match(
            spec.id,
            [item for item in items if isinstance(item, ExactMatchItem)],
            model,
            scope=scope,
            max_new_tokens=spec.max_new_tokens or 32,
            stop=spec.stop if spec.stop is not None else ("\n",),
            normalization=spec.normalization or DEFAULT_NORMALIZATION,
        )
    if spec.type == "math":
        return run_math(
            spec.id,
            [item for item in items if isinstance(item, MathItem)],
            model,
            scope=scope,
            max_new_tokens=spec.max_new_tokens or 64,
            stop=spec.stop if spec.stop is not None else ("\n\n", "Question:"),
        )
    return run_code(
        spec.id,
        [item for item in items if isinstance(item, CodeItem)],
        model,
        scope=scope,
        allow_code_execution=allow_code_execution,
        allow_unisolated_network=allow_unisolated_network,
        sandbox=sandbox,
        max_new_tokens=spec.max_new_tokens or 128,
        stop=spec.stop if spec.stop is not None else CODE_STOP_SEQUENCES,
    )


__all__ = [
    "FIXTURE_SUITE_PATH",
    "SUITE_DIRECTORY",
    "EvaluationCancelledError",
    "ExternalSourceRef",
    "SuiteDefinition",
    "TaskSpec",
    "fixture_suite",
    "load_suite",
    "run_suite",
]
