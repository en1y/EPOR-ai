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
        type: multiple-choice        # see "Task types" below
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
without ``allow_code_execution=True``, validation-loss tasks without a kernel
checkpoint, and contamination/memorization tasks without training data.  A
suite containing *only* skipped tasks is an error.

Task types
----------
Item tasks (``items`` or ``source``): ``multiple-choice``, ``exact-match``,
``math``, ``code`` (:mod:`~epor.evaluation.tasks`), ``safety-preference`` and
``safety-rubric`` (:mod:`~epor.evaluation.safety_behaviour`).

Data tasks (no items): ``validation-loss`` (the checkpoint's own build);
``contamination`` (optional ``contamination:`` settings — n-gram overlap of
the suite's other item tasks, or of ``targets``, with the training build);
``memorization`` (optional ``memorization:`` settings — train vs held-out
extraction rate); ``canary-exposure`` (optional ``canary:`` settings — a
baseline unless an operator planted the canaries).

Training data is the kernel checkpoint's tokenized build, or an explicit
:class:`TrainingData` passed to :func:`run_suite`.  :func:`run_contamination`
runs a suite's contamination tasks without any model (a ``data-only`` report).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from epor.data.models import Identifier, Sha256
from epor.data.store import DataStore, sha256_model

from .calibration import DEFAULT_CALIBRATION_BINS
from .contamination import (
    ContaminationSettings,
    ScannableItem,
    build_corpus,
    contamination_items,
    scan_contamination,
    scan_split_leakage,
)
from .memorization import (
    CanarySettings,
    MemorizationSettings,
    memorization_from_build,
    run_canary_exposure,
)
from .report import (
    CLAIM_BOUNDARIES,
    PROJECT_ROOT,
    DataSource,
    EvaluationReport,
    HarnessIdentity,
    MeasurementScope,
    ModelIdentity,
    TaskResult,
    TaskType,
    harness_identity,
    seal_report,
)
from .safety_behaviour import (
    SafetyPreferenceItem,
    SafetyRubricItem,
    run_safety_preference,
    run_safety_rubric,
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
    run_code,
    run_exact_match,
    run_math,
    run_multiple_choice,
    run_validation_loss,
)

SUITE_DIRECTORY = PROJECT_ROOT / "configs" / "evaluation"
FIXTURE_SUITE_PATH = SUITE_DIRECTORY / "harness-fixtures.yaml"
SAFETY_SUITE_PATH = SUITE_DIRECTORY / "safety-behaviour.yaml"

_ITEM_MODELS: dict[str, type[ScannableItem]] = {
    "multiple-choice": MultipleChoiceItem,
    "exact-match": ExactMatchItem,
    "math": MathItem,
    "code": CodeItem,
    "safety-preference": SafetyPreferenceItem,
    "safety-rubric": SafetyRubricItem,
}
DATA_TASK_TYPES = frozenset({"validation-loss", "contamination", "memorization", "canary-exposure"})


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
    contamination: ContaminationSettings | None = None
    memorization: MemorizationSettings | None = None
    canary: CanarySettings | None = None

    @model_validator(mode="after")
    def one_item_origin(self) -> TaskSpec:
        for name, owner in (
            ("contamination", "contamination"),
            ("memorization", "memorization"),
            ("canary", "canary-exposure"),
        ):
            if getattr(self, name) is not None and self.type != owner:
                raise ValueError(f"{name} settings apply only to {owner} tasks")
        if self.type in DATA_TASK_TYPES:
            if self.items is not None or self.source is not None:
                raise ValueError(f"{self.type} tasks take no items or source")
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
        item_tasks = {task.id for task in self.tasks if task.type not in DATA_TASK_TYPES}
        for task in self.tasks:
            targets = task.contamination.targets if task.contamination is not None else None
            if targets is not None and not set(targets) <= item_tasks:
                raise ValueError(f"contamination task {task.id} targets unknown item tasks")
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


def safety_suite() -> SuiteDefinition:
    """The project-authored G10 model-behaviour suite (not evidence of safety)."""

    return load_suite(SAFETY_SUITE_PATH)


@dataclass(frozen=True, slots=True)
class TrainingData:
    """The tokenized build a model was trained on (and its parent data-engine build)."""

    data_root: Path
    tokenized_id: str


def _cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise EvaluationCancelledError("evaluation cancellation requested")


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
    training_data: TrainingData | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_task: Callable[[str], None] | None = None,
) -> EvaluationReport:
    """Run every task of ``suite`` against ``model`` and return the sealed report.

    External sources need ``store`` (for registrations) and a local path per
    source ID in ``source_paths``.  Contamination and memorization tasks use
    ``training_data``, defaulting to a kernel checkpoint's own tokenized build.
    ``should_cancel`` is polled between tasks and inside long data scans.
    """

    checkpoint: Path | None = getattr(model, "checkpoint_dir", None)
    data_root: Path | None = getattr(model, "data_root", None)
    model_build: str | None = getattr(model, "tokenized_id", None)
    if training_data is None and data_root is not None and model_build is not None:
        training_data = TrainingData(data_root=data_root, tokenized_id=model_build)
    runner = _SuiteRun(suite, store, dict(source_paths or {}), training_data, should_cancel)
    for spec in suite.tasks:
        _cancel(should_cancel)
        if on_task is not None:
            on_task(spec.id)
        if spec.type == "code" and not allow_code_execution:
            runner.skip(spec, "code execution not enabled")
        elif spec.type == "validation-loss":
            if checkpoint is None or data_root is None:
                runner.skip(spec, "model is not a kernel checkpoint")
                continue
            runner.add(
                run_validation_loss(
                    spec.id,
                    checkpoint,
                    data_root,
                    scope=suite.scope,
                    split=spec.split,
                    max_steps=spec.max_steps,
                    should_cancel=should_cancel,
                ),
                uses_build=model_build,
            )
        elif spec.type == "contamination":
            runner.contamination(spec)
        elif spec.type == "memorization":
            runner.memorization(spec, model)
        elif spec.type == "canary-exposure":
            runner.add(
                run_canary_exposure(
                    spec.id,
                    model,
                    scope=suite.scope,
                    settings=spec.canary,
                    should_cancel=should_cancel,
                )
            )
        else:
            runner.add(
                _run_task(
                    spec,
                    runner.items(spec),
                    model,
                    suite.scope,
                    allow_code_execution=allow_code_execution,
                    allow_unisolated_network=allow_unisolated_network,
                    sandbox=sandbox,
                )
            )
    if model_build is not None:
        runner.build_tasks.setdefault(model_build, [])
    seeds: dict[str, int] = {"evaluation": suite.seed, **getattr(model, "seeds", {})}
    return runner.seal(model.identity, seeds, harness)


def run_contamination(
    suite: SuiteDefinition,
    training_data: TrainingData,
    *,
    store: DataStore | None = None,
    source_paths: Mapping[str, Path] | None = None,
    harness: HarnessIdentity | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_task: Callable[[str], None] | None = None,
) -> EvaluationReport:
    """Scan a suite's items against a training build without any model.

    Runs the suite's ``contamination`` tasks; a suite without one gets a
    default task ``contamination`` over all of its item tasks.  Every other
    task is listed as skipped.  The report's model identity is ``data-only``
    and names the build's tokenizer.
    """

    from epor.tokenizer.shards import load_tokenized_manifest

    manifest = load_tokenized_manifest(training_data.data_root, training_data.tokenized_id)
    specs = [task for task in suite.tasks if task.type == "contamination"]
    if not specs:
        specs = [TaskSpec(id="contamination", type="contamination")]
    runner = _SuiteRun(suite, store, dict(source_paths or {}), training_data, should_cancel)
    for spec in suite.tasks:
        if spec.type != "contamination":
            runner.skip(spec, "data-only contamination run")
    for spec in specs:
        _cancel(should_cancel)
        if on_task is not None:
            on_task(spec.id)
        runner.contamination(spec)
    identity = ModelIdentity(
        kind="data-only",
        model_id="none",
        tokenizer_id=manifest.tokenizer_id,
        tokenizer_kind="epor-byte-bpe",
    )
    return runner.seal(identity, {"evaluation": suite.seed}, harness)


class _SuiteRun:
    """Accumulates results, skips, lineage, and loaded items for one suite run."""

    def __init__(
        self,
        suite: SuiteDefinition,
        store: DataStore | None,
        paths: dict[str, Path],
        training_data: TrainingData | None,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        self.suite = suite
        self.store = store
        self.paths = paths
        self.training_data = training_data
        self.should_cancel = should_cancel
        self.lineage: list[DataSource] = []
        self.results: list[TaskResult] = []
        self.skipped: list[dict[str, str]] = []
        self.build_tasks: dict[str, list[str]] = {}
        self._items: dict[str, list[ScannableItem]] = {}

    def skip(self, spec: TaskSpec, reason: str) -> None:
        self.skipped.append({"task_id": spec.id, "reason": reason})

    def add(self, result: TaskResult, *, uses_build: str | None = None) -> None:
        self.results.append(result)
        if uses_build is not None:
            self.build_tasks.setdefault(uses_build, []).append(result.task_id)

    def items(self, spec: TaskSpec) -> list[ScannableItem]:
        if spec.id not in self._items:
            self._items[spec.id] = _task_items(
                self.suite, spec, self.store, self.paths, self.lineage
            )
        return self._items[spec.id]

    def contamination(self, spec: TaskSpec) -> None:
        if self.training_data is None:
            self.skip(spec, "no training data (tokenized build) was supplied")
            return
        settings = spec.contamination or ContaminationSettings()
        corpus = build_corpus(self.training_data.data_root, self.training_data.tokenized_id)
        if settings.mode == "split-leakage":
            result = scan_split_leakage(
                spec.id,
                corpus,
                scope=self.suite.scope,
                settings=settings,
                should_cancel=self.should_cancel,
            )
        else:
            targets = settings.targets or [
                task.id for task in self.suite.tasks if task.type not in DATA_TASK_TYPES
            ]
            if not targets:
                self.skip(spec, "the suite has no item tasks to check")
                return
            by_id = {task.id: task for task in self.suite.tasks}
            scanned = [
                item
                for target in targets
                for item in contamination_items(target, self.items(by_id[target]))
            ]
            # Record that this task read each target's sources too.
            self.lineage += [
                source.model_copy(update={"task_ids": [spec.id]})
                for source in self.lineage
                if set(source.task_ids) & set(targets)
            ]
            result = scan_contamination(
                spec.id,
                scanned,
                corpus,
                scope=self.suite.scope,
                settings=settings,
                should_cancel=self.should_cancel,
            )
        self.add(result, uses_build=self.training_data.tokenized_id)

    def memorization(self, spec: TaskSpec, model: ScoringModel) -> None:
        if self.training_data is None:
            self.skip(spec, "no training data (tokenized build) was supplied")
            return
        from epor.tokenizer.shards import load_tokenized_manifest

        manifest = load_tokenized_manifest(
            self.training_data.data_root, self.training_data.tokenized_id
        )
        if model.identity.tokenizer_id != manifest.tokenizer_id:
            self.skip(spec, "model tokenizer differs from the training build's tokenizer")
            return
        result = memorization_from_build(
            spec.id,
            model,
            self.training_data.data_root,
            self.training_data.tokenized_id,
            scope=self.suite.scope,
            settings=spec.memorization,
            seed=self.suite.seed,
            should_cancel=self.should_cancel,
        )
        self.add(result, uses_build=self.training_data.tokenized_id)

    def seal(
        self,
        identity: ModelIdentity,
        seeds: dict[str, int],
        harness: HarnessIdentity | None,
    ) -> EvaluationReport:
        if not self.results:
            raise ValueError("no task in the suite could run with the given model and options")
        lineage = list(self.lineage)
        for tokenized_id, task_ids in self.build_tasks.items():
            lineage.append(
                DataSource(
                    kind="tokenized-build",
                    source_id=tokenized_id,
                    content_sha256=tokenized_id,
                    task_ids=task_ids,
                )
            )
        return seal_report(
            suite_id=self.suite.id,
            suite_version=self.suite.version,
            suite_sha256=self.suite.sha256,
            scope=self.suite.scope,
            claim_boundary=CLAIM_BOUNDARIES[self.suite.scope],
            model=identity,
            data=_merge_lineage(lineage),
            seeds=seeds,
            harness=harness or harness_identity(),
            tasks=self.results,
            skipped_tasks=self.skipped,
        )


def _task_items(
    suite: SuiteDefinition,
    spec: TaskSpec,
    store: DataStore | None,
    paths: Mapping[str, Path],
    lineage: list[DataSource],
) -> list[ScannableItem]:
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
    return list[ScannableItem](loaded.items)


def _run_task(
    spec: TaskSpec,
    items: list[ScannableItem],
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
    if spec.type == "safety-preference":
        return run_safety_preference(
            spec.id,
            [item for item in items if isinstance(item, SafetyPreferenceItem)],
            model,
            scope=scope,
        )
    if spec.type == "safety-rubric":
        return run_safety_rubric(
            spec.id,
            [item for item in items if isinstance(item, SafetyRubricItem)],
            model,
            scope=scope,
            max_new_tokens=spec.max_new_tokens or 48,
            stop=spec.stop if spec.stop is not None else ("\n",),
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
    "DATA_TASK_TYPES",
    "FIXTURE_SUITE_PATH",
    "SAFETY_SUITE_PATH",
    "SUITE_DIRECTORY",
    "EvaluationCancelledError",
    "ExternalSourceRef",
    "SuiteDefinition",
    "TaskSpec",
    "TrainingData",
    "fixture_suite",
    "load_suite",
    "run_contamination",
    "run_suite",
    "safety_suite",
]
