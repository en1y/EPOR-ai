"""Covenant-gated evaluation and scaling-laboratory commands (v0.0.6).

``epor eval suite|contamination|compare|reports`` and
``epor scale ladder|profile|sweep|mixture|fit|project|reports``.  Every command
resolves its declared action (:data:`epor.actions.CLI_ACTIONS`) before doing
anything; running model-generated code additionally resolves the separate
``eval_code_execution`` action.  Every operator-supplied path must resolve
beneath ``EPOR_PROJECT_ROOT``.

Exit codes: 0 success, 1 any other failure, 2 input or verification failure,
3 refusal by the covenant or by a safety control (an unregistered benchmark
source, or a sandbox that cannot isolate the network without the separate
opt-in), 130 stopped by SIGINT/SIGTERM.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

if TYPE_CHECKING:
    from epor.evaluation.suites import SuiteDefinition

scale_app = typer.Typer(
    help="Scaling laboratory: ladder, host profiles, sweeps, mixtures, fits, projections."
)

_CANCELLED_EXIT = 130
_Split = Literal["train", "validation", "test"]
_ValidationSplit = Literal["validation", "test", "none"]
_Precision = Literal["bf16", "fp32"]
_SUITE_HELP = "'fixtures', 'safety', or a suite YAML beneath the project root."


def _cli() -> Any:
    import epor.cli

    return epor.cli


def _refuse(message: str, *, code: int = 2) -> typer.Exit:
    _cli().console.print(f"[red]refused[/red]: {message}")
    return typer.Exit(code=code)


def _input_errors() -> tuple[type[BaseException], ...]:
    import yaml

    from epor.data.store import DataStoreError
    from epor.evaluation import ContaminationError
    from epor.tokenizer import CorpusError, TokenizedBuildError, TokenizerArtifactError

    # ``ValueError`` covers pydantic validation, ReportError, ScalingReportError,
    # CheckpointError, bounded settings, and refused containment.
    return (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        DataStoreError,
        ContaminationError,
        CorpusError,
        TokenizedBuildError,
        TokenizerArtifactError,
        yaml.YAMLError,
    )


def _failure(exc: BaseException) -> typer.Exit:
    """Map a laboratory failure to the documented exit code."""

    from epor.evaluation import (
        CodeExecutionRefusedError,
        EvaluationCancelledError,
        SourceRefusedError,
    )
    from epor.training.runner import KernelRunError, is_cancellation

    if isinstance(exc, EvaluationCancelledError) or is_cancellation(exc):
        typer.echo(f"stopped: {exc}", err=True)
        return typer.Exit(code=_CANCELLED_EXIT)
    if isinstance(exc, (CodeExecutionRefusedError, SourceRefusedError)):
        return _refuse(str(exc), code=3)
    if isinstance(exc, KernelRunError):
        return _cli()._kernel_failure(exc)
    if isinstance(exc, _input_errors()):
        return _refuse(str(exc))
    _cli().console.print(f"[red]failed[/red] ({type(exc).__name__}): {exc}")
    return typer.Exit(code=1)


def _project_file(path: Path, *, label: str) -> Path:
    return _cli()._reference_path(path, label=label, must_exist=True)[1]


def _project_output(path: Path, *, label: str) -> Path:
    return _cli()._reference_path(path, label=label, must_exist=False)[1]


def _project_relative(path: Path, *, label: str) -> str:
    from epor.control.settings import project_root_from_environment

    resolved = _project_file(path, label=label)
    return resolved.relative_to(project_root_from_environment().resolve()).as_posix()


def _load_suite(value: str) -> SuiteDefinition:
    from epor.evaluation import fixture_suite, load_suite, safety_suite

    builtin = {"fixtures": fixture_suite, "safety": safety_suite}
    if value in builtin:
        return builtin[value]()
    return load_suite(_project_file(Path(value), label="suite path"))


def _source_paths(entries: list[str] | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for entry in entries or []:
        source_id, separator, raw = entry.partition("=")
        if not separator or not source_id or not raw:
            raise _refuse(f"--source expects ID=PATH, got {entry!r}")
        if source_id in paths:
            raise _refuse(f"--source names {source_id!r} twice")
        paths[source_id] = _project_file(Path(raw), label=f"source {source_id} path")
    return paths


def _store() -> Any:
    from epor.data.store import DataStore

    return DataStore(_cli()._data_root())


def _run_cancellable(work: Any) -> Any:
    """Run ``work(should_cancel)`` with SIGINT/SIGTERM mapped to a cooperative stop."""

    stop = threading.Event()
    try:
        with _cli()._StopRequests(stop.set):
            return work(stop.is_set)
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc


# --------------------------------------------------------------------------- eval


def eval_suite(
    checkpoint: Annotated[
        Path, typer.Argument(help="Kernel run directory, its checkpoints/ directory, or a step-*.")
    ],
    suite: Annotated[str, typer.Option(help=_SUITE_HELP)] = "fixtures",
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Registered benchmark file as SOURCE_ID=PATH; repeatable."),
    ] = None,
    training_data: Annotated[
        str | None,
        typer.Option(
            "--training-data",
            help="Tokenized build for contamination/memorization (default: the checkpoint's).",
        ),
    ] = None,
    allow_code_execution: Annotated[
        bool,
        typer.Option(
            "--allow-code-execution",
            help="Run model-generated code in the sandbox (separately authorized; off by default).",
        ),
    ] = False,
    allow_unisolated_network: Annotated[
        bool,
        typer.Option(
            "--allow-unisolated-network",
            help="Also run code when the sandbox cannot isolate the network (needs the above).",
        ),
    ] = False,
    batch_size: Annotated[int, typer.Option(min=1, max=256)] = 8,
    max_length: Annotated[int | None, typer.Option(min=2, max=8_192)] = None,
) -> None:
    """Score a kernel checkpoint on an evaluation suite and store the sealed report."""

    cli = _cli()
    cli.authorize_command("eval suite")
    if allow_unisolated_network and not allow_code_execution:
        raise _refuse("--allow-unisolated-network only applies with --allow-code-execution")
    if allow_code_execution:
        # Ordinary evaluation authority never implies running generated code.
        cli.authorize("eval_code_execution")
    directory = cli._contained_directory(checkpoint, label="checkpoint path")
    try:
        definition = _load_suite(suite)
        paths = _source_paths(source)
        data_root = cli._data_root()
        from epor.evaluation import TrainingData, run_suite, store_report
        from epor.evaluation.decoder import load_kernel_scoring_model
        from epor.evaluation.summary import report_summary
        from epor.training.run_checkpoint import resolve_checkpoint

        model = load_kernel_scoring_model(
            resolve_checkpoint(directory), data_root, batch_size=batch_size, max_length=max_length
        )
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc

    def work(should_cancel: Any) -> Any:
        store = _store()
        report = run_suite(
            definition,
            model,
            store=store,
            source_paths=paths,
            allow_code_execution=allow_code_execution,
            allow_unisolated_network=allow_unisolated_network,
            training_data=(
                None
                if training_data is None
                else TrainingData(data_root=data_root, tokenized_id=training_data)
            ),
            should_cancel=should_cancel,
            on_task=lambda task_id: typer.echo(f"task {task_id}", err=True),
        )
        return report, store_report(store, report)

    report, path = _run_cancellable(work)
    cli._echo_json(
        {
            **report_summary(report),
            "skipped_tasks": [item.model_dump(mode="json") for item in report.skipped_tasks],
            "sandbox": [
                {"task_id": task.task_id, **task.sandbox.model_dump(mode="json")}
                for task in report.tasks
                if task.sandbox is not None
            ],
            "report_path": str(path),
        }
    )


def eval_contamination(
    training_data: Annotated[
        str, typer.Option("--training-data", help="Tokenized build to scan against.")
    ],
    suite: Annotated[str, typer.Option(help=_SUITE_HELP)] = "fixtures",
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Registered benchmark file as SOURCE_ID=PATH; repeatable."),
    ] = None,
) -> None:
    """Scan a suite's items for n-gram overlap with a training build (no model)."""

    cli = _cli()
    cli.authorize_command("eval contamination")
    try:
        definition = _load_suite(suite)
        paths = _source_paths(source)
        data_root = cli._data_root()
        from epor.evaluation import TrainingData, run_contamination, store_report
        from epor.evaluation.summary import report_summary
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc

    def work(should_cancel: Any) -> Any:
        store = _store()
        report = run_contamination(
            definition,
            TrainingData(data_root=data_root, tokenized_id=training_data),
            store=store,
            source_paths=paths,
            should_cancel=should_cancel,
        )
        return report, store_report(store, report)

    report, path = _run_cancellable(work)
    flagged = [
        {
            "task_id": task.task_id,
            "flagged_items": task.metric("flagged_items").value,
            "assessed_items": task.metric("assessed_items").value,
            "flagged_rate": task.metric("flagged_rate").value,
        }
        for task in report.tasks
        if task.task_type == "contamination"
    ]
    cli._echo_json(
        {
            **report_summary(report),
            "contamination": flagged,
            "note": "n-gram overlap is a lower bound on contamination, never a clearance",
            "report_path": str(path),
        }
    )


def eval_compare(
    baseline: Annotated[str, typer.Argument(help="Baseline evaluation report ID.")],
    candidate: Annotated[str, typer.Argument(help="Candidate evaluation report ID.")],
    resamples: Annotated[int, typer.Option(min=1, max=100_000)] = 1_000,
    confidence: Annotated[float, typer.Option(min=0.5, max=0.999)] = 0.95,
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 0,
) -> None:
    """Compare two stored v0.0.6 evaluation reports (metric deltas, paired bootstrap)."""

    cli = _cli()
    cli.authorize_command("eval compare")
    from epor.evaluation import compare_reports
    from epor.evaluation.summary import load_verified_report

    try:
        data_root = cli._data_root()
        comparison = compare_reports(
            load_verified_report(data_root, baseline),
            load_verified_report(data_root, candidate),
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
    except Exception as exc:
        raise _failure(exc) from exc
    cli._echo_json(comparison.model_dump(mode="json"))


def eval_reports(
    report_id: Annotated[
        str | None, typer.Argument(help="Print one verified report in full.")
    ] = None,
) -> None:
    """List stored evaluation reports (verified), or print one in full."""

    cli = _cli()
    cli.authorize_command("eval reports")
    from epor.evaluation.summary import list_reports, load_verified_report

    data_root = cli._data_root()
    if report_id is not None:
        try:
            report = load_verified_report(data_root, report_id)
        except Exception as exc:
            raise _failure(exc) from exc
        cli._echo_json(report.model_dump(mode="json"))
        return
    listing = list_reports(data_root)
    cli._echo_json(listing)
    if listing["integrity_errors"]:
        raise typer.Exit(code=2)


def register_evaluation_commands(eval_app: typer.Typer) -> None:
    eval_app.command("suite")(eval_suite)
    eval_app.command("contamination")(eval_contamination)
    eval_app.command("compare")(eval_compare)
    eval_app.command("reports")(eval_reports)


# --------------------------------------------------------------------------- scale


def _load_config(path: Path, *, vocab_size: int | None, tokenized_id: str | None) -> Any:
    from epor.models.config import load_model_config
    from epor.scaling import derive_rung

    if vocab_size is not None and tokenized_id is not None:
        raise _refuse("pass at most one of --vocab-size and --vocab-from-build")
    config = load_model_config(_project_file(path, label="model config path"))
    if tokenized_id is not None:
        from epor.training.runner import build_vocab_size

        vocab_size = build_vocab_size(_cli()._data_root(), tokenized_id)
    return derive_rung(config, vocab_size)


def _store_scaling(report: Any) -> dict[str, Any]:
    from epor.scaling import store_scaling_report
    from epor.scaling.summary import scaling_summary

    path = store_scaling_report(_store(), report)
    return {**scaling_summary(report), "report_path": str(path), "warnings": report.warnings}


def _template(
    tokenized_id: str,
    *,
    seq_len: int,
    tokens_per_batch: int,
    micro_batch_size: int | None,
    precision: str,
    learning_rate: float,
    min_lr_ratio: float,
    warmup_steps: int,
    weight_decay: float,
    gradient_clip: float,
    checkpoint_every: int,
    validation_split: str,
    eval_max_steps: int | None,
    data_seed: int = 23,
    model_seed: int = 17,
    total_steps: int = 1,
) -> Any:
    from epor.training import OptimizerSettings, PackedTrainingSettings

    return PackedTrainingSettings(
        tokenized_id=tokenized_id,
        seq_len=seq_len,
        tokens_per_batch=tokens_per_batch,
        micro_batch_size=micro_batch_size,
        precision=precision,  # pyright: ignore[reportArgumentType]
        data_seed=data_seed,
        model_seed=model_seed,
        checkpoint_every=checkpoint_every,
        validation_split=None if validation_split == "none" else validation_split,  # pyright: ignore[reportArgumentType]
        eval_max_steps=eval_max_steps,
        optimizer=OptimizerSettings(
            learning_rate=learning_rate,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
        ),
    )


def _progress(kind: str) -> Any:
    def report(done: int, total: int, result: dict[str, Any]) -> None:
        name = result.get("mixture_id") or result.get("rung", "")
        typer.echo(f"{kind} {done}/{total} {name} {result.get('status', 'completed')}", err=True)

    return report


def scale_ladder(
    vocab_size: Annotated[
        int | None,
        typer.Option(min=8, max=1 << 20, help="Derive every rung with this vocabulary."),
    ] = None,
) -> None:
    """Show each ladder rung's parameter counts and local or projection-only status."""

    cli = _cli()
    cli.authorize_command("scale ladder")
    from epor.control.settings import project_root_from_environment
    from epor.scaling.summary import ladder_table

    try:
        rows = ladder_table(vocab_size, project_root_from_environment())
    except Exception as exc:
        raise _failure(exc) from exc
    cli._echo_json(
        {
            "rungs": rows,
            "note": (
                "Rung files are architecture contracts, not trained models; projection-only "
                "rungs are never allocated on this machine."
            ),
        }
    )


def scale_profile(
    config: Annotated[Path, typer.Argument(dir_okay=False, help="Model config (YAML/JSON).")],
    vocab_size: Annotated[int | None, typer.Option(min=8, max=1 << 20)] = None,
    vocab_from_build: Annotated[
        str | None,
        typer.Option("--vocab-from-build", help="Use this tokenized build's vocabulary."),
    ] = None,
    batch_size: Annotated[int, typer.Option(min=1, max=256)] = 4,
    seq_len: Annotated[int, typer.Option(min=2, max=8_192)] = 128,
    warmup_steps: Annotated[int, typer.Option(min=0, max=100)] = 2,
    timed_steps: Annotated[int, typer.Option(min=1, max=900)] = 5,
    precision: Annotated[_Precision, typer.Option()] = "fp32",
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 0,
    threads: Annotated[int | None, typer.Option(min=1, max=512)] = None,
    peak_flops: Annotated[
        float | None,
        typer.Option(
            "--peak-flops", min=1.0, help="Operator-supplied device peak FLOP/s (else MFU unknown)."
        ),
    ] = None,
    device_label: Annotated[str, typer.Option(help="Label recorded with the profile.")] = "cpu",
) -> None:
    """Measure training-step throughput and memory for one local config on this host."""

    cli = _cli()
    cli.authorize_command("scale profile")
    from epor.scaling import ProfileSettings, profile_training_step

    try:
        model = _load_config(config, vocab_size=vocab_size, tokenized_id=vocab_from_build)
        settings = ProfileSettings(
            batch_size=batch_size,
            seq_len=seq_len,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            precision=precision,
            seed=seed,
            threads=threads,
            peak_flops_per_second=peak_flops,
            device_label=device_label,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc
    report = _run_cancellable(
        lambda should_cancel: profile_training_step(model, settings, should_cancel=should_cancel)
    )
    payload = _store_scaling(report)
    payload["measured"] = {
        key: (report.measured or {}).get(key)
        for key in (
            "tokens_per_second",
            "step_seconds_mean",
            "achieved_flops_per_second",
            "mfu",
            "mfu_status",
            "process_peak_rss_bytes",
            "torch_threads",
        )
    }
    cli._echo_json(payload)


def scale_sweep(
    tokenized_id: Annotated[str, typer.Option("--tokenized-id", help="Verified tokenized build.")],
    budget: Annotated[
        list[int], typer.Option("--budget", min=1, help="Token budget per point; repeatable.")
    ],
    output: Annotated[Path, typer.Option("--output", help="Sweep directory (reuse to resume).")],
    rung: Annotated[
        list[Path] | None,
        typer.Option("--rung", help="Model config per rung; repeatable (default: the ladder)."),
    ] = None,
    seed: Annotated[
        list[int] | None, typer.Option("--seed", min=0, help="Seed per point; repeatable.")
    ] = None,
    seq_len: Annotated[int, typer.Option(min=1, max=8_192)] = 128,
    tokens_per_batch: Annotated[int, typer.Option(min=1, max=1_048_576)] = 4_096,
    micro_batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    precision: Annotated[_Precision, typer.Option()] = "bf16",
    learning_rate: Annotated[float, typer.Option(min=1e-8, max=1.0)] = 3e-4,
    min_lr_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    warmup_steps: Annotated[int, typer.Option(min=0, max=100_000)] = 0,
    weight_decay: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    gradient_clip: Annotated[float, typer.Option(min=1e-6, max=1_000.0)] = 1.0,
    checkpoint_every: Annotated[int, typer.Option(min=1, max=100_000)] = 50,
    validation_split: Annotated[_ValidationSplit, typer.Option()] = "validation",
    eval_max_steps: Annotated[int | None, typer.Option(min=1, max=100_000)] = None,
) -> None:
    """Train a rung x token-budget x seed sweep; rungs above the bound are projection-only."""

    cli = _cli()
    cli.authorize_command("scale sweep")
    from epor.control.settings import project_root_from_environment
    from epor.scaling import SweepPlan, run_sweep

    project_root = project_root_from_environment()
    destination = _project_output(output, label="output path")
    try:
        template = _template(
            tokenized_id,
            seq_len=seq_len,
            tokens_per_batch=tokens_per_batch,
            micro_batch_size=micro_batch_size,
            precision=precision,
            learning_rate=learning_rate,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            checkpoint_every=checkpoint_every,
            validation_split=validation_split,
            eval_max_steps=eval_max_steps,
            # Each point sets its own schedule length; this only carries warmup.
            total_steps=max(1, warmup_steps),
        )
        fields: dict[str, Any] = {
            "token_budgets": tuple(budget),
            "seeds": tuple(seed or [0]),
            "training": template,
        }
        if rung:
            fields["rungs"] = tuple(_project_relative(path, label="rung path") for path in rung)
        else:
            from epor.scaling import ladder_paths

            fields["rungs"] = tuple(
                _project_relative(path, label="rung path") for path in ladder_paths(project_root)
            )
        plan = SweepPlan(**fields)
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc
    report = _run_cancellable(
        lambda should_cancel: run_sweep(
            plan,
            data_root=cli._data_root(),
            output_dir=destination,
            project_root=project_root,
            should_cancel=should_cancel,
            on_point=_progress("point"),
        )
    )
    payload = _store_scaling(report)
    payload["points"] = [
        {
            key: point.get(key)
            for key in (
                "rung",
                "status",
                "seed",
                "max_steps",
                "parameters_non_embedding",
                "final_main_loss",
                "validation_main_loss",
                "reason",
                "error",
            )
            if point.get(key) is not None
        }
        for point in report.results["points"]
    ]
    cli._echo_json(payload)


def _parse_mixture(entry: str) -> Any:
    """``ID:domain=weight,domain=weight[@seed]`` -> :class:`NamedMixture`."""

    from epor.scaling import NamedMixture

    mixture_id, separator, rest = entry.partition(":")
    if not separator or not rest:
        raise _refuse(f"--mixture expects ID:domain=weight,...[@seed], got {entry!r}")
    rest, _, seed_text = rest.partition("@")
    weights: dict[str, float] = {}
    for part in rest.split(","):
        domain, equals, value = part.partition("=")
        if not equals or not domain:
            raise _refuse(f"--mixture weight must be domain=weight, got {part!r}")
        try:
            weights[domain.strip()] = float(value)
        except ValueError as exc:
            raise _refuse(f"--mixture weight {value!r} is not a number") from exc
    try:
        seed = int(seed_text) if seed_text else 0
    except ValueError as exc:
        raise _refuse(f"--mixture seed {seed_text!r} is not an integer") from exc
    return NamedMixture(mixture_id=mixture_id.strip(), weights=weights, seed=seed)


def scale_mixture(
    model: Annotated[Path, typer.Argument(dir_okay=False, help="Model config (YAML/JSON).")],
    tokenized_id: Annotated[str, typer.Option("--tokenized-id", help="Verified tokenized build.")],
    mixture: Annotated[
        list[str],
        typer.Option("--mixture", help="ID:domain=weight,domain=weight[@seed]; repeatable."),
    ],
    max_steps: Annotated[int, typer.Option(min=1, max=100_000, help="Steps per mixture.")],
    output: Annotated[Path, typer.Option("--output", help="Ablation directory (reuse to resume).")],
    evaluation_split: Annotated[_Split, typer.Option()] = "validation",
    evaluation_max_steps: Annotated[int | None, typer.Option(min=1, max=100_000)] = None,
    seq_len: Annotated[int, typer.Option(min=1, max=8_192)] = 128,
    tokens_per_batch: Annotated[int, typer.Option(min=1, max=1_048_576)] = 4_096,
    micro_batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    precision: Annotated[_Precision, typer.Option()] = "bf16",
    learning_rate: Annotated[float, typer.Option(min=1e-8, max=1.0)] = 3e-4,
    min_lr_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    warmup_steps: Annotated[int, typer.Option(min=0, max=100_000)] = 0,
    weight_decay: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    gradient_clip: Annotated[float, typer.Option(min=1e-6, max=1_000.0)] = 1.0,
    data_seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 23,
    model_seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 17,
    checkpoint_every: Annotated[int, typer.Option(min=1, max=100_000)] = 50,
) -> None:
    """Train one model per domain mixture and measure per-domain validation loss."""

    cli = _cli()
    cli.authorize_command("scale mixture")
    from epor.control.settings import project_root_from_environment
    from epor.scaling import MixturePlan, mixture_summary, run_mixture_ablation

    project_root = project_root_from_environment()
    destination = _project_output(output, label="output path")
    try:
        mixtures = tuple(_parse_mixture(entry) for entry in mixture)
        plan = MixturePlan(
            model=_project_relative(model, label="model config path"),
            mixtures=mixtures,
            max_steps=max_steps,
            evaluation_split=evaluation_split,
            evaluation_max_steps=evaluation_max_steps,
            training=_template(
                tokenized_id,
                seq_len=seq_len,
                tokens_per_batch=tokens_per_batch,
                micro_batch_size=micro_batch_size,
                precision=precision,
                learning_rate=learning_rate,
                min_lr_ratio=min_lr_ratio,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                gradient_clip=gradient_clip,
                checkpoint_every=checkpoint_every,
                validation_split="none",
                eval_max_steps=None,
                data_seed=data_seed,
                model_seed=model_seed,
                total_steps=max_steps,
            ),
        )
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc
    report = _run_cancellable(
        lambda should_cancel: run_mixture_ablation(
            plan,
            data_root=cli._data_root(),
            output_dir=destination,
            project_root=project_root,
            should_cancel=should_cancel,
            on_mixture=_progress("mixture"),
        )
    )
    payload = _store_scaling(report)
    payload["mixtures"] = mixture_summary(report)
    cli._echo_json(payload)


def scale_fit(
    sweep_report: Annotated[str, typer.Argument(help="Scaling-sweep report ID.")],
) -> None:
    """Fit L(N), L(D), and the joint form to a stored sweep report."""

    cli = _cli()
    cli.authorize_command("scale fit")
    from epor.scaling import fit_sweep
    from epor.scaling.summary import load_verified_scaling_reports

    try:
        source = load_verified_scaling_reports(cli._data_root(), sweep_report)[0]
        report = fit_sweep(source)
    except Exception as exc:
        raise _failure(exc) from exc
    payload = _store_scaling(report)
    payload["results"] = report.results
    cli._echo_json(payload)


def scale_project(
    inputs: Annotated[
        Path, typer.Option("--inputs", dir_okay=False, help="Operator-written inputs YAML.")
    ],
    target: Annotated[Path, typer.Option("--target", dir_okay=False, help="Target model config.")],
    fit: Annotated[
        str | None, typer.Option("--fit", help="Scaling-fit report ID for expected loss.")
    ] = None,
) -> None:
    """Project compute, storage, energy, and cost for a target run; provisions nothing."""

    cli = _cli()
    cli.authorize_command("scale project")
    from epor.models.config import load_model_config
    from epor.scaling import load_projection_inputs, project_training
    from epor.scaling.summary import load_verified_scaling_reports

    try:
        operator_inputs = load_projection_inputs(_project_file(inputs, label="inputs path"))
        config = load_model_config(_project_file(target, label="target config path"))
        fit_report = None
        if fit is not None:
            fit_report = load_verified_scaling_reports(cli._data_root(), fit)[0]
            if fit_report.kind != "scaling-fit":
                raise ValueError(f"report {fit} is a {fit_report.kind}, not a scaling-fit")
        report = project_training(config, operator_inputs, fit_report=fit_report)
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc
    payload = _store_scaling(report)
    payload["unknown_inputs"] = report.results["unknown_inputs"]
    payload["approval"] = report.results["approval"]
    payload["results"] = report.results
    cli._echo_json(payload)
    typer.echo(
        "approval required: a projection is not authorization and provisions nothing",
        err=True,
    )


def scale_reports(
    report_id: Annotated[
        str | None, typer.Argument(help="Print every stored measurement of one report ID.")
    ] = None,
) -> None:
    """List stored scaling reports (verified), or print one in full."""

    cli = _cli()
    cli.authorize_command("scale reports")
    from epor.scaling.summary import list_scaling_reports, load_verified_scaling_reports

    data_root = cli._data_root()
    if report_id is not None:
        try:
            reports = load_verified_scaling_reports(data_root, report_id)
        except Exception as exc:
            raise _failure(exc) from exc
        cli._echo_json([report.model_dump(mode="json") for report in reports])
        return
    listing = list_scaling_reports(data_root)
    cli._echo_json(listing)
    if listing["integrity_errors"]:
        raise typer.Exit(code=2)


scale_app.command("ladder")(scale_ladder)
scale_app.command("profile")(scale_profile)
scale_app.command("sweep")(scale_sweep)
scale_app.command("mixture")(scale_mixture)
scale_app.command("fit")(scale_fit)
scale_app.command("project")(scale_project)
scale_app.command("reports")(scale_reports)


__all__ = ["register_evaluation_commands", "scale_app"]
