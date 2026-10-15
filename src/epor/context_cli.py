"""Covenant-gated context-laboratory commands (v0.0.8): ``epor context ...``.

Every command resolves its declared action (:data:`epor.actions.CLI_ACTIONS`)
before doing anything, and every operator-supplied path must resolve beneath
``EPOR_PROJECT_ROOT``.

Exit codes: 0 success, 1 any other failure, 2 input or verification failure
(including a training run refused by its memory budget), 3 refusal by the
covenant or by the certification gate, 4 training stopped at its memory budget
(checkpoints kept), 130 stopped by SIGINT/SIGTERM.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

context_app = typer.Typer(
    help="Context laboratory: extension, G6, host probes, sweeps, projections, certification."
)

_CANCELLED_EXIT = 130
_Method = Literal["yarn", "linear", "ntk", "none"]
_Quantize = Literal["none", "int8"]


def _cli() -> Any:
    import epor.cli

    return epor.cli


def _refuse(message: str, *, code: int = 2) -> typer.Exit:
    _cli().console.print(f"[red]refused[/red]: {message}")
    return typer.Exit(code=code)


def _failure(exc: BaseException) -> typer.Exit:
    from epor.context.status import CertificationRefusedError
    from epor.training.runner import KernelRunError, is_cancellation

    if is_cancellation(exc) or isinstance(exc, InterruptedError):
        typer.echo(f"stopped: {exc}", err=True)
        return typer.Exit(code=_CANCELLED_EXIT)
    if isinstance(exc, CertificationRefusedError):
        _cli().console.print("[red]validated context not raised[/red]:")
        for reason in exc.reasons:
            _cli().console.print(f"  - {reason}")
        return typer.Exit(code=3)
    if isinstance(exc, KernelRunError):
        return _cli()._kernel_failure(exc)
    if isinstance(exc, (ValueError, FileNotFoundError, NotADirectoryError, KeyError)):
        return _refuse(str(exc))
    _cli().console.print(f"[red]failed[/red] ({type(exc).__name__}): {exc}")
    return typer.Exit(code=1)


def _run(work: Any) -> Any:
    stop = threading.Event()
    try:
        with _cli()._StopRequests(stop.set):
            return work(stop.is_set)
    except typer.Exit:
        raise
    except Exception as exc:
        raise _failure(exc) from exc


def _checkpoint(path: Path) -> Path:
    from epor.training.run_checkpoint import resolve_checkpoint

    directory = _cli()._contained_directory(path, label="checkpoint path")
    try:
        return resolve_checkpoint(directory)
    except Exception as exc:
        raise _failure(exc) from exc


def _parent(path: Path) -> tuple[Path, dict[str, Any], Any]:
    from epor.models.config import model_config_from_mapping
    from epor.training.run_checkpoint import read_kernel_checkpoint

    directory = _checkpoint(path)
    trainer = read_kernel_checkpoint(directory)
    return directory, trainer, model_config_from_mapping(trainer["model_config"])


def _ref(directory: Path) -> Any:
    from epor.alignment.model import checkpoint_identity
    from epor.training.kernel import CheckpointRef

    return CheckpointRef(path=str(directory), checkpoint_sha256=checkpoint_identity(directory))


def _train(
    model: Any,
    parent_trainer: dict[str, Any],
    parent_dir: Path,
    *,
    output: Path,
    max_steps: int,
    seq_len: int,
    tokens_per_batch: int,
    learning_rate: float,
    warmup_steps: int,
    min_document_pairs: int | None,
    precision: str,
    strategy: str,
    world_size: int,
    checkpoint_every: int,
    memory_budget_gib: float | None,
) -> dict[str, Any]:
    from epor.training import OptimizerSettings, PackedTrainingSettings

    cli = _cli()
    project_root, destination = cli._reference_path(output, label="output path", must_exist=False)
    if (strategy == "none") != (world_size == 1):
        raise _refuse("strategy 'none' runs one process and ddp/fsdp2 need --world-size >= 2")
    try:
        parent_settings = PackedTrainingSettings.model_validate(parent_trainer["settings"])
        settings = PackedTrainingSettings(
            tokenized_id=parent_settings.tokenized_id,
            init_checkpoint=_ref(parent_dir),
            seq_len=seq_len,
            tokens_per_batch=tokens_per_batch,
            min_document_pairs=min_document_pairs,
            precision=precision,  # type: ignore[arg-type]
            checkpoint_every=checkpoint_every,
            keep_checkpoints=2,
            optimizer=OptimizerSettings(
                learning_rate=learning_rate, warmup_steps=warmup_steps, total_steps=max_steps
            ),
        )
    except Exception as exc:
        raise _failure(exc) from exc
    return cli._run_kernel_command(
        model,
        destination,
        settings,
        max_steps=max_steps,
        resume_from=None,
        strategy=strategy,
        world_size=world_size,
        launch_timeout=6 * 3600.0,
        project_root=project_root,
        memory_budget_gib=memory_budget_gib,
    )


_Output = Annotated[Path, typer.Option("--output", help="New run directory.")]
_Steps = Annotated[int, typer.Option(min=1, max=100_000)]
_Batch = Annotated[int, typer.Option(min=1, max=1_048_576, help="Global positions per step.")]
_Rate = Annotated[float, typer.Option(min=1e-8, max=1.0)]
_MemoryBudget = Annotated[
    float | None,
    typer.Option(
        min=0.1,
        max=1024,
        help="Host memory budget for the run (default: 3/4 of available memory at start).",
    ),
]


@context_app.command("extend")
def context_extend(
    checkpoint: Annotated[Path, typer.Argument(help="Parent run or checkpoint.")],
    target_context: Annotated[int, typer.Option(min=64, max=1_048_576)],
    output: _Output,
    max_steps: _Steps = 50,
    method: Annotated[_Method, typer.Option()] = "yarn",
    seq_len: Annotated[int | None, typer.Option(min=1, help="Default: the target.")] = None,
    tokens_per_batch: Annotated[
        int | None, typer.Option(min=1, max=1_048_576, help="Default: one row of seq_len.")
    ] = None,
    learning_rate: _Rate = 3e-4,
    warmup_steps: Annotated[int, typer.Option(min=0)] = 0,
    min_document_pairs: Annotated[int | None, typer.Option(min=1)] = None,
    precision: Annotated[Literal["bf16", "fp32"], typer.Option()] = "bf16",
    strategy: Annotated[Literal["none", "ddp", "fsdp2"], typer.Option()] = "none",
    world_size: Annotated[int, typer.Option(min=1, max=8)] = 1,
    checkpoint_every: Annotated[int, typer.Option(min=1)] = 50,
    memory_budget_gib: _MemoryBudget = None,
) -> None:
    """Continue a checkpoint at a longer context with a RoPE extension method."""

    _cli().authorize_command("context extend")
    from epor.context.extension import extension_config
    from epor.training.kernel import resolve_model_config, trained_context

    directory, trainer, parent = _parent(checkpoint)
    length = seq_len or target_context
    try:
        model = resolve_model_config(
            extension_config(
                parent, target_context, trained_context=trained_context(trainer), method=method
            )
        )
    except Exception as exc:
        raise _failure(exc) from exc
    summary = _train(
        model,
        trainer,
        directory,
        output=output,
        max_steps=max_steps,
        seq_len=length,
        tokens_per_batch=tokens_per_batch or length,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        min_document_pairs=min_document_pairs,
        precision=precision,
        strategy=strategy,
        world_size=world_size,
        checkpoint_every=checkpoint_every,
        memory_budget_gib=memory_budget_gib,
    )
    _cli()._echo_json(
        {
            **summary,
            "context": {"configured": model.configured_max_context, "trained": length},
            "scope": "training at a length never raises the validated context",
        }
    )


@context_app.command("retrofit")
def context_retrofit(
    checkpoint: Annotated[Path, typer.Argument(help="Dense or local/global checkpoint.")],
    output: _Output,
    max_steps: _Steps = 50,
    select: Annotated[
        bool, typer.Option("--select/--warmup", help="Sparse top-k (after a warm-up stage).")
    ] = False,
    top_k: Annotated[int, typer.Option(min=1)] = 64,
    index_heads: Annotated[int, typer.Option(min=1, max=64)] = 2,
    index_dim: Annotated[int, typer.Option(min=1, max=512)] = 16,
    kl_weight: Annotated[float, typer.Option(min=0.0, max=100.0)] = 1.0,
    seq_len: Annotated[int | None, typer.Option(min=1, help="Default: trained context.")] = None,
    tokens_per_batch: Annotated[
        int | None, typer.Option(min=1, max=1_048_576, help="Default: one row of seq_len.")
    ] = None,
    learning_rate: _Rate = 3e-4,
    precision: Annotated[Literal["bf16", "fp32"], typer.Option()] = "bf16",
    memory_budget_gib: _MemoryBudget = None,
) -> None:
    """Continue a certified-style checkpoint with an experimental sparse-attention indexer."""

    _cli().authorize_command("context retrofit")
    from epor.context.extension import retrofit_config
    from epor.models.config import SparseAttention
    from epor.training.kernel import resolve_model_config, trained_context

    directory, trainer, parent = _parent(checkpoint)
    length = seq_len or trained_context(trainer)
    try:
        base = parent.model_copy(update={"sparse_attention": None})
        model = resolve_model_config(
            retrofit_config(
                base,
                SparseAttention(
                    index_heads=index_heads,
                    index_dim=index_dim,
                    top_k=top_k,
                    select=select,
                    kl_weight=kl_weight,
                ),
            )
        )
    except Exception as exc:
        raise _failure(exc) from exc
    summary = _train(
        model,
        trainer,
        directory,
        output=output,
        max_steps=max_steps,
        seq_len=length,
        tokens_per_batch=tokens_per_batch or length,
        learning_rate=learning_rate,
        warmup_steps=0,
        min_document_pairs=None,
        precision=precision,
        strategy="none",
        world_size=1,
        checkpoint_every=max_steps,
        memory_budget_gib=memory_budget_gib,
    )
    _cli()._echo_json(
        {
            **summary,
            "stage": "select" if select else "warmup",
            "scope": (
                "experimental retrofit; judged against its parent checkpoint, no benefit claimed"
            ),
        }
    )


@context_app.command("g6")
def context_g6(
    checkpoint: Annotated[Path, typer.Argument(help="Run or checkpoint to evaluate.")],
    length: Annotated[list[int], typer.Option("--length", help="Context length; repeatable.")],
    items_per_cell: Annotated[int, typer.Option(min=1, max=64)] = 2,
    seed: Annotated[int, typer.Option(min=0)] = 0,
    split: Annotated[Literal["train", "validation", "test"], typer.Option()] = "validation",
    batch_size: Annotated[int, typer.Option(min=1, max=64)] = 4,
) -> None:
    """Score a checkpoint on the generated G6 suite and store the report."""

    cli = _cli()
    cli.authorize_command("context g6")
    from epor.context.g6 import run_g6
    from epor.data.store import DataStore
    from epor.evaluation.decoder import load_kernel_scoring_model
    from epor.evaluation.report import store_report

    directory = _checkpoint(checkpoint)
    data_root = cli._data_root()
    try:
        model = load_kernel_scoring_model(directory, data_root, batch_size=batch_size)
    except Exception as exc:
        raise _failure(exc) from exc

    def work(cancel: Any) -> Any:
        report = run_g6(
            model,
            lengths=sorted(set(length)),
            items_per_cell=items_per_cell,
            seed=seed,
            perplexity=(data_root, model.tokenized_id, split),
            should_cancel=cancel,
        )
        store_report(DataStore(data_root), report)
        return report

    report = _run(work)
    from epor.context.summary import g6_view

    cli._echo_json(
        {
            **g6_view(report),
            "note": "certify a length with `epor context certify`; scoring alone validates nothing",
        }
    )


@context_app.command("profile")
def context_profile(
    checkpoint: Annotated[Path, typer.Argument(help="Run or checkpoint to profile.")],
    length: Annotated[list[int], typer.Option("--length", help="Context length; repeatable.")],
    quantize: Annotated[_Quantize, typer.Option()] = "none",
    decode_tokens: Annotated[int, typer.Option(min=0, max=4096)] = 16,
    memory_budget_gib: Annotated[float | None, typer.Option(min=0.1, max=1024)] = None,
    timeout: Annotated[float, typer.Option(min=1.0, max=24 * 3600.0)] = 300.0,
) -> None:
    """Measure TTFT, decode throughput, peak RSS, and OOM behaviour per length (host only)."""

    cli = _cli()
    cli.authorize_command("context profile")
    from epor.alignment.model import checkpoint_identity
    from epor.context.probe import profile_checkpoint, profile_report
    from epor.scaling.report import store_scaling_report

    directory = _checkpoint(checkpoint)
    budget = None if memory_budget_gib is None else int(memory_budget_gib * 1024**3)

    def work(cancel: Any) -> Any:
        def on_length(value: int) -> None:
            if cancel():
                raise InterruptedError("profile cancelled")
            typer.echo(f"profiling {value} tokens", err=True)

        config, records = profile_checkpoint(
            directory,
            sorted(set(length)),
            quantize=quantize,
            decode_tokens=decode_tokens,
            memory_budget_bytes=budget,
            timeout_seconds=timeout,
            on_length=on_length,
        )
        report = profile_report(config, checkpoint_identity(directory), records, quantize=quantize)
        store_scaling_report(cli._data_root(), report)
        return report

    report = _run(work)
    cli._echo_json(
        {
            "report_id": report.report_id,
            "lengths": [
                {**content, **measured}
                for content, measured in zip(
                    report.results["lengths"], report.measured["lengths"], strict=True
                )
            ],
            "claim_boundary": report.claim_boundary,
        }
    )


@context_app.command("feasibility")
def context_feasibility(
    recipe: Annotated[Path, typer.Argument(dir_okay=False, help="Model recipe.")],
    variant: Annotated[
        list[str], typer.Option("--variant", help="full, local, or mla; repeatable.")
    ],
    max_length: Annotated[int, typer.Option(min=128, max=1 << 20)] = 65_536,
    min_length: Annotated[int, typer.Option(min=64, max=1 << 20)] = 1024,
    vocab_size: Annotated[int | None, typer.Option(min=260)] = None,
    window: Annotated[int, typer.Option(min=1)] = 1024,
    ratio: Annotated[int, typer.Option(min=1)] = 1,
    mla_rank: Annotated[int, typer.Option(min=1)] = 32,
    mla_rope_dim: Annotated[int, typer.Option(min=2)] = 8,
    memory_budget_gib: Annotated[float | None, typer.Option(min=0.1, max=1024)] = None,
    timeout: Annotated[float, typer.Option(min=1.0, max=24 * 3600.0)] = 300.0,
) -> None:
    """Measure the largest length per attention variant whose train step and prefill fit."""

    cli = _cli()
    cli.authorize_command("context feasibility")
    from epor.context.probe import default_memory_budget, feasibility_report, feasible_maximum
    from epor.models.config import ModelConfig, load_model_config
    from epor.scaling.report import store_scaling_report

    path = cli._reference_path(recipe, label="recipe path", must_exist=True)[1]
    unknown = sorted(set(variant) - {"full", "local", "mla"})
    if unknown:
        raise _refuse(f"unknown variant(s): {', '.join(unknown)} (use full, local, or mla)")
    try:
        base = load_model_config(path).model_dump(mode="json")
        if vocab_size is not None:
            base["vocab_size"] = vocab_size
        overrides = {
            "full": {},
            "local": {"sliding_window": window, "local_layers_per_global": ratio},
            "mla": {
                "attention_variant": "mla",
                "mla_kv_rank": mla_rank,
                "mla_rope_dim": mla_rope_dim,
            },
        }
        variants = {
            name: ModelConfig.model_validate(
                {**base, **overrides[name], "configured_max_context": max_length}
            )
            for name in dict.fromkeys(variant)
        }
    except Exception as exc:
        raise _failure(exc) from exc
    lengths, value = [], min_length
    while value <= max_length:
        lengths.append(value)
        value *= 2
    budget = int(memory_budget_gib * 1024**3) if memory_budget_gib else default_memory_budget()

    def work(cancel: Any) -> Any:
        def on_point(name: str, mode: str, value: int) -> None:
            if cancel():
                raise InterruptedError("feasibility search cancelled")
            typer.echo(f"{name} {mode} {value}", err=True)

        results = feasible_maximum(
            variants,
            lengths,
            memory_budget_bytes=budget,
            timeout_seconds=timeout,
            on_point=on_point,
        )
        report = feasibility_report(results, budget=budget, timeout_seconds=timeout)
        store_scaling_report(cli._data_root(), report)
        return report

    report = _run(work)
    cli._echo_json(
        {
            "report_id": report.report_id,
            "feasible_maximum": {
                name: {
                    mode: report.results[name][mode]["max_ok_length"]
                    for mode in ("train", "inference")
                }
                for name in report.results
            },
            "warnings": report.warnings,
        }
    )


@context_app.command("sweep")
def context_sweep(
    recipe: Annotated[Path, typer.Argument(dir_okay=False, help="Model recipe.")],
    axis: Annotated[Literal["window", "ratio", "rope-base"], typer.Option()],
    value: Annotated[list[float], typer.Option("--value", help="Swept value; repeatable.")],
    tokenized_id: Annotated[str, typer.Option()],
    output: _Output,
    seq_len: Annotated[int, typer.Option(min=32)] = 256,
    tokens_per_batch: _Batch = 1024,
    max_steps: _Steps = 50,
    window: Annotated[int, typer.Option(min=1)] = 64,
    ratio: Annotated[int, typer.Option(min=1)] = 1,
    eval_length: Annotated[int, typer.Option(min=96)] = 256,
    items_per_cell: Annotated[int, typer.Option(min=1, max=16)] = 2,
    learning_rate: _Rate = 3e-3,
    vocab_from_build: Annotated[bool, typer.Option("--vocab-from-build")] = False,
) -> None:
    """Sweep one context axis against a full-attention control."""

    cli = _cli()
    cli.authorize_command("context sweep")
    from epor.context.sweep import ContextSweepPlan, run_context_sweep
    from epor.scaling.report import store_scaling_report
    from epor.training.runner import derive_kernel_model

    path = cli._reference_path(recipe, label="recipe path", must_exist=True)[1]
    _, destination = cli._reference_path(output, label="output path", must_exist=False)
    data_root = cli._data_root()
    try:
        base = derive_kernel_model(
            path, data_root=data_root, tokenized_id=tokenized_id, vocab_from_build=vocab_from_build
        )
        plan = ContextSweepPlan(
            axis=axis,
            values=tuple(value),
            window=window,
            ratio=ratio,
            tokenized_id=tokenized_id,
            seq_len=seq_len,
            tokens_per_batch=tokens_per_batch,
            max_steps=max_steps,
            learning_rate=learning_rate,
            eval_length=eval_length,
            items_per_cell=items_per_cell,
        )
    except Exception as exc:
        raise _failure(exc) from exc

    def work(cancel: Any) -> Any:
        report = run_context_sweep(
            base,
            plan,
            data_root=data_root,
            output_dir=destination,
            should_cancel=cancel,
            on_point=lambda name: typer.echo(f"point {name}", err=True),
        )
        store_scaling_report(data_root, report)
        return report

    report = _run(work)
    cli._echo_json({"report_id": report.report_id, **report.results, "warnings": report.warnings})


@context_app.command("project")
def context_project(
    target: Annotated[Path, typer.Argument(dir_okay=False, help="Target model config.")],
    inputs: Annotated[
        Path | None, typer.Option(help="Operator-written projection inputs (YAML).")
    ] = None,
    feasibility_report: Annotated[
        str | None, typer.Option(help="Feasibility report ID to cite as the local maximum.")
    ] = None,
) -> None:
    """Project KV cache, FLOPs, time, and cost for the 32K/128K/256K recipe (no approval)."""

    cli = _cli()
    cli.authorize_command("context project")
    from epor.context.projection import project_context
    from epor.models.config import load_model_config
    from epor.scaling.projection import ProjectionInputs, load_projection_inputs
    from epor.scaling.report import load_scaling_reports, store_scaling_report

    target_path = cli._reference_path(target, label="target config", must_exist=True)[1]
    try:
        config = load_model_config(target_path)
        projection_inputs = ProjectionInputs()
        if inputs is not None:
            projection_inputs = load_projection_inputs(
                cli._reference_path(inputs, label="inputs path", must_exist=True)[1]
            )
        feasibility = None
        if feasibility_report is not None:
            [report] = load_scaling_reports(cli._data_root(), feasibility_report)[-1:]
            if report.kind != "context-feasibility":
                raise ValueError("--feasibility-report must name a context-feasibility report")
            feasibility = {
                "report_id": report.report_id,
                "max_ok_length": {
                    name: {
                        mode: report.results[name][mode]["max_ok_length"]
                        for mode in ("train", "inference")
                    }
                    for name in report.results
                },
            }
        report = project_context(config, projection_inputs, feasibility=feasibility)
        store_scaling_report(cli._data_root(), report)
    except Exception as exc:
        raise _failure(exc) from exc
    cli._echo_json(
        {
            "report_id": report.report_id,
            "stages": [
                {
                    "length": row["length"],
                    "stage_rule": row["stage_rule"],
                    "kv_cache": row["kv_cache"],
                    "unknown_inputs": row["training"].get("unknown_inputs", []),
                }
                for row in report.results["stages"]
            ],
            "claim_boundary": report.claim_boundary,
        }
    )


@context_app.command("certify")
def context_certify(
    checkpoint: Annotated[Path, typer.Argument(help="Run or checkpoint to certify.")],
    length: Annotated[int, typer.Option(min=1)],
    g6_report: Annotated[str, typer.Option(help="G6 report of this checkpoint.")],
    baseline_report: Annotated[str, typer.Option(help="G6 report of the pre-extension parent.")],
    base_length: Annotated[int, typer.Option(min=1, help="Length compared for regression.")],
    profile_report: Annotated[str, typer.Option(help="Host profile report of this checkpoint.")],
) -> None:
    """Raise a checkpoint's validated context, only if its G6 evidence passes (CLI only)."""

    cli = _cli()
    cli.authorize_command("context certify")
    from epor.context.status import certify

    directory = _checkpoint(checkpoint)
    try:
        certificate = certify(
            directory,
            cli._data_root(),
            length=length,
            g6_report_id=g6_report,
            baseline_report_id=baseline_report,
            base_length=base_length,
            profile_report_id=profile_report,
        )
    except Exception as exc:
        raise _failure(exc) from exc
    cli._echo_json(certificate)


@context_app.command("status")
def context_status_command(
    checkpoint: Annotated[Path, typer.Argument(help="Run or checkpoint.")],
) -> None:
    """Show configured, trained, and validated context for one checkpoint."""

    cli = _cli()
    cli.authorize_command("context status")
    from epor.context.status import context_status

    directory = _checkpoint(checkpoint)
    try:
        cli._echo_json(context_status(directory, cli._data_root()))
    except Exception as exc:
        raise _failure(exc) from exc


@context_app.command("reports")
def context_reports() -> None:
    """List context runs, G6 reports, probes, sweeps, projections, and certificates."""

    cli = _cli()
    cli.authorize_command("context reports")
    from epor.context.summary import context_summary
    from epor.control.settings import ControlSettings

    settings = ControlSettings.from_environment()
    assert settings.artifact_root is not None
    try:
        cli._echo_json(
            context_summary(cli._data_root(), settings.artifact_root, settings.project_root)
        )
    except Exception as exc:
        raise _failure(exc) from exc


__all__ = ["context_app"]
