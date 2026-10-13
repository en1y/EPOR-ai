"""Command-line entry point for the EPOR research platform."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from epor import __version__
from epor.lab_cli import register_evaluation_commands, scale_app
from epor.research.cli import research_app
from epor.system import probe_system

app = typer.Typer(
    name="epor",
    help="Original-weight language-model research and local orchestration.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
auth_app = typer.Typer(help="Manage the local owner credential and delegations.")
config_app = typer.Typer(help="Validate versioned model and run configuration.")
data_app = typer.Typer(help="Provenance-first data pipeline (post-v0.0.1).")
tokenizer_app = typer.Typer(
    help="Train, shard with, and measure original byte-level BPE tokenizers."
)
train_app = typer.Typer(help="Reference pretraining and later alignment workflows.")
eval_app = typer.Typer(help="Evaluate checkpoints on suites, scan contamination, compare reports.")
export_app = typer.Typer(help="Export release artifacts (post-v0.0.1).")

app.add_typer(research_app, name="research")
app.add_typer(auth_app, name="auth")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(scale_app, name="scale")
app.add_typer(export_app, name="export")

console = Console()


class FutureCommand(StrEnum):
    train_sft = "train sft"
    train_preference = "train preference"
    train_grpo = "train grpo"
    export_hf = "export hf"
    export_gguf = "export gguf"
    serve = "serve"


def _planned(command: FutureCommand, target: str) -> None:
    console.print(f"[yellow]epor {command.value} is intentionally gated until {target}.[/yellow]")
    raise typer.Exit(code=2)


def authorize(action_id: str) -> None:
    """Resolve ``action_id`` against the safety covenant before executing it.

    The CLI runs work that never passes through the control plane, so it
    carries the same gate.  An action the covenant does not permit exits
    non-zero having done nothing, and every resolution — permitted or not — is
    written to durable safety truth, because a gate that records only its
    refusals cannot be audited for what it let through.

    The shell on this machine is the owner's own shell, so the actor is the
    owner.  Nothing here defends against an already compromised local account;
    that limitation is documented rather than papered over.
    """

    from epor.actions import resolve_action
    from epor.control.authority import LOCAL_CLI, AuthorityStore
    from epor.control.settings import ControlSettings
    from epor.safety import load_covenant

    settings = ControlSettings.from_environment()
    assert settings.safety_root is not None
    resolution = resolve_action(action_id, load_covenant(), actor=LOCAL_CLI)
    AuthorityStore(settings.safety_root).record_decision(
        actor=LOCAL_CLI,
        action_id=action_id,
        surface="cli",
        outcome=resolution.outcome,
        binding_priority=resolution.binding_priority,
        reasons=resolution.reasons,
        covenant_sha256=resolution.covenant_sha256,
    )
    if resolution.outcome == "allow":
        return
    console.print(
        f"[red]refused by the EPOR safety covenant[/red] "
        f"({resolution.outcome}, covenant v{resolution.covenant_version})"
    )
    for reason in resolution.reasons:
        console.print(f"  - {reason}")
    raise typer.Exit(code=3)


def authorize_command(command: str) -> None:
    """Gate one CLI command through the shared command-to-action registry.

    An unregistered command is passed through under its own name, which has no
    declaration and therefore escalates.  Adding an executing command without
    declaring its covenant standing fails closed rather than running.
    """

    from epor.actions import CLI_ACTIONS

    authorize(CLI_ACTIONS.get(command, command))


def _reference_path(
    supplied: Path,
    *,
    label: str,
    must_exist: bool,
) -> tuple[Path, Path]:
    """Resolve a direct reference-command path beneath ``EPOR_PROJECT_ROOT``."""

    from epor.control.security import PathOutsideRootError, contained_path
    from epor.control.settings import project_root_from_environment

    project_root = project_root_from_environment()
    try:
        resolved = contained_path(project_root, supplied, must_exist=must_exist)
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(resolved)
    except (OSError, PathOutsideRootError) as exc:
        console.print(f"[red]reference command refused[/red]: {label} {exc}")
        raise typer.Exit(code=2) from exc
    return project_root, resolved


@app.command()
def version() -> None:
    """Print the repository interface version."""

    typer.echo(__version__)


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
    root: Annotated[
        Path,
        typer.Option(help="Project root used to resolve the source revision."),
    ] = Path("."),
) -> None:
    """Inspect local capabilities without network or accelerator side effects."""

    authorize_command("doctor")
    capabilities = probe_system(root)
    payload = capabilities.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title=f"EPOR AI {__version__} — local capabilities")
    table.add_column("Capability", style="cyan")
    table.add_column("Value")
    for name, value in payload.items():
        table.add_row(name.replace("_", " "), str(value))
    console.print(table)
    if capabilities.python.split(".")[:2] != ["3", "11"]:
        console.print("[yellow]Use `uv run` to select the pinned Python 3.11 environment.[/yellow]")


@config_app.command("validate")
def config_validate(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the canonical configuration as JSON."),
    ] = False,
    count_parameters: Annotated[
        bool,
        typer.Option(
            "--count-parameters/--no-count-parameters",
            help="Use PyTorch's meta device to measure parameters when available.",
        ),
    ] = True,
) -> None:
    """Strictly validate a model recipe without allocating its weights."""

    authorize_command("config validate")
    from epor.models.config import validate_config

    config = validate_config(path)
    payload = config.model_dump(mode="json")
    if count_parameters:
        try:
            from epor.models.reference import parameter_report

            payload["parameter_report"] = parameter_report(config).model_dump(mode="json")
        except ImportError:
            payload["parameter_report"] = None
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    console.print(
        f"[green]valid[/green] {config.display_name} ({config.slug}) config_sha256={config.sha256}"
    )
    report = payload.get("parameter_report")
    if isinstance(report, dict):
        console.print(f"measured_total_parameters={int(report['measured_total_parameters']):,}")


@train_app.command("pretrain")
def train_pretrain(
    config: Annotated[
        Path,
        typer.Argument(dir_okay=False),
    ],
    output: Annotated[Path, typer.Option("--output", help="Run output directory.")],
    max_steps: Annotated[int, typer.Option(min=1, max=1_000)] = 2,
    corpus: Annotated[
        Path,
        typer.Option(dir_okay=False),
    ] = Path("fixtures/tiny_corpus.txt"),
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    sequence_length: Annotated[int | None, typer.Option(min=1)] = None,
    model_seed: Annotated[int | None, typer.Option(min=0)] = None,
    data_seed: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    """Run the bounded, single-process CPU reference pretrainer."""

    authorize_command("train pretrain")
    from epor.training import pretrain

    project_root, config = _reference_path(config, label="config path", must_exist=True)
    _, corpus = _reference_path(corpus, label="corpus path", must_exist=True)
    _, output = _reference_path(output, label="output path", must_exist=False)
    result = pretrain(
        config,
        output,
        max_steps=max_steps,
        corpus_path=corpus,
        batch_size=batch_size,
        sequence_length=sequence_length,
        model_seed=model_seed,
        data_seed=data_seed,
        project_root=project_root,
    )
    typer.echo(result.model_dump_json(indent=2))


@train_app.command("resume")
def train_resume(
    config: Annotated[
        Path,
        typer.Argument(dir_okay=False),
    ],
    checkpoint: Annotated[
        Path,
        typer.Argument(dir_okay=False),
    ],
    output: Annotated[Path, typer.Option("--output", help="Run output directory.")],
    max_steps: Annotated[
        int,
        typer.Option(min=1, max=1_000, help="Absolute final optimizer step."),
    ],
    corpus: Annotated[
        Path,
        typer.Option(dir_okay=False),
    ] = Path("fixtures/tiny_corpus.txt"),
) -> None:
    """Resume an exact deterministic trajectory from a trusted EPOR checkpoint."""

    authorize_command("train resume")
    from epor.training import pretrain

    project_root, config = _reference_path(config, label="config path", must_exist=True)
    _, checkpoint = _reference_path(checkpoint, label="checkpoint path", must_exist=True)
    _, corpus = _reference_path(corpus, label="corpus path", must_exist=True)
    _, output = _reference_path(output, label="output path", must_exist=False)
    result = pretrain(
        config,
        output,
        max_steps=max_steps,
        resume_from=checkpoint,
        corpus_path=corpus,
        project_root=project_root,
    )
    typer.echo(result.model_dump_json(indent=2))


@eval_app.command("run")
def eval_run(
    checkpoint: Annotated[
        Path,
        typer.Argument(dir_okay=False),
    ],
    corpus: Annotated[
        Path,
        typer.Option(dir_okay=False),
    ] = Path("fixtures/tiny_corpus.txt"),
    max_batches: Annotated[int | None, typer.Option(min=1, max=10_000)] = None,
) -> None:
    """Evaluate a tiny reference checkpoint on deterministic fixture windows."""

    authorize_command("eval run")
    from epor.training import evaluate

    _, checkpoint = _reference_path(checkpoint, label="checkpoint path", must_exist=True)
    _, corpus = _reference_path(corpus, label="corpus path", must_exist=True)
    result = evaluate(checkpoint, corpus_path=corpus, max_batches=max_batches)
    typer.echo(result.model_dump_json(indent=2))


@app.command("generate")
def generate_command(
    checkpoint: Annotated[
        Path,
        typer.Argument(dir_okay=False),
    ],
    prompt: Annotated[str, typer.Argument(help="Prompt encoded with the debug tokenizer.")],
    max_new_tokens: Annotated[int, typer.Option(min=0, max=4096)] = 32,
) -> None:
    """Greedily generate from a tiny reference checkpoint."""

    authorize_command("generate")
    from epor.training import generate

    _, checkpoint = _reference_path(checkpoint, label="checkpoint path", must_exist=True)
    result = generate(checkpoint, prompt, max_new_tokens=max_new_tokens)
    typer.echo(result.model_dump_json(indent=2))


@app.command("api")
def api_command(
    host: Annotated[str, typer.Option(help="Numeric loopback bind address only.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8742,
) -> None:
    """Run the loopback-only control API."""

    authorize_command("api")
    import uvicorn

    from epor.control.api import create_app
    from epor.control.settings import ControlSettings

    settings = replace(ControlSettings.from_environment(), host=host, port=port)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        access_log=False,
    )


@app.command("worker")
def worker_command(
    once: Annotated[
        bool,
        typer.Option("--once", help="Process at most one queued job and exit."),
    ] = False,
) -> None:
    """Run the separate allowlisted local worker."""

    authorize_command("worker")
    from epor.control.worker import JobWorker

    with JobWorker() as worker:
        if once:
            job = worker.run_once()
            typer.echo("idle" if job is None else f"{job.id} {job.status.value}")
            return
        worker.run_forever()


@app.command("ui")
def ui_command() -> None:
    """Start the React development console on its fixed loopback origin."""

    authorize_command("ui")
    npm = shutil.which("npm")
    ui_root = Path(__file__).resolve().parents[2] / "ui"
    if npm is None:
        console.print("[red]npm is required to run the local console.[/red]")
        raise typer.Exit(code=1)
    if not (ui_root / "package.json").is_file():
        console.print("[red]The UI package is not present in this checkout.[/red]")
        raise typer.Exit(code=1)
    completed = subprocess.run(
        [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
        cwd=ui_root,
        check=False,
    )
    raise typer.Exit(code=completed.returncode)


@auth_app.command("bootstrap")
def auth_bootstrap() -> None:
    """Mint the single local owner credential. First run only.

    Deliberately CLI-only: the first credential can only legitimately come from
    the shell of the machine owner, and an HTTP endpoint that minted it would
    be reachable by anything able to open a socket on loopback.
    """

    authorize_command("auth bootstrap")
    from epor.control.authority import AuthorityStore
    from epor.control.errors import ControlError
    from epor.control.settings import ControlSettings

    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.safety_root is not None
    store = AuthorityStore(settings.safety_root)
    try:
        issued = store.bootstrap_owner()
    except ControlError as exc:
        console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(code=1) from exc
    console.print("[green]owner credential created[/green]")
    console.print(f"  stored at: {store.owner_token_path}")
    console.print("  Shown once here; only a SHA-256 digest is kept in safety truth.")
    typer.echo(issued.token)


@auth_app.command("status")
def auth_status() -> None:
    """List local principals. Prints no credential value.

    Read-only and outside the execution gate, like ``version``: it starts
    nothing and changes nothing.
    """

    from epor.control.authority import AuthorityStore
    from epor.control.settings import ControlSettings

    settings = ControlSettings.from_environment()
    assert settings.safety_root is not None
    store = AuthorityStore(settings.safety_root)
    if store.owner() is None:
        console.print("[yellow]no owner credential; run `epor auth bootstrap`[/yellow]")
        raise typer.Exit(code=1)
    table = Table(title="EPOR local authority")
    table.add_column("Principal", style="cyan")
    table.add_column("Role")
    table.add_column("Scopes")
    table.add_column("Status")
    for principal in store.principals():
        scopes = (
            "every declared action"
            if principal.role.value == "owner"
            else ", ".join(sorted(principal.scopes)) or "-"
        )
        table.add_row(
            principal.label,
            principal.role.value,
            scopes,
            "active" if principal.active() else "inactive",
        )
    console.print(table)


@app.command("serve")
def serve_command() -> None:
    """Reserve the future provider-neutral local inference surface."""

    _planned(FutureCommand.serve, "v0.0.9")


@data_app.command("ingest")
def data_ingest(
    registration: Annotated[Path, typer.Argument(dir_okay=False)],
    inputs: Annotated[list[Path], typer.Argument(dir_okay=False)],
    max_input_bytes: Annotated[
        int,
        typer.Option(min=1, max=256 * 1024 * 1024),
    ] = 256 * 1024 * 1024,
) -> None:
    """Register and stream reviewed local files into immutable data layers."""

    authorize_command("data ingest")
    from epor.control.settings import ControlSettings
    from epor.data.service import DataEngine, load_registration

    _, registration_path = _reference_path(
        registration,
        label="data registration path",
        must_exist=True,
    )
    resolved_inputs = [
        _reference_path(path, label="data input path", must_exist=True)[1] for path in inputs
    ]
    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.data_root is not None
    result = DataEngine(settings.data_root).ingest(
        load_registration(registration_path),
        resolved_inputs,
        relative_names=[
            path.relative_to(settings.project_root).as_posix() for path in resolved_inputs
        ],
        max_input_bytes=max_input_bytes,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


@data_app.command("build")
def data_build(
    dataset_id: Annotated[str, typer.Argument()],
    source_ids: Annotated[
        list[str] | None,
        typer.Option("--source-id", help="Include a registered source; repeat as needed."),
    ] = None,
    train_percent: Annotated[int, typer.Option(min=1, max=100)] = 98,
    validation_percent: Annotated[int, typer.Option(min=0, max=100)] = 1,
    test_percent: Annotated[int, typer.Option(min=0, max=100)] = 1,
    near_duplicate_distance: Annotated[int, typer.Option(min=0, max=8)] = 3,
) -> None:
    """Build deterministic deduplicated splits and an immutable dataset card."""

    authorize_command("data build")
    from epor.control.settings import ControlSettings
    from epor.data.models import BuildRequest, SplitRatios
    from epor.data.service import DataEngine

    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.data_root is not None
    request = BuildRequest(
        dataset_id=dataset_id,
        source_ids=source_ids,
        split_ratios=SplitRatios(
            train=train_percent,
            validation=validation_percent,
            test=test_percent,
        ),
        near_duplicate_hamming_distance=near_duplicate_distance,
        intended_uses=["offline original-weight language-model training"],
        prohibited_uses=["identity inference", "unreviewed redistribution"],
    )
    manifest, manifest_path = DataEngine(settings.data_root).build(request)
    typer.echo(
        json.dumps(
            {
                "dataset_id": manifest.dataset_id,
                "build_id": manifest.build_id,
                "manifest_path": str(manifest_path),
                "counts": manifest.counts,
                "split_counts": manifest.split_counts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@data_app.command("audit")
def data_audit() -> None:
    """Verify content hashes and summarize the local provenance store."""

    authorize_command("data audit")
    from epor.control.settings import ControlSettings
    from epor.data.service import DataEngine

    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.data_root is not None
    audit = DataEngine(settings.data_root).audit()
    typer.echo(audit.model_dump_json(indent=2))
    if audit.integrity_errors:
        raise typer.Exit(code=1)


@data_app.command("remove")
def data_remove(
    target_kind: Annotated[
        Literal["source_id", "document_id", "raw_sha256"],
        typer.Argument(help="One of: source_id, document_id, raw_sha256."),
    ],
    target: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option(prompt=True)],
    requested_at: Annotated[datetime | None, typer.Option()] = None,
    removal_contact: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create a removal tombstone applied by every future dataset build."""

    authorize_command("data remove")
    from epor.control.settings import ControlSettings
    from epor.data.service import DataEngine

    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.data_root is not None
    tombstone = DataEngine(settings.data_root).remove(
        target_kind=target_kind,
        target=target,
        reason=reason,
        requested_by="local-cli-owner",
        requested_at=requested_at,
        removal_contact=removal_contact,
    )
    typer.echo(tombstone.model_dump_json(indent=2))


_TokenizerSplit = Literal["train", "validation", "test"]


def _data_root() -> Path:
    from epor.control.settings import ControlSettings

    settings = ControlSettings.from_environment()
    settings.prepare_directories()
    assert settings.data_root is not None
    return settings.data_root


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _refuse_tokenizer_command(exc: Exception) -> typer.Exit:
    console.print(f"[red]tokenizer command refused[/red]: {exc}")
    return typer.Exit(code=2)


def _tokenizer_errors() -> tuple[type[Exception], ...]:
    from epor.data.store import DataStoreError
    from epor.tokenizer import CorpusError, TokenizedBuildError, TokenizerArtifactError

    # ``ValueError`` covers bounded settings and invalid identifiers, including
    # pydantic validation of stored manifests.
    return (CorpusError, DataStoreError, TokenizedBuildError, TokenizerArtifactError, ValueError)


@tokenizer_app.command("train")
def tokenizer_train(
    dataset_id: Annotated[str, typer.Argument()],
    build_id: Annotated[str, typer.Argument()],
    vocab_size: Annotated[
        int,
        typer.Option(min=520, max=65_536, help="Declared vocabulary size (IDs incl. reserved)."),
    ] = 65_536,
    min_frequency: Annotated[int, typer.Option(min=1, max=1_000_000)] = 2,
) -> None:
    """Train a deterministic byte-level BPE from one build's train split only."""

    authorize_command("tokenizer train")
    from epor.tokenizer import describe_tokenizer, train_tokenizer

    try:
        trained = train_tokenizer(
            _data_root(),
            dataset_id,
            build_id,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
    except _tokenizer_errors() as exc:
        raise _refuse_tokenizer_command(exc) from exc
    _echo_json(
        {
            **describe_tokenizer(trained.tokenizer_id, trained.manifest),
            "directory": str(trained.directory),
        }
    )


@tokenizer_app.command("shard")
def tokenizer_shard(
    dataset_id: Annotated[str, typer.Argument()],
    build_id: Annotated[str, typer.Argument()],
    tokenizer_id: Annotated[str, typer.Argument()],
    max_shard_tokens: Annotated[
        int,
        typer.Option(min=4, max=1 << 24, help="Upper bound on tokens per uint16 shard."),
    ] = 1 << 24,
) -> None:
    """Encode every split of a build into content-addressed token shards."""

    authorize_command("tokenizer shard")
    from epor.tokenizer import build_token_shards, describe_tokenized

    try:
        result = build_token_shards(
            _data_root(),
            dataset_id,
            build_id,
            tokenizer_id,
            max_shard_tokens=max_shard_tokens,
        )
    except _tokenizer_errors() as exc:
        raise _refuse_tokenizer_command(exc) from exc
    _echo_json(
        {
            **describe_tokenized(result.tokenized_id, result.manifest),
            "created": result.created,
            "directory": str(result.directory),
        }
    )


@tokenizer_app.command("evaluate")
def tokenizer_evaluate(
    tokenizer_id: Annotated[str, typer.Argument()],
    dataset_id: Annotated[str, typer.Argument()],
    build_id: Annotated[str, typer.Argument()],
    split: Annotated[
        _TokenizerSplit,
        typer.Option(help="One of: validation, test, train (train is flagged in-sample)."),
    ] = "validation",
) -> None:
    """Measure fertility and exact byte round-trip on one build split.

    Exits 1 when any document fails to round-trip, so the gate cannot pass
    silently; the report is still stored for review.
    """

    authorize_command("tokenizer evaluate")
    from epor.tokenizer import describe_report, evaluate_tokenizer

    try:
        report = evaluate_tokenizer(_data_root(), tokenizer_id, dataset_id, build_id, split=split)
    except _tokenizer_errors() as exc:
        raise _refuse_tokenizer_command(exc) from exc
    _echo_json(describe_report(report))
    if report.in_sample:
        typer.echo(
            "warning: in-sample measurement on the tokenizer's own training split; "
            "it does not describe held-out fertility",
            err=True,
        )
    if report.round_trip_failures:
        typer.echo(
            f"round-trip gate failed: {report.round_trip_failures} document(s) did not decode "
            "to their exact source bytes",
            err=True,
        )
        raise typer.Exit(code=1)


_KERNEL_SPLIT = Literal["train", "validation", "test"]
_KernelPrecision = Literal["bf16", "fp32"]
_KernelStrategy = Literal["none", "ddp", "fsdp2"]
_KernelValidation = Literal["validation", "test", "none"]
_KERNEL_INPUT_ERRORS = frozenset(
    {
        "CheckpointError",
        "CorpusError",
        "DataStoreError",
        "FileExistsError",
        "FileNotFoundError",
        "NotADirectoryError",
        "TokenizedBuildError",
        "TokenizerArtifactError",
        "ValidationError",
        "ValueError",
    }
)
_KERNEL_CANCELLED_EXIT = 130


def _contained_directory(supplied: Path, *, label: str) -> Path:
    """Resolve an existing run or checkpoint directory beneath ``EPOR_PROJECT_ROOT``."""

    from epor.control.security import PathOutsideRootError, contained_path
    from epor.control.settings import project_root_from_environment

    try:
        resolved = contained_path(project_root_from_environment(), supplied, must_exist=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
    except (OSError, PathOutsideRootError) as exc:
        console.print(f"[red]training command refused[/red]: {label} {exc}")
        raise typer.Exit(code=2) from exc
    return resolved


class _StopRequests:
    """First SIGINT/SIGTERM asks for a checkpointed stop; a second aborts at once."""

    def __init__(self, request: Callable[[], None]) -> None:
        self._request = request
        self._count = 0
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: object) -> None:
        self._count += 1
        if self._count > 1:
            raise KeyboardInterrupt
        typer.echo(
            "stop requested: checkpointing the last completed step (repeat to abort at once)",
            err=True,
        )
        self._request()

    def __enter__(self) -> _StopRequests:
        import signal

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_args: object) -> None:
        import signal

        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def _kernel_failure(exc: BaseException) -> typer.Exit:
    from epor.training.runner import KernelRunError, is_cancellation

    if is_cancellation(exc):
        typer.echo(f"training stopped: {exc}", err=True)
        return typer.Exit(code=_KERNEL_CANCELLED_EXIT)
    name = exc.error_type if isinstance(exc, KernelRunError) else type(exc).__name__
    message = exc.message if isinstance(exc, KernelRunError) else str(exc)
    if name in _KERNEL_INPUT_ERRORS:
        console.print(f"[red]training command refused[/red]: {message}")
        return typer.Exit(code=2)
    console.print(f"[red]training failed[/red] ({name}): {message}")
    return typer.Exit(code=1)


def _run_kernel_command(
    model: Any,
    output: Path,
    settings: Any,
    *,
    max_steps: int,
    resume_from: Path | None,
    strategy: str,
    world_size: int,
    launch_timeout: float,
    project_root: Path,
) -> dict[str, Any]:
    import threading

    from epor.training.runner import run_kernel

    if strategy == "none":
        stop: Any = threading.Event()
    else:
        import torch.multiprocessing as mp

        stop = mp.get_context("spawn").Event()

    def on_step(step: int, total: int, loss: float) -> None:
        typer.echo(f"step {step}/{total} loss {loss:.6f}", err=True)

    try:
        with _StopRequests(stop.set):
            return run_kernel(
                model,
                _data_root(),
                output,
                settings,
                max_steps=max_steps,
                resume_from=resume_from,
                strategy=strategy,  # pyright: ignore[reportArgumentType]
                world_size=world_size,
                project_root=project_root,
                should_cancel=stop.is_set,
                on_step=on_step,
                launch_timeout_seconds=launch_timeout,
                stop_event=None if strategy == "none" else stop,
            )
    except Exception as exc:
        raise _kernel_failure(exc) from exc


@train_app.command("kernel")
def train_kernel(
    recipe: Annotated[Path, typer.Argument(dir_okay=False, help="Model recipe (YAML/JSON).")],
    tokenized_id: Annotated[
        str, typer.Option("--tokenized-id", help="Verified tokenized build to train on.")
    ],
    output: Annotated[Path, typer.Option("--output", help="New run directory.")],
    max_steps: Annotated[int, typer.Option(min=1, max=100_000, help="Optimizer steps.")],
    vocab_from_build: Annotated[
        bool,
        typer.Option(
            "--vocab-from-build",
            help="Train a derived config with the build's vocabulary (the recipe is unchanged).",
        ),
    ] = False,
    qk_norm: Annotated[
        bool | None,
        typer.Option("--qk-norm/--no-qk-norm", help="Override the recipe's QK-norm flag."),
    ] = None,
    mtp_depth: Annotated[
        int | None, typer.Option(min=0, max=4, help="Override the recipe's MTP depth.")
    ] = None,
    seq_len: Annotated[int, typer.Option(min=1, max=8_192)] = 128,
    tokens_per_batch: Annotated[
        int, typer.Option(min=1, max=1_048_576, help="Global positions per optimizer step.")
    ] = 4_096,
    micro_batch_size: Annotated[
        int | None, typer.Option(min=1, help="Rows per micro-batch per rank (accumulation).")
    ] = None,
    precision: Annotated[_KernelPrecision, typer.Option()] = "bf16",
    strategy: Annotated[_KernelStrategy, typer.Option()] = "none",
    world_size: Annotated[int, typer.Option(min=1, max=8, help="Local ranks for ddp/fsdp2.")] = 1,
    learning_rate: Annotated[float, typer.Option(min=1e-8, max=1.0)] = 3e-4,
    min_lr_ratio: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    warmup_steps: Annotated[int, typer.Option(min=0, max=100_000)] = 0,
    total_steps: Annotated[
        int | None,
        typer.Option(min=1, max=100_000, help="Schedule length (default: --max-steps)."),
    ] = None,
    beta1: Annotated[float, typer.Option(min=0.0, max=0.999999)] = 0.9,
    beta2: Annotated[float, typer.Option(min=0.0, max=0.999999)] = 0.95,
    weight_decay: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.1,
    gradient_clip: Annotated[float, typer.Option(min=1e-6, max=1_000.0)] = 1.0,
    data_seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 23,
    model_seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 17,
    checkpoint_every: Annotated[int, typer.Option(min=1, max=100_000)] = 50,
    keep_checkpoints: Annotated[int, typer.Option(min=1, max=100)] = 3,
    activation_checkpointing: Annotated[bool, typer.Option()] = False,
    validation_split: Annotated[_KernelValidation, typer.Option()] = "validation",
    eval_max_steps: Annotated[int | None, typer.Option(min=1, max=100_000)] = None,
    launch_timeout: Annotated[
        float, typer.Option(min=10.0, max=7 * 24 * 3600.0, help="Distributed wall-clock limit (s).")
    ] = 6 * 3600.0,
) -> None:
    """Train the kernel on packed token shards (exit 2 input, 3 covenant, 130 stopped)."""

    authorize_command("train kernel")
    from epor.training import OptimizerSettings, PackedTrainingSettings
    from epor.training.runner import derive_kernel_model

    project_root, recipe = _reference_path(recipe, label="recipe path", must_exist=True)
    _, output = _reference_path(output, label="output path", must_exist=False)
    if (strategy == "none") != (world_size == 1):
        console.print(
            "[red]training command refused[/red]: strategy 'none' runs one process and "
            "ddp/fsdp2 need --world-size of at least 2"
        )
        raise typer.Exit(code=2)
    try:
        model = derive_kernel_model(
            recipe,
            data_root=_data_root(),
            tokenized_id=tokenized_id,
            vocab_from_build=vocab_from_build,
            qk_norm=qk_norm,
            mtp_depth=mtp_depth,
        )
        settings = PackedTrainingSettings(
            tokenized_id=tokenized_id,
            seq_len=seq_len,
            tokens_per_batch=tokens_per_batch,
            micro_batch_size=micro_batch_size,
            data_seed=data_seed,
            model_seed=model_seed,
            precision=precision,
            activation_checkpointing=activation_checkpointing,
            checkpoint_every=checkpoint_every,
            keep_checkpoints=keep_checkpoints,
            validation_split=None if validation_split == "none" else validation_split,
            eval_max_steps=eval_max_steps,
            optimizer=OptimizerSettings(
                learning_rate=learning_rate,
                min_lr_ratio=min_lr_ratio,
                warmup_steps=warmup_steps,
                total_steps=total_steps or max_steps,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                gradient_clip=gradient_clip,
            ),
        )
    except Exception as exc:
        raise _kernel_failure(exc) from exc
    summary = _run_kernel_command(
        model,
        output,
        settings,
        max_steps=max_steps,
        resume_from=None,
        strategy=strategy,
        world_size=world_size,
        launch_timeout=launch_timeout,
        project_root=project_root,
    )
    _echo_json({**summary, "recipe": str(recipe.relative_to(project_root))})


@train_app.command("kernel-resume")
def train_kernel_resume(
    checkpoint: Annotated[
        Path, typer.Argument(help="Run directory, its checkpoints/ directory, or a step-*.")
    ],
    max_steps: Annotated[
        int, typer.Option(min=1, max=100_000, help="Absolute final optimizer step.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Run directory (default: the checkpoint's own run)."),
    ] = None,
    strategy: Annotated[_KernelStrategy, typer.Option()] = "none",
    world_size: Annotated[int, typer.Option(min=1, max=8)] = 1,
    micro_batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    checkpoint_every: Annotated[int | None, typer.Option(min=1, max=100_000)] = None,
    keep_checkpoints: Annotated[int | None, typer.Option(min=1, max=100)] = None,
    activation_checkpointing: Annotated[
        bool | None, typer.Option("--activation-checkpointing/--no-activation-checkpointing")
    ] = None,
    eval_max_steps: Annotated[int | None, typer.Option(min=1, max=100_000)] = None,
    launch_timeout: Annotated[float, typer.Option(min=10.0, max=7 * 24 * 3600.0)] = 6 * 3600.0,
) -> None:
    """Continue a kernel run exactly; only runtime settings and topology may change."""

    authorize_command("train kernel-resume")
    from epor.control.settings import project_root_from_environment
    from epor.training.runner import resume_inputs

    project_root = project_root_from_environment()
    checkpoint = _contained_directory(checkpoint, label="checkpoint path")
    if (strategy == "none") != (world_size == 1):
        console.print(
            "[red]training command refused[/red]: strategy 'none' runs one process and "
            "ddp/fsdp2 need --world-size of at least 2"
        )
        raise typer.Exit(code=2)
    try:
        model, settings, run_dir, step = resume_inputs(
            checkpoint,
            {
                "micro_batch_size": micro_batch_size,
                "checkpoint_every": checkpoint_every,
                "keep_checkpoints": keep_checkpoints,
                "activation_checkpointing": activation_checkpointing,
                "eval_max_steps": eval_max_steps,
            },
        )
    except Exception as exc:
        raise _kernel_failure(exc) from exc
    if output is None:
        destination = _contained_directory(run_dir, label="run directory")
    else:
        _, destination = _reference_path(output, label="output path", must_exist=False)
    typer.echo(f"resuming from step {step} toward step {max_steps}", err=True)
    summary = _run_kernel_command(
        model,
        destination,
        settings,
        max_steps=max_steps,
        resume_from=checkpoint,
        strategy=strategy,
        world_size=world_size,
        launch_timeout=launch_timeout,
        project_root=project_root,
    )
    _echo_json(summary)


@eval_app.command("kernel")
def eval_kernel(
    checkpoint: Annotated[
        Path, typer.Argument(help="Run directory, its checkpoints/ directory, or a step-*.")
    ],
    split: Annotated[_KERNEL_SPLIT, typer.Option()] = "validation",
    max_steps: Annotated[
        int | None, typer.Option(min=1, max=100_000, help="Cap on evaluation batches.")
    ] = None,
) -> None:
    """Measure a kernel checkpoint's main loss, perplexity, and per-depth MTP loss."""

    authorize_command("eval kernel")
    from epor.training import evaluate_packed_checkpoint
    from epor.training.run_checkpoint import resolve_checkpoint

    checkpoint = _contained_directory(checkpoint, label="checkpoint path")
    try:
        directory = resolve_checkpoint(checkpoint)
        result = evaluate_packed_checkpoint(
            directory,
            _data_root(),
            split=split,
            max_steps=max_steps,
            report_dir=directory.parent.parent / "evaluations",
        )
    except Exception as exc:
        raise _kernel_failure(exc) from exc
    payload = result.model_dump(mode="json")
    payload["scope"] = (
        "proxy-scale local measurement of one checkpoint on one split; not a validated result"
    )
    _echo_json(payload)
    if split == "train":
        typer.echo("warning: in-sample measurement on the training split", err=True)


@train_app.command("sft")
def train_sft() -> None:
    _planned(FutureCommand.train_sft, "v0.0.7")


@train_app.command("preference")
def train_preference() -> None:
    _planned(FutureCommand.train_preference, "v0.0.7")


@train_app.command("grpo")
def train_grpo() -> None:
    _planned(FutureCommand.train_grpo, "v0.0.7")


@export_app.command("hf")
def export_hf() -> None:
    _planned(FutureCommand.export_hf, "v0.0.9")


@export_app.command("gguf")
def export_gguf() -> None:
    _planned(FutureCommand.export_gguf, "v0.0.9")


register_evaluation_commands(eval_app)


if __name__ == "__main__":
    app()
