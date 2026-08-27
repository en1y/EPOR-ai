"""Command-line entry point for the EPOR research platform."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from epor import __version__
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
tokenizer_app = typer.Typer(help="Original tokenizer laboratory (post-v0.0.1).")
train_app = typer.Typer(help="Reference pretraining and later alignment workflows.")
eval_app = typer.Typer(help="Evaluate reference checkpoints and compare reports.")
export_app = typer.Typer(help="Export release artifacts (post-v0.0.1).")

app.add_typer(research_app, name="research")
app.add_typer(auth_app, name="auth")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")

console = Console()


class FutureCommand(StrEnum):
    tokenizer_train = "tokenizer train"
    tokenizer_evaluate = "tokenizer evaluate"
    train_sft = "train sft"
    train_preference = "train preference"
    train_grpo = "train grpo"
    eval_compare = "eval compare"
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


@tokenizer_app.command("train")
def tokenizer_train() -> None:
    _planned(FutureCommand.tokenizer_train, "v0.0.4")


@tokenizer_app.command("evaluate")
def tokenizer_evaluate() -> None:
    _planned(FutureCommand.tokenizer_evaluate, "v0.0.4")


@train_app.command("sft")
def train_sft() -> None:
    _planned(FutureCommand.train_sft, "v0.0.7")


@train_app.command("preference")
def train_preference() -> None:
    _planned(FutureCommand.train_preference, "v0.0.7")


@train_app.command("grpo")
def train_grpo() -> None:
    _planned(FutureCommand.train_grpo, "v0.0.7")


@eval_app.command("compare")
def eval_compare() -> None:
    _planned(FutureCommand.eval_compare, "v0.0.6")


@export_app.command("hf")
def export_hf() -> None:
    _planned(FutureCommand.export_hf, "v0.0.9")


@export_app.command("gguf")
def export_gguf() -> None:
    _planned(FutureCommand.export_gguf, "v0.0.9")


if __name__ == "__main__":
    app()
