"""Command-line entry point for the EPOR research platform."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

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
config_app = typer.Typer(help="Validate versioned model and run configuration.")
data_app = typer.Typer(help="Provenance-first data pipeline (post-v0.0.1).")
tokenizer_app = typer.Typer(help="Original tokenizer laboratory (post-v0.0.1).")
train_app = typer.Typer(help="Reference pretraining and later alignment workflows.")
eval_app = typer.Typer(help="Evaluate reference checkpoints and compare reports.")
export_app = typer.Typer(help="Export release artifacts (post-v0.0.1).")

app.add_typer(research_app, name="research")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")

console = Console()


class FutureCommand(StrEnum):
    data_ingest = "data ingest"
    data_build = "data build"
    data_audit = "data audit"
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
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Run output directory.")],
    max_steps: Annotated[int, typer.Option(min=1, max=100_000)] = 2,
    corpus: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("fixtures/tiny_corpus.txt"),
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    sequence_length: Annotated[int | None, typer.Option(min=1)] = None,
    model_seed: Annotated[int | None, typer.Option(min=0)] = None,
    data_seed: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    """Run the bounded, single-process CPU reference pretrainer."""

    from epor.training import pretrain

    result = pretrain(
        config,
        output,
        max_steps=max_steps,
        corpus_path=corpus,
        batch_size=batch_size,
        sequence_length=sequence_length,
        model_seed=model_seed,
        data_seed=data_seed,
    )
    typer.echo(result.model_dump_json(indent=2))


@train_app.command("resume")
def train_resume(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    checkpoint: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", help="Run output directory.")],
    max_steps: Annotated[
        int,
        typer.Option(min=1, help="Absolute final optimizer step."),
    ],
    corpus: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("fixtures/tiny_corpus.txt"),
) -> None:
    """Resume an exact deterministic trajectory from a trusted EPOR checkpoint."""

    from epor.training import pretrain

    result = pretrain(
        config,
        output,
        max_steps=max_steps,
        resume_from=checkpoint,
        corpus_path=corpus,
    )
    typer.echo(result.model_dump_json(indent=2))


@eval_app.command("run")
def eval_run(
    checkpoint: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    corpus: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("fixtures/tiny_corpus.txt"),
    max_batches: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Evaluate a tiny reference checkpoint on deterministic fixture windows."""

    from epor.training import evaluate

    result = evaluate(checkpoint, corpus_path=corpus, max_batches=max_batches)
    typer.echo(result.model_dump_json(indent=2))


@app.command("generate")
def generate_command(
    checkpoint: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    prompt: Annotated[str, typer.Argument(help="Prompt encoded with the debug tokenizer.")],
    max_new_tokens: Annotated[int, typer.Option(min=0, max=4096)] = 32,
) -> None:
    """Greedily generate from a tiny reference checkpoint."""

    from epor.training import generate

    result = generate(checkpoint, prompt, max_new_tokens=max_new_tokens)
    typer.echo(result.model_dump_json(indent=2))


@app.command("api")
def api_command(
    host: Annotated[str, typer.Option(help="Numeric loopback bind address only.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8742,
) -> None:
    """Run the loopback-only control API."""

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


@app.command("serve")
def serve_command() -> None:
    """Reserve the future provider-neutral local inference surface."""

    _planned(FutureCommand.serve, "v0.0.8")


@data_app.command("ingest")
def data_ingest() -> None:
    _planned(FutureCommand.data_ingest, "v0.0.2")


@data_app.command("build")
def data_build() -> None:
    _planned(FutureCommand.data_build, "v0.0.2")


@data_app.command("audit")
def data_audit() -> None:
    _planned(FutureCommand.data_audit, "v0.0.2")


@tokenizer_app.command("train")
def tokenizer_train() -> None:
    _planned(FutureCommand.tokenizer_train, "v0.0.3")


@tokenizer_app.command("evaluate")
def tokenizer_evaluate() -> None:
    _planned(FutureCommand.tokenizer_evaluate, "v0.0.3")


@train_app.command("sft")
def train_sft() -> None:
    _planned(FutureCommand.train_sft, "v0.0.6")


@train_app.command("preference")
def train_preference() -> None:
    _planned(FutureCommand.train_preference, "v0.0.6")


@train_app.command("grpo")
def train_grpo() -> None:
    _planned(FutureCommand.train_grpo, "v0.0.6")


@eval_app.command("compare")
def eval_compare() -> None:
    _planned(FutureCommand.eval_compare, "v0.0.5")


@export_app.command("hf")
def export_hf() -> None:
    _planned(FutureCommand.export_hf, "v0.0.8")


@export_app.command("gguf")
def export_gguf() -> None:
    _planned(FutureCommand.export_gguf, "v0.0.8")


if __name__ == "__main__":
    app()
