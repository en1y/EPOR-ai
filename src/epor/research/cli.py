"""Typer command group for ``epor research``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .catalog import DEFAULT_CATALOG_PATH
from .models import IndexResult, IndexStatus, SyncResult, SyncStatus
from .service import index_catalog, sync_catalog, verify_catalog

research_app = typer.Typer(
    name="research",
    help="Synchronize and verify the allowlisted research archive.",
    no_args_is_help=True,
)


def _emit(results: Sequence[SyncResult | IndexResult]) -> None:
    typer.echo(json.dumps([asdict(item) for item in results], indent=2, default=str))


@research_app.command("sync")
def sync_command(
    catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CATALOG_PATH,
    offline: Annotated[
        bool, typer.Option(help="Forbid network access and check local files only.")
    ] = False,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Limit work to a catalog source ID."),
    ] = None,
) -> None:
    results = sync_catalog(catalog, offline=offline, source_ids=source)
    _emit(results)
    if any(item.status in {SyncStatus.ERROR, SyncStatus.MISSING} for item in results):
        raise typer.Exit(code=1)


@research_app.command("verify")
def verify_command(
    catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CATALOG_PATH,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Limit work to a catalog source ID."),
    ] = None,
) -> None:
    results = verify_catalog(catalog, source_ids=source)
    _emit(results)
    if any(item.status is not SyncStatus.VERIFIED for item in results):
        raise typer.Exit(code=1)


@research_app.command("index")
def index_command(
    catalog: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CATALOG_PATH,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Limit work to a catalog source ID."),
    ] = None,
) -> None:
    results = index_catalog(catalog, source_ids=source)
    _emit(results)
    if any(item.status in {IndexStatus.ERROR, IndexStatus.MISSING} for item in results):
        raise typer.Exit(code=1)
