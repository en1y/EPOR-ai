"""Loading and selecting entries from the tracked research ledger."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from .models import CatalogError, ResearchCatalog, ResearchEntry

DEFAULT_CATALOG_PATH = Path("research/catalog.yaml")


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> ResearchCatalog:
    """Safely load and validate a research catalog.

    PyYAML's safe loader is required because catalog files are data, never code.
    """

    catalog_path = Path(path)
    try:
        with catalog_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except OSError as exc:
        raise CatalogError(f"cannot read research catalog {catalog_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"invalid YAML in research catalog {catalog_path}: {exc}") from exc
    return ResearchCatalog.from_mapping(payload)


def select_sources(
    catalog: ResearchCatalog, source_ids: Iterable[str] | None
) -> tuple[ResearchEntry, ...]:
    if source_ids is None:
        return catalog.sources
    requested = tuple(dict.fromkeys(source_ids))
    by_id = {source.id: source for source in catalog.sources}
    unknown = [source_id for source_id in requested if source_id not in by_id]
    if unknown:
        raise CatalogError(f"unknown research source id(s): {', '.join(unknown)}")
    return tuple(by_id[source_id] for source_id in requested)
