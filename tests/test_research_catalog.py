from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from epor.research import CatalogError, IntegrityStatus, load_catalog, select_sources
from epor.research.models import ResearchCatalog, validate_source_url

REQUIRED_SOURCE_IDS = {
    "anthropic-general-assistant",
    "anthropic-hh-rlhf",
    "anthropic-constitutional-ai",
    "anthropic-repeated-data",
    "anthropic-long-context-prompting",
    "anthropic-contextual-retrieval",
    "anthropic-claude-3-model-card",
    "anthropic-model-written-evals",
    "anthropic-towards-monosemanticity",
    "anthropic-scaling-monosemanticity",
    "anthropic-circuit-tracing",
    "deepseek-llm",
    "deepseek-moe",
    "deepseek-coder",
    "deepseek-math",
    "deepseek-v2",
    "deepseek-coder-v2",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-v3-2",
    "deepseek-engram",
    "deepseek-v3-repository",
    "deepseek-deepgemm",
    "deepseek-flashmla",
    "deepseek-deepep",
    "gemma-3-technical-report",
    "gemma-3n-overview",
    "gemma-3n-developer-guide",
    "matformer",
}


def _entry() -> dict[str, object]:
    return {
        "id": "example-paper",
        "canonical_url": "https://arxiv.org/pdf/1234.56789",
        "title": "Example paper",
        "authors": ["A. Researcher"],
        "organization": "Example Lab",
        "publication_date": "2026-01-02",
        "retrieved_at": None,
        "media_type": "application/pdf",
        "integrity": "unpinned",
        "sha256": None,
        "license_access": "Public access; rights require review.",
        "tags": ["example"],
        "evidence_classes": ["disclosed", "unknown"],
        "local_paths": {
            "raw": "research/raw/example-paper.pdf",
            "extracted": "research/extracted/example-paper.txt",
            "metadata": "research/cache/example-paper.json",
        },
    }


def test_tracked_catalog_covers_required_archive() -> None:
    catalog = load_catalog(Path("research/catalog.yaml"))
    by_id = {source.id: source for source in catalog.sources}
    assert set(by_id) == REQUIRED_SOURCE_IDS
    assert all(source.authors for source in catalog.sources)
    assert all(source.organization for source in catalog.sources)
    assert all(source.publication_date for source in catalog.sources)
    assert all(source.license_access for source in catalog.sources)
    assert all(source.tags for source in catalog.sources)
    assert all(source.evidence_classes for source in catalog.sources)
    assert all(source.integrity is IntegrityStatus.PINNED for source in catalog.sources)
    assert all(source.retrieved_at is not None for source in catalog.sources)
    assert all(source.sha256 is not None for source in catalog.sources)


def test_catalog_rejects_unknown_fields_and_inconsistent_integrity() -> None:
    entry = _entry()
    entry["surprise"] = True
    with pytest.raises(CatalogError, match="unknown field"):
        ResearchCatalog.from_mapping({"schema_version": 1, "sources": [entry]})

    entry = _entry()
    entry["integrity"] = "pinned"
    with pytest.raises(CatalogError, match="requires a SHA-256"):
        ResearchCatalog.from_mapping({"schema_version": 1, "sources": [entry]})

    entry = _entry()
    entry["sha256"] = "0" * 64
    with pytest.raises(CatalogError, match="requires sha256: null"):
        ResearchCatalog.from_mapping({"schema_version": 1, "sources": [entry]})


def test_catalog_rejects_path_collisions_and_escapes() -> None:
    first = _entry()
    second = _entry()
    second["id"] = "second-paper"
    second["canonical_url"] = "https://arxiv.org/pdf/9999.00001"
    with pytest.raises(CatalogError, match="duplicate local path"):
        ResearchCatalog.from_mapping({"schema_version": 1, "sources": [first, second]})

    escaped = _entry()
    escaped_paths = dict(cast(dict[str, str], escaped["local_paths"]))
    escaped_paths["raw"] = "research/raw/../../outside.pdf"
    escaped["local_paths"] = escaped_paths
    with pytest.raises(CatalogError, match="must be relative"):
        ResearchCatalog.from_mapping({"schema_version": 1, "sources": [escaped]})


@pytest.mark.parametrize(
    "url",
    [
        "http://arxiv.org/pdf/1234.56789",
        "https://arxiv.org.evil.example/pdf/1234.56789",
        "https://user@arxiv.org/pdf/1234.56789",
        "https://arxiv.org/",
        "https://arxiv.org/pdf/1234.56789#fragment",
    ],
)
def test_source_url_policy_rejects_unsafe_authorities(url: str) -> None:
    with pytest.raises(CatalogError):
        validate_source_url(url)


def test_source_selection_is_ordered_deduplicated_and_strict() -> None:
    catalog = load_catalog()
    selected = select_sources(catalog, ["deepseek-r1", "matformer", "deepseek-r1"])
    assert [source.id for source in selected] == ["deepseek-r1", "matformer"]
    with pytest.raises(CatalogError, match="unknown research source"):
        select_sources(catalog, ["not-in-the-ledger"])
