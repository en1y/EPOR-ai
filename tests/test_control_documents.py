from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from epor.control.api import create_app
from epor.control.documents import (
    MAX_DOCUMENT_BYTES,
    DocumentTooLargeError,
    build_index,
    read_document,
)
from epor.control.settings import ControlSettings


def _settings(tmp_path: Path) -> ControlSettings:
    project_root = Path(__file__).resolve().parents[1]
    return ControlSettings(
        project_root=project_root,
        database_path=tmp_path / "documents.sqlite3",
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        research_catalog_path=project_root / "research" / "catalog.yaml",
    )


def test_index_covers_tracked_markdown_and_titles_it(tmp_path: Path) -> None:
    index = build_index(_settings(tmp_path).project_root)

    assert "README.md" in index
    assert "docs/ROADMAP.md" in index
    assert "docs/research/anthropic.md" in index
    assert index["docs/ROADMAP.md"].title == "EPOR AI roadmap"
    assert index["docs/ROADMAP.md"].group == "docs"
    assert index["README.md"].group == ""
    # Ignored third-party downloads are never indexed, whatever is on disk.
    assert not any(slug.startswith("research/") for slug in index)


def test_index_refuses_paths_outside_the_document_roots(tmp_path: Path) -> None:
    index = build_index(_settings(tmp_path).project_root)

    assert "pyproject.toml" not in index
    assert "../pyproject.toml" not in index
    assert "docs/../pyproject.toml" not in index


def test_read_document_refuses_oversized_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "huge.md").write_text("#" * (MAX_DOCUMENT_BYTES + 1), encoding="utf-8")

    document = build_index(root)["docs/huge.md"]
    with pytest.raises(DocumentTooLargeError):
        read_document(document)


def test_document_endpoints_serve_only_indexed_slugs(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://control.test") as client:
            listing = (await client.get("/api/v1/documents")).json()
            assert listing["count"] == len(listing["items"])
            slugs = {item["slug"] for item in listing["items"]}
            assert {"README.md", "docs/ARCHITECTURE.md"} <= slugs

            document = await client.get("/api/v1/documents/docs/ARCHITECTURE.md")
            assert document.status_code == 200
            body = document.json()
            assert body["slug"] == "docs/ARCHITECTURE.md"
            assert body["markdown"].startswith("#")

            for escape in ("../pyproject.toml", "docs/../pyproject.toml", "research/catalog.yaml"):
                refused = await client.get(f"/api/v1/documents/{escape}")
                assert refused.status_code == 404, escape

            # The FastAPI schema UI keeps its own path; the reader never shadows it.
            assert (await client.get("/api/v1/docs")).status_code == 200

    asyncio.run(scenario())
