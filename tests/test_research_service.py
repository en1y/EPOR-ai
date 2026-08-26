from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from epor.research import IndexStatus, SyncStatus
from epor.research.cli import research_app
from epor.research.extract import extract_text
from epor.research.service import index_catalog, sync_catalog, verify_catalog

HTML = b"<!doctype html><html><body><h1>Hello</h1><script>bad()</script><p>World</p></body></html>"
PDF_TEXT = "Offline PDF evidence"


def _minimal_pdf_bytes() -> bytes:
    content = f"BT\n/F1 12 Tf\n72 720 Td\n({PDF_TEXT}) Tj\nET\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"endstream",
    ]
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")

    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(document)


def _write_catalog(
    root: Path,
    *,
    integrity: str = "pinned",
    digest: str | None = None,
    source_url: str = "https://ai.google.dev/example",
    media_type: str = "text/html",
    raw_path: str = "research/raw/example-source.html",
) -> Path:
    if integrity == "pinned" and digest is None:
        digest = hashlib.sha256(HTML).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "sources": [
            {
                "id": "example-source",
                "canonical_url": source_url,
                "title": "Example source",
                "authors": ["Example Author"],
                "organization": "Example Organization",
                "publication_date": "2026-01-02",
                "retrieved_at": None,
                "media_type": media_type,
                "integrity": integrity,
                "sha256": digest,
                "license_access": "Test fixture only.",
                "tags": ["fixture"],
                "evidence_classes": ["disclosed", "EPOR-adaptation"],
                "local_paths": {
                    "raw": raw_path,
                    "extracted": "research/extracted/example-source.txt",
                    "metadata": "research/cache/example-source.json",
                },
            }
        ],
    }
    path = root / "research" / "catalog.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _client(
    handler: httpx.MockTransport | Any,
) -> httpx.Client:
    transport = (
        handler if isinstance(handler, httpx.MockTransport) else httpx.MockTransport(handler)
    )
    return httpx.Client(transport=transport)


def _streaming_response(
    body: bytes = HTML, *, headers: dict[str, str] | None = None
) -> httpx.Response:
    response_headers = {"content-type": "text/html"} if headers is None else headers
    return httpx.Response(
        200,
        headers=response_headers,
        stream=httpx.ByteStream(body),
    )


def test_pinned_sync_is_atomic_verified_and_idempotent(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["accept-encoding"] == "identity"
        return _streaming_response()

    with _client(handler) as client:
        first = sync_catalog(project_root=tmp_path, client=client)
        second = sync_catalog(project_root=tmp_path, client=client)

    assert calls == 1
    assert first[0].status is SyncStatus.DOWNLOADED
    assert second[0].status is SyncStatus.VERIFIED
    assert verify_catalog(project_root=tmp_path)[0].status is SyncStatus.VERIFIED
    assert (tmp_path / "research/raw/example-source.html").read_bytes() == HTML

    metadata = json.loads(
        (tmp_path / "research/cache/example-source.json").read_text(encoding="utf-8")
    )
    assert metadata["sha256"] == hashlib.sha256(HTML).hexdigest()
    assert metadata["size_bytes"] == len(HTML)
    assert not list((tmp_path / "research/raw").glob("*.part"))


def test_unpinned_source_is_reusable_but_never_reported_verified(tmp_path: Path) -> None:
    _write_catalog(tmp_path, integrity="unpinned", digest=None)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _streaming_response()

    with _client(handler) as client:
        first = sync_catalog(project_root=tmp_path, client=client)
        reused = sync_catalog(project_root=tmp_path, client=client)
        offline = sync_catalog(project_root=tmp_path, client=client, offline=True)

    assert calls == 1
    assert first[0].status is SyncStatus.DOWNLOADED
    assert reused[0].status is SyncStatus.UNPINNED
    assert offline[0].status is SyncStatus.UNPINNED
    assert "no digest is pinned" in offline[0].message
    assert verify_catalog(project_root=tmp_path)[0].status is SyncStatus.UNPINNED


def test_index_extracts_locally_and_is_deterministic(tmp_path: Path) -> None:
    _write_catalog(tmp_path, integrity="unpinned", digest=None)

    def handler(request: httpx.Request) -> httpx.Response:
        return _streaming_response()

    with _client(handler) as client:
        sync_catalog(project_root=tmp_path, client=client)

    first = index_catalog(project_root=tmp_path)
    index_path = tmp_path / "research/cache/index.json"
    first_index = index_path.read_bytes()
    second = index_catalog(project_root=tmp_path)

    assert first[0].status is IndexStatus.INDEXED
    assert second[0].status is IndexStatus.UNCHANGED
    assert index_path.read_bytes() == first_index
    assert (tmp_path / "research/extracted/example-source.txt").read_text(
        encoding="utf-8"
    ) == "Hello\n\nWorld\n"
    record = json.loads(first_index)["sources"][0]
    assert record["integrity"] == "unpinned"
    assert record["integrity_status"] == "unpinned"
    assert "bad()" not in extract_text(tmp_path / "research/raw/example-source.html", "text/html")


def test_index_extracts_pdf_offline_and_is_deterministic(tmp_path: Path) -> None:
    pdf = _minimal_pdf_bytes()
    raw_path = "research/raw/example-source.pdf"
    _write_catalog(
        tmp_path,
        digest=hashlib.sha256(pdf).hexdigest(),
        media_type="application/pdf",
        raw_path=raw_path,
    )
    source_path = tmp_path / raw_path
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(pdf)

    first = index_catalog(project_root=tmp_path)
    index_path = tmp_path / "research/cache/index.json"
    first_index = index_path.read_bytes()
    extracted_path = tmp_path / "research/extracted/example-source.txt"
    second = index_catalog(project_root=tmp_path)

    extracted = f"{PDF_TEXT}\n"
    extracted_digest = hashlib.sha256(extracted.encode()).hexdigest()
    assert first[0].status is IndexStatus.INDEXED
    assert first[0].sha256 == extracted_digest
    assert first[0].characters == len(extracted)
    assert second[0].status is IndexStatus.UNCHANGED
    assert index_path.read_bytes() == first_index
    assert extracted_path.read_text(encoding="utf-8") == extracted
    record = json.loads(first_index)["sources"][0]
    assert record["media_type"] == "application/pdf"
    assert record["raw_sha256"] == hashlib.sha256(pdf).hexdigest()
    assert record["extracted_sha256"] == extracted_digest


@pytest.mark.parametrize(
    ("headers", "body", "message"),
    [
        ({"content-type": "application/pdf"}, HTML, "Content-Type"),
        ({"content-type": "text/html", "content-encoding": "gzip"}, HTML, "Content-Encoding"),
        ({"content-type": "text/html", "content-length": "2"}, HTML, "body size"),
    ],
)
def test_sync_rejects_invalid_response_contract(
    tmp_path: Path, headers: dict[str, str], body: bytes, message: str
) -> None:
    _write_catalog(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return _streaming_response(body, headers=headers)

    with _client(handler) as client:
        result = sync_catalog(project_root=tmp_path, client=client)
    assert result[0].status is SyncStatus.ERROR
    assert message in result[0].message
    assert not (tmp_path / "research/raw/example-source.html").exists()
    assert not list((tmp_path / "research/raw").glob("*.part"))


def test_sync_rejects_oversize_stream_and_preserves_existing_file(tmp_path: Path) -> None:
    _write_catalog(tmp_path, digest="f" * 64)
    destination = tmp_path / "research/raw/example-source.html"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        return _streaming_response()

    with _client(handler) as client:
        result = sync_catalog(project_root=tmp_path, client=client, max_bytes=len(HTML) - 1)
    assert result[0].status is SyncStatus.ERROR
    assert destination.read_bytes() == b"old bytes"
    assert not list(destination.parent.glob("*.part"))


def test_sync_rejects_redirect_outside_allowlist(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"location": "https://example.com/paper"})

    with _client(handler) as client:
        result = sync_catalog(project_root=tmp_path, client=client)
    assert calls == 1
    assert result[0].status is SyncStatus.ERROR
    assert "not allowlisted" in result[0].message


def test_missing_offline_source_never_touches_transport(tmp_path: Path) -> None:
    _write_catalog(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"offline sync attempted network request: {request.url}")

    with _client(handler) as client:
        result = sync_catalog(project_root=tmp_path, client=client, offline=True)
    assert result[0].status is SyncStatus.MISSING


def test_verify_cli_exits_nonzero_for_unpinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_catalog(tmp_path, integrity="unpinned", digest=None)

    def handler(request: httpx.Request) -> httpx.Response:
        return _streaming_response()

    with _client(handler) as client:
        sync_catalog(project_root=tmp_path, client=client)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(research_app, ["verify"])
    assert result.exit_code == 1
    assert '"status": "unpinned"' in result.stdout
