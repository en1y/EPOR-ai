"""Safe synchronization, verification, and indexing for research sources."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from .catalog import DEFAULT_CATALOG_PATH, load_catalog, select_sources
from .extract import ExtractionError, extract_text
from .models import (
    CatalogError,
    IndexResult,
    IndexStatus,
    IntegrityStatus,
    ResearchEntry,
    SyncResult,
    SyncStatus,
    validate_source_url,
)

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=15.0)
MAX_REDIRECTS = 5
USER_AGENT = "epor-research/0.0.1 (+https://github.com/epor-ai/epor-ai)"


class ResearchSyncError(RuntimeError):
    """Raised for a rejected or failed source transfer."""


def sync_catalog(
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    *,
    project_root: Path | str | None = None,
    offline: bool = False,
    source_ids: Sequence[str] | None = None,
    client: httpx.Client | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[SyncResult]:
    """Synchronize selected catalog sources without ever accepting arbitrary URLs.

    When ``offline`` is true, this is a local integrity check and performs no HTTP
    calls. Existing verified files are idempotently reused in either mode.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    path, root = _resolve_catalog_and_root(catalog_path, project_root)
    catalog = load_catalog(path)
    sources = select_sources(catalog, source_ids)
    results: list[SyncResult] = []

    if offline:
        for source in sources:
            results.append(_verify_entry(source, root, max_bytes=max_bytes))
        return results

    manager = (
        nullcontext(client)
        if client is not None
        else httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=False)
    )
    with manager as active_client:
        assert active_client is not None
        for source in sources:
            existing = _verify_entry(source, root, max_bytes=max_bytes)
            if existing.status in {SyncStatus.VERIFIED, SyncStatus.UNPINNED}:
                results.append(existing)
                continue
            try:
                results.append(
                    _download_entry(source, root, client=active_client, max_bytes=max_bytes)
                )
            except (OSError, ValueError, httpx.HTTPError, ResearchSyncError) as exc:
                results.append(
                    SyncResult(
                        source_id=source.id,
                        status=SyncStatus.ERROR,
                        path=str(source.local_paths.raw),
                        message=str(exc),
                    )
                )
    return results


def verify_catalog(
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    *,
    project_root: Path | str | None = None,
    source_ids: Sequence[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[SyncResult]:
    """Verify local files against pinned or locally recorded SHA-256 digests."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    path, root = _resolve_catalog_and_root(catalog_path, project_root)
    sources = select_sources(load_catalog(path), source_ids)
    return [_verify_entry(source, root, max_bytes=max_bytes) for source in sources]


def index_catalog(
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    *,
    project_root: Path | str | None = None,
    source_ids: Sequence[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[IndexResult]:
    """Extract synchronized sources and write a deterministic ignored JSON index."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    path, root = _resolve_catalog_and_root(catalog_path, project_root)
    catalog = load_catalog(path)
    sources = select_sources(catalog, source_ids)
    results: list[IndexResult] = []
    index_records: list[dict[str, Any]] = []

    for source in sources:
        verification = _verify_entry(source, root, max_bytes=max_bytes)
        extracted_path = _contained_path(root, source.local_paths.extracted)
        if verification.status is SyncStatus.MISSING:
            results.append(
                IndexResult(
                    source.id,
                    IndexStatus.MISSING,
                    str(source.local_paths.extracted),
                    message=verification.message,
                )
            )
            continue
        if verification.status not in {SyncStatus.VERIFIED, SyncStatus.UNPINNED}:
            results.append(
                IndexResult(
                    source.id,
                    IndexStatus.ERROR,
                    str(source.local_paths.extracted),
                    message=verification.message,
                )
            )
            continue
        raw_path = _contained_path(root, source.local_paths.raw)
        try:
            text = extract_text(raw_path, source.media_type)
            encoded = text.encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            unchanged = extracted_path.is_file() and _sha256(extracted_path) == digest
            if not unchanged:
                _atomic_write_bytes(extracted_path, encoded)
            status = IndexStatus.UNCHANGED if unchanged else IndexStatus.INDEXED
            results.append(
                IndexResult(
                    source.id,
                    status,
                    str(source.local_paths.extracted),
                    sha256=digest,
                    characters=len(text),
                )
            )
            index_records.append(
                {
                    "authors": list(source.authors),
                    "canonical_url": source.canonical_url,
                    "evidence_classes": [item.value for item in source.evidence_classes],
                    "extracted_path": str(source.local_paths.extracted),
                    "extracted_sha256": digest,
                    "id": source.id,
                    "integrity": source.integrity.value,
                    "integrity_status": verification.status.value,
                    "media_type": source.media_type,
                    "organization": source.organization,
                    "publication_date": source.publication_date,
                    "raw_sha256": verification.sha256,
                    "tags": list(source.tags),
                    "text_characters": len(text),
                    "title": source.title,
                }
            )
        except (OSError, ExtractionError) as exc:
            results.append(
                IndexResult(
                    source.id,
                    IndexStatus.ERROR,
                    str(source.local_paths.extracted),
                    message=str(exc),
                )
            )

    index_path = root / "research" / "cache" / "index.json"
    payload = {
        "schema_version": 1,
        "catalog_sha256": _sha256(path),
        "sources": sorted(index_records, key=lambda item: item["id"]),
    }
    _atomic_write_json(index_path, payload)
    return results


def _download_entry(
    source: ResearchEntry,
    root: Path,
    *,
    client: httpx.Client,
    max_bytes: int,
) -> SyncResult:
    destination = _contained_path(root, source.local_paths.raw)
    metadata_path = _contained_path(root, source.local_paths.metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = source.canonical_url
    temporary: Path | None = None

    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            validate_source_url(url, source_id=source.id)
            with client.stream(
                "GET",
                url,
                headers={
                    "Accept": source.media_type,
                    "Accept-Encoding": "identity",
                    "User-Agent": USER_AGENT,
                },
                follow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ResearchSyncError(f"{source.id}: redirect has no Location header")
                    if redirect_count == MAX_REDIRECTS:
                        raise ResearchSyncError(f"{source.id}: too many redirects")
                    url = urljoin(url, location)
                    validate_source_url(url, source_id=source.id)
                    continue

                response.raise_for_status()
                validate_source_url(str(response.url), source_id=source.id)
                actual_type, declared_length = _validate_response_headers(
                    source, response.headers, max_bytes
                )
                fd, name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".part",
                    dir=destination.parent,
                )
                temporary = Path(name)
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(fd, "wb") as handle:
                    for chunk in response.iter_raw(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            raise ResearchSyncError(
                                f"{source.id}: response exceeded {max_bytes} byte limit"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if size == 0:
                    raise ResearchSyncError(f"{source.id}: empty response rejected")
                if declared_length is not None and size != declared_length:
                    raise ResearchSyncError(
                        f"{source.id}: body size {size} does not match Content-Length "
                        f"{declared_length}"
                    )
                observed = digest.hexdigest()
                if source.sha256 is not None and observed != source.sha256:
                    raise ResearchSyncError(
                        f"{source.id}: SHA-256 mismatch (expected {source.sha256}, got {observed})"
                    )
                _validate_file_signature(temporary, source.media_type)
                os.replace(temporary, destination)
                temporary = None
                metadata = {
                    "schema_version": 1,
                    "source_id": source.id,
                    "canonical_url": source.canonical_url,
                    "final_url": str(response.url),
                    "retrieved_at": _utc_now(),
                    "media_type": actual_type,
                    "sha256": observed,
                    "size_bytes": size,
                }
                _atomic_write_json(metadata_path, metadata)
                return SyncResult(
                    source.id,
                    SyncStatus.DOWNLOADED,
                    str(source.local_paths.raw),
                    sha256=observed,
                    size_bytes=size,
                    message=(
                        "downloaded digest is recorded only in ignored local metadata; "
                        "the tracked catalog remains unpinned"
                        if source.integrity is IntegrityStatus.UNPINNED
                        else ""
                    ),
                )
        raise ResearchSyncError(f"{source.id}: redirect handling failed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_entry(source: ResearchEntry, root: Path, *, max_bytes: int) -> SyncResult:
    raw_path = _contained_path(root, source.local_paths.raw)
    if not raw_path.is_file():
        return SyncResult(
            source.id,
            SyncStatus.MISSING,
            str(source.local_paths.raw),
            message="local source is missing; run research sync without --offline",
        )
    try:
        size = raw_path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise ResearchSyncError(f"local size {size} is outside the allowed range")
        digest = _sha256(raw_path)
        expected = source.sha256
        if source.integrity is IntegrityStatus.UNPINNED:
            metadata_path = _contained_path(root, source.local_paths.metadata)
            metadata = _load_metadata(metadata_path, source)
            expected = metadata.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ResearchSyncError(
                    "unpinned source has no valid local synchronization metadata"
                )
            recorded_size = metadata.get("size_bytes")
            if recorded_size != size:
                raise ResearchSyncError(f"size mismatch (metadata {recorded_size!r}, local {size})")
        if digest != expected:
            raise ResearchSyncError(f"SHA-256 mismatch (expected {expected}, got {digest})")
        _validate_file_signature(raw_path, source.media_type)
        status = (
            SyncStatus.VERIFIED
            if source.integrity is IntegrityStatus.PINNED
            else SyncStatus.UNPINNED
        )
        return SyncResult(
            source.id,
            status,
            str(source.local_paths.raw),
            sha256=digest,
            size_bytes=size,
            message=(
                "local bytes match ignored synchronization metadata, but no digest is "
                "pinned in the tracked catalog"
                if status is SyncStatus.UNPINNED
                else ""
            ),
        )
    except (OSError, ValueError, ResearchSyncError) as exc:
        return SyncResult(
            source.id,
            SyncStatus.ERROR,
            str(source.local_paths.raw),
            message=str(exc),
        )


def _validate_response_headers(
    source: ResearchEntry, headers: Mapping[str, str], max_bytes: int
) -> tuple[str, int | None]:
    declared_length = headers.get("content-length")
    length: int | None = None
    if declared_length is not None:
        try:
            length = int(declared_length)
        except ValueError as exc:
            raise ResearchSyncError(f"{source.id}: invalid Content-Length") from exc
        if length <= 0 or length > max_bytes:
            raise ResearchSyncError(
                f"{source.id}: Content-Length {length} is outside the allowed range"
            )
    raw_type = headers.get("content-type")
    if not raw_type:
        raise ResearchSyncError(f"{source.id}: response has no Content-Type")
    actual_type = raw_type.partition(";")[0].strip().lower()
    compatible: dict[str, set[str]] = {
        "application/pdf": {"application/pdf", "application/octet-stream"},
        "text/html": {"text/html", "application/xhtml+xml"},
        "text/markdown": {"text/markdown", "text/plain"},
        "text/plain": {"text/plain", "text/markdown"},
    }
    if actual_type not in compatible[source.media_type]:
        raise ResearchSyncError(
            f"{source.id}: Content-Type {actual_type!r} does not match {source.media_type!r}"
        )
    content_encoding = headers.get("content-encoding", "identity").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise ResearchSyncError(
            f"{source.id}: Content-Encoding {content_encoding!r} is forbidden; "
            "the archive hashes transfer bytes"
        )
    return actual_type, length


def _validate_file_signature(path: Path, media_type: str) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(512).lstrip()
    if media_type == "application/pdf" and not prefix.startswith(b"%PDF-"):
        raise ResearchSyncError("PDF source does not have a PDF signature")
    if media_type == "text/html":
        lowered = prefix.lower()
        if b"<html" not in lowered and not lowered.startswith((b"<!doctype html", b"<article")):
            raise ResearchSyncError("HTML source does not have a recognizable HTML signature")
    if media_type.startswith("text/"):
        try:
            prefix.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResearchSyncError("text source is not valid UTF-8") from exc


def _load_metadata(path: Path, source: ResearchEntry) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchSyncError(f"cannot read local metadata {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ResearchSyncError(f"local metadata {path} is not an object")
    if payload.get("schema_version") != 1:
        raise ResearchSyncError(f"local metadata {path} has an unsupported schema version")
    if (
        payload.get("source_id") != source.id
        or payload.get("canonical_url") != source.canonical_url
    ):
        raise ResearchSyncError("local metadata does not match catalog identity")
    digest = payload.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ResearchSyncError("local metadata has an invalid SHA-256")
    size = payload.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ResearchSyncError("local metadata has an invalid size_bytes")
    return payload


def _resolve_catalog_and_root(
    catalog_path: Path | str, project_root: Path | str | None
) -> tuple[Path, Path]:
    root = Path.cwd() if project_root is None else Path(project_root)
    root = root.resolve()
    path = Path(catalog_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CatalogError(f"catalog path must be contained by project root {root}") from exc
    return path, root


def _contained_path(root: Path, relative: object) -> Path:
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # Defensive check in addition to catalog validation.
        raise CatalogError(f"research path escapes project root: {relative}") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    _atomic_write_bytes(path, encoded)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
