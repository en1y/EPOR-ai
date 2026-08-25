"""Validated schemas used by the EPOR research archive.

The tracked catalog intentionally describes both downloaded and not-yet-downloaded
sources.  A ``null`` SHA-256 means that the source has not been pinned yet; the
first successful synchronization records the observed digest in ignored local
metadata without silently editing the tracked ledger.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

ENTRY_FIELDS = frozenset(
    {
        "id",
        "canonical_url",
        "title",
        "authors",
        "organization",
        "publication_date",
        "retrieved_at",
        "media_type",
        "integrity",
        "sha256",
        "license_access",
        "tags",
        "evidence_classes",
        "local_paths",
    }
)
LOCAL_PATH_FIELDS = frozenset({"raw", "extracted", "metadata"})


class CatalogError(ValueError):
    """Raised when the tracked research catalog violates its schema or policy."""


class EvidenceClass(StrEnum):
    """Required labels for claims made in EPOR research documentation."""

    DISCLOSED = "disclosed"
    REPORTED_RESULT = "reported-result"
    EPOR_ADAPTATION = "EPOR-adaptation"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"


class IntegrityStatus(StrEnum):
    """Whether a source digest is part of the reviewed, tracked ledger."""

    PINNED = "pinned"
    UNPINNED = "unpinned"


class SyncStatus(StrEnum):
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    UNPINNED = "unpinned"
    MISSING = "missing"
    ERROR = "error"


class IndexStatus(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    MISSING = "missing"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LocalPaths:
    raw: PurePosixPath
    extracted: PurePosixPath
    metadata: PurePosixPath

    @classmethod
    def from_mapping(cls, value: object, *, source_id: str) -> LocalPaths:
        if not isinstance(value, Mapping):
            raise CatalogError(f"{source_id}: local_paths must be a mapping")
        _reject_unknown_fields(value, LOCAL_PATH_FIELDS, context=f"{source_id}.local_paths")

        def parse(name: str, expected_parent: str) -> PurePosixPath:
            raw_value = value.get(name)
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise CatalogError(f"{source_id}: local_paths.{name} must be a string")
            path = PurePosixPath(raw_value)
            if path.is_absolute() or ".." in path.parts:
                raise CatalogError(f"{source_id}: local_paths.{name} must be relative")
            expected = ("research", expected_parent)
            if path.parts[:2] != expected:
                raise CatalogError(
                    f"{source_id}: local_paths.{name} must be below {'/'.join(expected)}/"
                )
            return path

        return cls(
            raw=parse("raw", "raw"),
            extracted=parse("extracted", "extracted"),
            metadata=parse("metadata", "cache"),
        )


@dataclass(frozen=True, slots=True)
class ResearchEntry:
    id: str
    canonical_url: str
    title: str
    authors: tuple[str, ...]
    organization: str
    publication_date: str
    retrieved_at: str | None
    media_type: str
    integrity: IntegrityStatus
    sha256: str | None
    license_access: str
    tags: tuple[str, ...]
    evidence_classes: tuple[EvidenceClass, ...]
    local_paths: LocalPaths

    @classmethod
    def from_mapping(cls, value: object) -> ResearchEntry:
        if not isinstance(value, Mapping):
            raise CatalogError("each source entry must be a mapping")
        _reject_unknown_fields(value, ENTRY_FIELDS, context="source")
        source_id = _required_string(value, "id", context="source")
        if not SOURCE_ID_RE.fullmatch(source_id):
            raise CatalogError(f"{source_id!r}: id must be lower-kebab-case")

        canonical_url = _required_string(value, "canonical_url", context=source_id)
        validate_source_url(canonical_url, source_id=source_id)
        title = _required_string(value, "title", context=source_id)
        organization = _required_string(value, "organization", context=source_id)
        publication_date = _date_string(value.get("publication_date"), source_id)
        retrieved_at = _optional_timestamp(value.get("retrieved_at"), source_id)
        media_type = _required_string(value, "media_type", context=source_id).lower()
        if media_type not in {
            "application/pdf",
            "text/html",
            "text/markdown",
            "text/plain",
        }:
            raise CatalogError(f"{source_id}: unsupported media_type {media_type!r}")

        try:
            integrity = IntegrityStatus(_required_string(value, "integrity", context=source_id))
        except ValueError as exc:
            raise CatalogError(f"{source_id}: invalid integrity status: {exc}") from exc

        digest = value.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
            raise CatalogError(f"{source_id}: sha256 must be null or 64 lowercase hex chars")
        if integrity is IntegrityStatus.PINNED and digest is None:
            raise CatalogError(f"{source_id}: pinned integrity requires a SHA-256")
        if integrity is IntegrityStatus.UNPINNED and digest is not None:
            raise CatalogError(f"{source_id}: unpinned integrity requires sha256: null")

        authors = _string_sequence(value.get("authors"), "authors", source_id)
        tags = _string_sequence(value.get("tags"), "tags", source_id)
        if not authors:
            raise CatalogError(f"{source_id}: authors cannot be empty")
        if not tags:
            raise CatalogError(f"{source_id}: tags cannot be empty")
        evidence_values = _string_sequence(
            value.get("evidence_classes"), "evidence_classes", source_id
        )
        if not evidence_values:
            raise CatalogError(f"{source_id}: evidence_classes cannot be empty")
        try:
            evidence_classes = tuple(EvidenceClass(item) for item in evidence_values)
        except ValueError as exc:
            raise CatalogError(f"{source_id}: invalid evidence class: {exc}") from exc

        return cls(
            id=source_id,
            canonical_url=canonical_url,
            title=title,
            authors=authors,
            organization=organization,
            publication_date=publication_date,
            retrieved_at=retrieved_at,
            media_type=media_type,
            integrity=integrity,
            sha256=digest,
            license_access=_required_string(value, "license_access", context=source_id),
            tags=tags,
            evidence_classes=evidence_classes,
            local_paths=LocalPaths.from_mapping(value.get("local_paths"), source_id=source_id),
        )


@dataclass(frozen=True, slots=True)
class ResearchCatalog:
    schema_version: int
    sources: tuple[ResearchEntry, ...]

    @classmethod
    def from_mapping(cls, value: object) -> ResearchCatalog:
        if not isinstance(value, Mapping):
            raise CatalogError("catalog root must be a mapping")
        _reject_unknown_fields(value, {"schema_version", "sources"}, context="catalog")
        if value.get("schema_version") != 1:
            raise CatalogError("catalog schema_version must be 1")
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise CatalogError("catalog sources must be a sequence")
        sources = tuple(ResearchEntry.from_mapping(item) for item in raw_sources)
        if not sources:
            raise CatalogError("catalog sources cannot be empty")
        seen: set[str] = set()
        seen_urls: set[str] = set()
        seen_paths: set[PurePosixPath] = set()
        for source in sources:
            if source.id in seen:
                raise CatalogError(f"duplicate source id: {source.id}")
            if source.canonical_url in seen_urls:
                raise CatalogError(f"duplicate canonical_url: {source.canonical_url}")
            seen.add(source.id)
            seen_urls.add(source.canonical_url)
            for local_path in (
                source.local_paths.raw,
                source.local_paths.extracted,
                source.local_paths.metadata,
            ):
                if local_path in seen_paths:
                    raise CatalogError(f"duplicate local path: {local_path}")
                seen_paths.add(local_path)
        return cls(schema_version=1, sources=sources)


@dataclass(frozen=True, slots=True)
class SyncResult:
    source_id: str
    status: SyncStatus
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class IndexResult:
    source_id: str
    status: IndexStatus
    path: str
    sha256: str | None = None
    characters: int | None = None
    message: str = ""


# Exact hosts only: suffix matching would permit attacker-controlled lookalikes.
ALLOWED_SOURCE_HOSTS = frozenset(
    {
        "ai.google.dev",
        "arxiv.org",
        "developers.googleblog.com",
        "github.com",
        "raw.githubusercontent.com",
        "research.google",
        "transformer-circuits.pub",
        "www.anthropic.com",
        "www-cdn.anthropic.com",
    }
)


def validate_source_url(url: str, *, source_id: str = "source") -> None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise CatalogError(f"{source_id}: canonical_url must use https")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise CatalogError(f"{source_id}: canonical_url contains forbidden authority data")
    if host not in ALLOWED_SOURCE_HOSTS:
        raise CatalogError(f"{source_id}: source host {host!r} is not allowlisted")
    if not parsed.path.startswith("/") or parsed.path == "/":
        raise CatalogError(f"{source_id}: canonical_url must include a path")
    if parsed.fragment:
        raise CatalogError(f"{source_id}: canonical_url fragments are forbidden")


def _required_string(value: Mapping[str, Any], key: str, *, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise CatalogError(f"{context}: {key} must be a non-empty string")
    return item.strip()


def _string_sequence(value: object, name: str, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CatalogError(f"{context}: {name} must be a sequence of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise CatalogError(f"{context}: {name} must contain only non-empty strings")
        result.append(item.strip())
    return tuple(result)


def _date_string(value: object, context: str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise CatalogError(f"{context}: publication_date must be an ISO date")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise CatalogError(f"{context}: invalid publication_date {value!r}") from exc


def _optional_timestamp(value: object, context: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CatalogError(f"{context}: retrieved_at must be an ISO timestamp or null") from exc
    else:
        raise CatalogError(f"{context}: retrieved_at must be an ISO timestamp or null")
    if parsed.tzinfo is None:
        raise CatalogError(f"{context}: retrieved_at must include a timezone")
    return parsed.isoformat().replace("+00:00", "Z")


def _reject_unknown_fields(
    value: Mapping[object, object], allowed: set[str] | frozenset[str], *, context: str
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise CatalogError(f"{context}: unknown field(s): {', '.join(unknown)}")
