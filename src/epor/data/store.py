"""Private immutable storage primitives for data provenance layers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import DataAudit, DatasetBuild, DocumentRecord, SourceRegistration, Tombstone

STREAM_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024


class DataStoreError(RuntimeError):
    """A data layer is invalid, mutable, or outside its storage contract."""


class DataInputTooLargeError(DataStoreError):
    """Streaming acquisition exceeded the reviewed local input limit."""


def canonical_json_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    if isinstance(value, BaseModel):
        payload: Any = value.model_dump(mode="json")
    else:
        payload = value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(STREAM_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_model(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or not all(c in "0123456789abcdef" for c in value):
        raise DataStoreError(f"{label} are lowercase SHA-256 hex digests")


class DataStore:
    """Immutable, content-addressed local data layers beneath one private root."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        for name in (
            "registrations",
            "raw/sha256",
            "records",
            "normalized",
            "quarantine",
            "tombstones",
            "filtered",
            "builds",
            "tokenizers",
            "tokenized",
            "evaluations",
            "evaluation-reports",
            "scaling-reports",
            "alignment",
            "context-certificates",
            "audit/ingest",
            "tmp",
        ):
            directory = self.root / name
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)

    def _write_immutable(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            if path.stat().st_size != len(payload) or sha256_file(path) != sha256_bytes(payload):
                raise DataStoreError(
                    f"immutable data record already exists with different bytes: {path}"
                )
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def register(self, registration: SourceRegistration) -> str:
        payload = canonical_json_bytes(registration)
        digest = sha256_bytes(payload)
        self._write_immutable(self.root / "registrations" / f"{registration.id}.json", payload)
        return digest

    def registration(self, source_id: str) -> SourceRegistration:
        path = self.root / "registrations" / f"{source_id}.json"
        try:
            return SourceRegistration.model_validate_json(path.read_bytes())
        except FileNotFoundError as exc:
            raise DataStoreError(f"source registration not found: {source_id}") from exc

    def registrations(self) -> list[SourceRegistration]:
        return [
            SourceRegistration.model_validate_json(path.read_bytes())
            for path in sorted((self.root / "registrations").glob("*.json"))
        ]

    def stream_raw(
        self,
        source: Path,
        *,
        max_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        check_cancelled: Callable[[], None] | None = None,
    ) -> tuple[Path, str, int]:
        """Copy one local input without materializing it, then address it by SHA-256."""

        if not source.is_file() or source.is_symlink():
            raise DataStoreError("data input must be a regular non-symlink file")
        digest = hashlib.sha256()
        total = 0
        descriptor, temporary_name = tempfile.mkstemp(prefix="raw-", dir=self.root / "tmp")
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
                while chunk := incoming.read(STREAM_CHUNK_BYTES):
                    if check_cancelled is not None:
                        check_cancelled()
                    total += len(chunk)
                    if total > max_bytes:
                        raise DataInputTooLargeError(
                            f"data input exceeds the {max_bytes}-byte streaming limit"
                        )
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            raw_sha256 = digest.hexdigest()
            target = self.root / "raw" / "sha256" / raw_sha256[:2] / f"{raw_sha256}.bin"
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists():
                if target.stat().st_size != total or sha256_file(target) != raw_sha256:
                    raise DataStoreError("existing raw object does not match its content address")
                temporary.unlink()
            else:
                temporary.chmod(0o600)
                os.replace(temporary, target)
            return target, raw_sha256, total
        finally:
            temporary.unlink(missing_ok=True)

    def raw_path(self, raw_sha256: str) -> Path:
        return self.root / "raw" / "sha256" / raw_sha256[:2] / f"{raw_sha256}.bin"

    def write_document(self, record: DocumentRecord) -> str:
        payload = canonical_json_bytes(record)
        record_sha256 = sha256_bytes(payload)
        self._write_immutable(self.root / "records" / f"{record.document_id}.json", payload)
        layer = "normalized" if record.disposition == "admitted" else "quarantine"
        self._write_immutable(self.root / layer / f"{record.document_id}.json", payload)
        return record_sha256

    def documents(self) -> list[DocumentRecord]:
        return [
            DocumentRecord.model_validate_json(path.read_bytes())
            for path in sorted((self.root / "records").glob("*.json"))
        ]

    def write_ingest_audit(self, payload: dict[str, Any]) -> tuple[Path, str]:
        encoded = canonical_json_bytes(payload)
        digest = sha256_bytes(encoded)
        target = self.root / "audit" / "ingest" / f"{digest}.json"
        self._write_immutable(target, encoded)
        return target, digest

    def write_tombstone(self, tombstone: Tombstone) -> Path:
        target = self.root / "tombstones" / f"{tombstone.id}.json"
        self._write_immutable(target, canonical_json_bytes(tombstone))
        return target

    def tombstones(self) -> list[Tombstone]:
        return [
            Tombstone.model_validate_json(path.read_bytes())
            for path in sorted((self.root / "tombstones").glob("*.json"))
        ]

    def build_path(self, dataset_id: str, build_id: str) -> Path:
        return self.root / "builds" / dataset_id / build_id

    def write_build_file(self, dataset_id: str, build_id: str, name: str, payload: bytes) -> Path:
        if name.startswith("/") or ".." in Path(name).parts:
            raise DataStoreError("build artifact path must remain inside the build")
        target = self.build_path(dataset_id, build_id) / name
        self._write_immutable(target, payload)
        return target

    def tokenizer_path(self, tokenizer_id: str) -> Path:
        _require_sha256(tokenizer_id, "tokenizer IDs")
        return self.root / "tokenizers" / tokenizer_id

    def write_tokenizer_file(self, tokenizer_id: str, name: str, payload: bytes) -> Path:
        if "/" in name or name in {"", ".", ".."}:
            raise DataStoreError("tokenizer artifact names must be plain file names")
        target = self.tokenizer_path(tokenizer_id) / name
        self._write_immutable(target, payload)
        return target

    def tokenizer_ids(self) -> list[str]:
        return sorted(
            path.parent.name for path in (self.root / "tokenizers").glob("*/manifest.json")
        )

    def document(self, document_id: str) -> DocumentRecord:
        _require_sha256(document_id, "document IDs")
        path = self.root / "records" / f"{document_id}.json"
        try:
            return DocumentRecord.model_validate_json(path.read_bytes())
        except FileNotFoundError as exc:
            raise DataStoreError(f"document record not found: {document_id}") from exc

    def tokenized_path(self, tokenized_id: str) -> Path:
        _require_sha256(tokenized_id, "tokenized build IDs")
        return self.root / "tokenized" / tokenized_id

    def tokenized_ids(self) -> list[str]:
        return sorted(
            path.parent.name for path in (self.root / "tokenized").glob("*/manifest.json")
        )

    def evaluation_path(self, report_id: str) -> Path:
        _require_sha256(report_id, "evaluation report IDs")
        return self.root / "evaluations" / f"{report_id}.json"

    def evaluation_ids(self) -> list[str]:
        return sorted(path.stem for path in (self.root / "evaluations").glob("*.json"))

    def write_evaluation(self, payload: bytes) -> tuple[str, Path]:
        report_id = sha256_bytes(payload)
        target = self.evaluation_path(report_id)
        self._write_immutable(target, payload)
        return report_id, target

    def evaluation_report_path(self, report_id: str) -> Path:
        """Location of a sealed :mod:`epor.evaluation` suite report.

        Distinct from ``evaluations/`` (tokenizer fertility reports, addressed
        by the hash of their bytes): a suite report is addressed by the
        ``report_id`` sealed inside it, which excludes the ID field itself.
        """

        _require_sha256(report_id, "evaluation report IDs")
        return self.root / "evaluation-reports" / f"{report_id}.json"

    def evaluation_report_ids(self) -> list[str]:
        return sorted(path.stem for path in (self.root / "evaluation-reports").glob("*.json"))

    def write_evaluation_report(self, report_id: str, payload: bytes) -> Path:
        target = self.evaluation_report_path(report_id)
        self._write_immutable(target, payload)
        return target

    def scaling_report_directory(self, report_id: str) -> Path:
        """Directory of one sealed :mod:`epor.scaling` report.

        A scaling report's ID excludes its ``measured`` section (timings,
        host snapshot), so repeated measurements of the same experiment share
        an ID; each is stored as ``<content_sha256>.json`` beneath it.
        """

        _require_sha256(report_id, "scaling report IDs")
        return self.root / "scaling-reports" / report_id

    def scaling_report_ids(self) -> list[str]:
        return sorted(
            path.name for path in (self.root / "scaling-reports").iterdir() if path.is_dir()
        )

    def write_scaling_report(self, report_id: str, content_sha256: str, payload: bytes) -> Path:
        _require_sha256(content_sha256, "scaling report content hashes")
        target = self.scaling_report_directory(report_id) / f"{content_sha256}.json"
        self._write_immutable(target, payload)
        return target

    def certificate_path(self, certificate_id: str) -> Path:
        """A v0.0.8 validated-context certificate (``epor.context.status``)."""

        _require_sha256(certificate_id, "context certificate IDs")
        return self.root / "context-certificates" / f"{certificate_id}.json"

    def write_certificate(self, certificate_id: str, payload: bytes) -> Path:
        target = self.certificate_path(certificate_id)
        self._write_immutable(target, payload)
        return target

    def certificate_ids(self) -> list[str]:
        return sorted(path.stem for path in (self.root / "context-certificates").glob("*.json"))

    def alignment_path(self, relative: str) -> Path:
        """A path beneath ``alignment/``, the v0.0.7 alignment-record layer."""

        parts = Path(relative).parts
        if not parts or relative.startswith("/") or ".." in parts:
            raise DataStoreError("alignment paths must remain inside the alignment layer")
        return self.root / "alignment" / relative

    def write_alignment_file(self, relative: str, payload: bytes) -> Path:
        target = self.alignment_path(relative)
        self._write_immutable(target, payload)
        return target

    @contextmanager
    def staging_directory(self, prefix: str) -> Iterator[Path]:
        """A private scratch directory under ``tmp`` that is always removed on exit.

        Multi-file artifacts are written here in full and then moved into their
        content-addressed place with ``publish_directory``, so a failure or
        cancellation never leaves a partial artifact at a final address.
        """

        path = Path(tempfile.mkdtemp(prefix=prefix, dir=self.root / "tmp"))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def publish_directory(staging: Path, target: Path) -> bool:
        """Atomically rename a completed staging directory to ``target``.

        Returns ``False`` without touching anything when ``target`` already
        exists; the caller must then verify the existing artifact.
        """

        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.replace(staging, target)
        except OSError:
            if target.exists():
                return False
            raise
        return True

    def builds(self) -> Iterator[tuple[Path, DatasetBuild]]:
        for path in sorted((self.root / "builds").glob("*/*/manifest.json")):
            yield path, DatasetBuild.model_validate_json(path.read_bytes())

    def audit(self) -> DataAudit:
        registrations = self.registrations()
        registration_hashes = {item.id: sha256_model(item) for item in registrations}
        documents = self.documents()
        tombstones = self.tombstones()
        builds = list(self.builds())
        errors: list[str] = []

        for record in documents:
            raw = self.raw_path(record.raw_sha256)
            if not raw.exists() or sha256_file(raw) != record.raw_sha256:
                errors.append(f"{record.document_id}: raw content hash mismatch")
            if registration_hashes.get(record.source_id) != record.registration_sha256:
                errors.append(f"{record.document_id}: registration hash mismatch")
            if sha256_bytes(record.text.encode("utf-8")) != record.normalized_sha256:
                errors.append(f"{record.document_id}: normalized content hash mismatch")

        latest_entries: dict[str, tuple[datetime, str]] = {}
        for manifest_path, build in builds:
            candidate = (build.deterministic_timestamp, build.build_id)
            latest_entries[build.dataset_id] = max(
                latest_entries.get(build.dataset_id, candidate), candidate
            )
            for name, expected in build.artifacts.items():
                artifact = manifest_path.parent / name
                if not artifact.exists() or sha256_file(artifact) != expected:
                    errors.append(f"{build.build_id}: artifact hash mismatch for {name}")

        return DataAudit(
            registrations=len(registrations),
            documents=len(documents),
            admitted=sum(item.disposition == "admitted" for item in documents),
            quarantined=sum(item.disposition == "quarantined" for item in documents),
            tombstones=len(tombstones),
            builds=len(builds),
            integrity_errors=errors,
            latest_builds={dataset_id: item[1] for dataset_id, item in latest_entries.items()},
        )
