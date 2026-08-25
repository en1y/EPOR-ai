"""Immutable job manifests and hash-chained append-only event truth store."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from .models import Job, JobEvent


class TruthStoreError(RuntimeError):
    """Raised when file-backed control truth is corrupt or inconsistent."""


@dataclass(frozen=True, slots=True)
class TruthJob:
    manifest: dict[str, Any]
    events: tuple[dict[str, Any], ...]


def utc_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TruthStoreError("truth timestamp must be a string or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TruthStoreError("truth timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise TruthStoreError("truth timestamp must include UTC offset")
    return parsed.astimezone(UTC)


def job_snapshot(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "type": job.type.value,
        "status": job.status.value,
        "spec": job.spec,
        "result": job.result,
        "progress": job.progress,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "worker_id": job.worker_id,
        "event_cursor": job.event_cursor,
        "retry_of_id": job.retry_of_id,
        "created_at": utc_text(job.created_at),
        "updated_at": utc_text(job.updated_at),
        "started_at": utc_text(job.started_at),
        "finished_at": utc_text(job.finished_at),
        "heartbeat_at": utc_text(job.heartbeat_at),
        "cancel_requested_at": utc_text(job.cancel_requested_at),
    }


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class JobTruthStore:
    """File authority from which the disposable SQLite index can be rebuilt."""

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve() / "_control" / "jobs"

    @staticmethod
    def _validated_id(job_id: str) -> str:
        try:
            parsed = UUID(job_id)
        except ValueError as exc:
            raise TruthStoreError("job truth ID is not a UUID") from exc
        if str(parsed) != job_id:
            raise TruthStoreError("job truth ID is not canonical")
        return job_id

    def _job_dir(self, job_id: str) -> Path:
        return self.root / self._validated_id(job_id)

    def ensure_manifest(self, job: Job) -> None:
        job_dir = self._job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        body = {
            "schema_version": 1,
            "id": job.id,
            "type": job.type.value,
            "spec": job.spec,
            "retry_of_id": job.retry_of_id,
            "created_at": utc_text(job.created_at),
        }
        document = {**body, "manifest_sha256": _digest(body)}
        encoded = _canonical(document) + b"\n"
        path = job_dir / "manifest.json"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise TruthStoreError("cannot read existing job manifest") from exc
            if existing != encoded:
                raise TruthStoreError(
                    "immutable job manifest does not match database job"
                ) from None
            return
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            path.unlink(missing_ok=True)
            raise TruthStoreError("cannot persist immutable job manifest") from exc

    def append_event(self, job: Job, event: JobEvent) -> None:
        self.ensure_manifest(job)
        path = self._job_dir(job.id) / "events.jsonl"
        try:
            with path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.seek(0)
                existing = self._decode_events(stream.read(), job.id)
                expected = len(existing) + 1
                if event.sequence != expected:
                    raise TruthStoreError(
                        f"event sequence {event.sequence} does not follow {expected}"
                    )
                previous = existing[-1]["record_sha256"] if existing else None
                body = {
                    "schema_version": 1,
                    "job_id": job.id,
                    "sequence": event.sequence,
                    "kind": event.kind,
                    "payload": event.payload,
                    "created_at": utc_text(event.created_at),
                    "state": job_snapshot(job),
                    "previous_sha256": previous,
                }
                record = {**body, "record_sha256": _digest(body)}
                stream.seek(0, os.SEEK_END)
                stream.write(_canonical(record) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TruthStoreError("cannot append job event truth") from exc

    def load_jobs(self) -> list[TruthJob]:
        if not self.root.is_dir():
            return []
        jobs: list[TruthJob] = []
        for job_dir in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not job_dir.is_dir():
                continue
            job_id = self._validated_id(job_dir.name)
            manifest_path = job_dir / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                event_bytes = (job_dir / "events.jsonl").read_bytes()
            except (OSError, json.JSONDecodeError) as exc:
                raise TruthStoreError(f"cannot load truth for job {job_id}") from exc
            if not isinstance(manifest, dict):
                raise TruthStoreError("job manifest must contain an object")
            expected_hash = manifest.get("manifest_sha256")
            body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            if expected_hash != _digest(body) or manifest.get("id") != job_id:
                raise TruthStoreError("job manifest checksum or identity is invalid")
            events = self._decode_events(event_bytes, job_id)
            if not events:
                raise TruthStoreError("job truth must contain at least its queued event")
            jobs.append(TruthJob(manifest=manifest, events=tuple(events)))
        return sorted(jobs, key=lambda item: str(item.manifest["created_at"]))

    @staticmethod
    def _decode_events(payload: bytes, job_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        previous: str | None = None
        for sequence, raw_line in enumerate(payload.splitlines(), start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TruthStoreError("job event log contains invalid JSON") from exc
            if not isinstance(record, dict):
                raise TruthStoreError("job event record must be an object")
            claimed_hash = record.get("record_sha256")
            body = {key: value for key, value in record.items() if key != "record_sha256"}
            if (
                record.get("job_id") != job_id
                or record.get("sequence") != sequence
                or record.get("previous_sha256") != previous
                or claimed_hash != _digest(body)
            ):
                raise TruthStoreError("job event hash chain or sequence is invalid")
            if not isinstance(claimed_hash, str):
                raise TruthStoreError("job event hash is missing")
            previous = claimed_hash
            records.append(record)
        return records
