"""Safe allowlisted local job worker with cooperative cancellation."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine

from epor.research.catalog import load_catalog
from epor.research.service import sync_catalog
from epor.system import probe_system, safe_environment

from .database import create_session_factory, create_sqlite_engine, initialize_database
from .models import Job, JobStatus, JobType
from .schemas import TINY_CONTROL_CORPUS_MAX_BYTES
from .security import PathOutsideRootError, contained_path
from .service import CovenantBlockedError, InvalidTransitionError, JobService
from .settings import ControlSettings


class JobCancelled(RuntimeError):
    """Raised by a cooperative handler at a safe interruption boundary."""


class TinyCorpusLimitError(ValueError):
    """Raised before a control job can materialize an oversized corpus."""

    code = "tiny_corpus_limit_error"


@dataclass(slots=True)
class WorkerContext:
    """Narrow capability object exposed to allowlisted in-process handlers."""

    service: JobService
    job_id: str
    worker_id: str
    _cancelled: Event = field(default_factory=Event)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def request_cancellation(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled("job cancellation requested")

    def progress(
        self,
        value: float,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.raise_if_cancelled()
        payload = dict(details or {})
        if message:
            payload["message"] = message
        self.service.update_progress(
            self.job_id,
            self.worker_id,
            value,
            payload=payload,
        )

    def log(self, message: str, *, level: str = "info") -> None:
        self.service.emit_event(
            self.job_id,
            self.worker_id,
            "job.log",
            {"level": level, "message": message},
        )

    def register_artifact(
        self,
        path: str | Path,
        *,
        kind: str = "file",
        media_type: str | None = None,
    ) -> None:
        self.service.register_artifact(
            self.job_id,
            path,
            kind=kind,
            media_type=media_type,
        )


JobHandler = Callable[[WorkerContext, dict[str, Any]], dict[str, Any] | None]


def _json_safe(value: Any) -> Any:
    """Normalize dataclasses and string enums before SQLite JSON persistence."""

    return json.loads(
        json.dumps(
            value,
            default=lambda item: item.value if isinstance(item, Enum) else str(item),
        )
    )


def _system_probe(context: WorkerContext, _spec: dict[str, Any]) -> dict[str, Any]:
    context.progress(0.2, message="Inspecting local capabilities")
    result = probe_system(context.service.settings.project_root).to_dict()
    context.progress(1.0, message="System probe complete")
    return {"hardware": result, "environment": safe_environment()}


def _research_sync(context: WorkerContext, spec: dict[str, Any]) -> dict[str, Any]:
    settings = context.service.settings
    assert settings.research_catalog_path is not None
    requested = spec.get("source_ids")
    if requested is None:
        source_ids = [entry.id for entry in load_catalog(settings.research_catalog_path).sources]
    else:
        source_ids = list(requested)
    results: list[dict[str, Any]] = []
    total = max(len(source_ids), 1)
    if not source_ids:
        context.progress(1.0, message="Research catalog is empty")
    for index, source_id in enumerate(source_ids, start=1):
        context.raise_if_cancelled()
        context.log(f"Synchronizing allowlisted research source {source_id}")
        source_results = sync_catalog(
            settings.research_catalog_path,
            project_root=settings.project_root,
            offline=bool(spec.get("offline", True)),
            source_ids=[source_id],
        )
        results.extend(_json_safe(asdict(item)) for item in source_results)
        context.progress(index / total, message=f"Processed {index} of {total} sources")
    return {"sources": results, "count": len(results)}


def _training_handler(name: str) -> JobHandler:
    """Load reference training lazily so the default control environment stays CPU-safe."""

    def run(context: WorkerContext, spec: dict[str, Any]) -> dict[str, Any] | None:
        try:
            corpus = contained_path(
                context.service.settings.project_root,
                str(spec.get("corpus_path", "fixtures/tiny_corpus.txt")),
                must_exist=True,
            )
            if not corpus.is_file():
                raise TinyCorpusLimitError("tiny control corpus must be a regular file")
            if corpus.stat().st_size > TINY_CONTROL_CORPUS_MAX_BYTES:
                raise TinyCorpusLimitError("tiny control corpus exceeds the 1 MiB v0.0.1 limit")
        except (FileNotFoundError, OSError, PathOutsideRootError) as exc:
            raise TinyCorpusLimitError(
                "tiny control corpus is unavailable or outside the project root"
            ) from exc
        try:
            from epor.training.control import run_control_job
        except ImportError as exc:
            raise RuntimeError(
                "reference training support is unavailable; sync the train-cpu profile"
            ) from exc
        result = run_control_job(name, spec, context)
        if result is None or isinstance(result, dict):
            return result
        raise TypeError("training control handler must return a mapping or None")

    return run


DEFAULT_HANDLERS: dict[JobType, JobHandler] = {
    JobType.SYSTEM_PROBE: _system_probe,
    JobType.RESEARCH_SYNC: _research_sync,
    JobType.TINY_TRAIN: _training_handler("tiny_train"),
    JobType.TINY_EVAL: _training_handler("tiny_eval"),
}


class JobWorker:
    """Claim and execute only the four v0.0.1 job types.

    No client-provided command, executable, URL, or environment value is ever
    evaluated.  Long handlers receive a cooperative context while the owner
    thread maintains heartbeats and observes cancellation requests.
    """

    def __init__(
        self,
        settings: ControlSettings | None = None,
        *,
        service: JobService | None = None,
        handlers: Mapping[JobType, JobHandler] | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.settings = settings or (
            service.settings if service else ControlSettings.from_environment()
        )
        self._engine: Engine | None = None
        if service is None:
            self.settings.prepare_directories()
            assert self.settings.database_path is not None
            self._engine = create_sqlite_engine(self.settings.database_path)
            if self.settings.create_schema:
                initialize_database(self._engine)
            service = JobService(create_session_factory(self._engine), self.settings)
        self.service = service
        self.handlers = dict(DEFAULT_HANDLERS if handlers is None else handlers)
        missing = set(JobType).difference(self.handlers)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"worker handlers missing allowlisted job type(s): {names}")
        self.worker_id = worker_id or f"{socket.gethostname()}:{uuid4()}"

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> JobWorker:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def run_once(self) -> Job | None:
        """Reconcile abandoned work, then execute at most one queued job."""

        self.service.reconcile_stale()
        claimed = self.service.claim_next(self.worker_id)
        if claimed is None:
            return None

        try:
            active = self.service.mark_running(claimed.id, self.worker_id)
        except InvalidTransitionError:
            refreshed = self.service.get_job(claimed.id)
            if refreshed.status is JobStatus.CANCELLING:
                return self.service.complete_success(claimed.id, self.worker_id)
            raise

        # Re-resolve at dispatch rather than trusting admission. A queued job may
        # have been admitted under an earlier covenant, and the worker is the
        # last point before the work actually runs.
        try:
            self.service.admit(active.type)
        except CovenantBlockedError as exc:
            return self.service.complete_failure(
                active.id,
                self.worker_id,
                code=exc.code,
                message=exc.message,
            )

        context = WorkerContext(self.service, active.id, self.worker_id)
        handler = self.handlers[active.type]
        future: Future[dict[str, Any] | None]
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="epor-job")
        try:
            future = executor.submit(handler, context, dict(active.spec))
            while True:
                try:
                    result = future.result(timeout=self.settings.poll_interval_seconds)
                    break
                except FutureTimeout:
                    if self.service.cancellation_requested(active.id, self.worker_id):
                        context.request_cancellation()
                    self.service.heartbeat(active.id, self.worker_id)

            if self.service.cancellation_requested(active.id, self.worker_id):
                context.request_cancellation()
            return self.service.complete_success(active.id, self.worker_id, result)
        except JobCancelled:
            # Cancellation is represented by the lifecycle, not as a job failure.
            current = self.service.get_job(active.id)
            if current.status is not JobStatus.CANCELLING:
                self.service.cancel(active.id)
            return self.service.complete_success(active.id, self.worker_id)
        except Exception as exc:  # worker boundary intentionally normalizes failures
            return self.service.complete_failure(
                active.id,
                self.worker_id,
                code=getattr(exc, "code", type(exc).__name__),
                message=str(exc) or "allowlisted job handler failed",
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def run_forever(self, stop_event: Event | None = None) -> None:
        """Poll until the process receives a cooperative stop request."""

        stop = stop_event or Event()
        while not stop.is_set():
            job = self.run_once()
            if job is None:
                stop.wait(self.settings.poll_interval_seconds)
