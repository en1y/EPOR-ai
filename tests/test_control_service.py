from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, timedelta
from pathlib import Path
from threading import Barrier, Event, Lock, Thread

import pytest
from sqlalchemy import text

from epor.control.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from epor.control.models import JobStatus, JobType, utc_now
from epor.control.schemas import TINY_CONTROL_CORPUS_MAX_BYTES
from epor.control.security import (
    PathOutsideRootError,
    contained_path,
    redact,
)
from epor.control.service import ArtifactPathError, InvalidJobSpecError, JobService
from epor.control.settings import ControlSettings
from epor.control.worker import DEFAULT_HANDLERS, JobWorker, WorkerContext


def make_service(
    runtime: Path,
    *,
    project_root: Path | None = None,
    stale_after_seconds: float = 1.0,
) -> tuple[ControlSettings, JobService]:
    root = project_root or runtime
    settings = ControlSettings(
        project_root=root,
        database_path=runtime / "control.sqlite3",
        artifact_root=runtime / "artifacts",
        research_catalog_path=root / "research" / "catalog.yaml",
        stale_after_seconds=stale_after_seconds,
        poll_interval_seconds=0.005,
    )
    settings.prepare_directories()
    assert settings.database_path is not None
    engine = create_sqlite_engine(settings.database_path)
    initialize_database(engine)
    return settings, JobService(create_session_factory(engine), settings)


def test_loopback_settings_path_containment_and_redaction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        ControlSettings(project_root=tmp_path, host="0.0.0.0")
    with pytest.raises(ValueError, match="exact loopback"):
        ControlSettings(project_root=tmp_path, allowed_origin="http://localhost:5173")
    with pytest.raises(ValueError, match="exact loopback"):
        ControlSettings(
            project_root=tmp_path,
            allowed_origin="http://127.0.0.1:5173@evil.example",
        )

    root = tmp_path / "root"
    root.mkdir()
    assert contained_path(root, "nested/file.txt") == root / "nested" / "file.txt"
    with pytest.raises(PathOutsideRootError):
        contained_path(root, "../escape.txt")
    assert redact({"api_token": "value", "message": "Bearer abcdefghijklmnop"}) == {
        "api_token": "[REDACTED]",
        "message": "Bearer [REDACTED]",
    }


def test_sqlite_wal_and_transactional_lifecycle(tmp_path: Path) -> None:
    settings, service = make_service(tmp_path)
    assert settings.database_path is not None
    engine = create_sqlite_engine(settings.database_path)
    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA journal_mode")) == "wal"
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1

    queued = service.create_job(JobType.SYSTEM_PROBE, {})
    assert queued.status is JobStatus.QUEUED
    persisted_queued = service.get_job(queued.id)
    assert persisted_queued.created_at.tzinfo is UTC
    assert persisted_queued.updated_at.tzinfo is UTC
    assert [(event.sequence, event.kind) for event in service.list_events(queued.id)] == [
        (1, "job.queued")
    ]

    claimed = service.claim_next("worker-a")
    assert claimed is not None and claimed.id == queued.id
    assert service.claim_next("worker-b") is None
    service.mark_running(queued.id, "worker-a")
    service.update_progress(
        queued.id,
        "worker-a",
        0.4,
        {"message": "token=abcdefghijklmnop", "password": "do-not-store"},
    )
    assert service.cancel(queued.id).status is JobStatus.CANCELLING
    assert service.cancel(queued.id).status is JobStatus.CANCELLING
    finished = service.complete_success(queued.id, "worker-a", {"ignored": True})
    assert finished.status is JobStatus.CANCELLED

    retried = service.retry(queued.id)
    assert retried.status is JobStatus.QUEUED
    assert retried.retry_of_id == queued.id
    events = service.list_events(queued.id)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    progress = next(event for event in events if event.kind == "job.progress")
    assert progress.payload["password"] == "[REDACTED]"
    assert "abcdefghijklmnop" not in str(progress.payload)
    assert all(event.created_at.tzinfo is UTC for event in events)


def test_concurrent_cancel_and_progress_event_sequences_are_serialized(
    tmp_path: Path,
) -> None:
    _, service = make_service(tmp_path)

    def running_job() -> str:
        job = service.create_job(JobType.SYSTEM_PROBE, {})
        claimed = service.claim_next("race-worker")
        assert claimed is not None and claimed.id == job.id
        service.mark_running(job.id, "race-worker")
        return job.id

    # Repeated synchronized starts exercise separate pooled SQLite connections.
    # Any thread exception is re-raised by Future.result(), making the regression
    # deterministic rather than relying on an unobserved background warning.
    for _round in range(20):
        cancel_id = running_job()
        cancel_barrier = Barrier(3)

        def concurrent_cancel(
            job_id: str = cancel_id,
            barrier: Barrier = cancel_barrier,
        ) -> JobStatus:
            barrier.wait(timeout=5)
            return service.cancel(job_id).status

        with ThreadPoolExecutor(max_workers=2) as executor:
            cancel_futures = [executor.submit(concurrent_cancel) for _ in range(2)]
            cancel_barrier.wait(timeout=5)
            assert [future.result(timeout=5) for future in cancel_futures] == [
                JobStatus.CANCELLING,
                JobStatus.CANCELLING,
            ]

        cancel_events = service.list_events(cancel_id)
        assert [event.kind for event in cancel_events].count("job.cancelling") == 1
        assert [event.sequence for event in cancel_events] == list(range(1, len(cancel_events) + 1))

        progress_id = running_job()
        progress_barrier = Barrier(3)

        def concurrent_progress(
            job_id: str = progress_id,
            barrier: Barrier = progress_barrier,
        ) -> JobStatus:
            barrier.wait(timeout=5)
            return service.update_progress(
                job_id,
                "race-worker",
                0.5,
                {"message": "concurrent progress"},
            ).status

        def cancel_during_progress(
            job_id: str = progress_id,
            barrier: Barrier = progress_barrier,
        ) -> JobStatus:
            barrier.wait(timeout=5)
            return service.cancel(job_id).status

        with ThreadPoolExecutor(max_workers=2) as executor:
            progress_future = executor.submit(concurrent_progress)
            cancel_future = executor.submit(cancel_during_progress)
            progress_barrier.wait(timeout=5)
            assert progress_future.result(timeout=5) in {
                JobStatus.RUNNING,
                JobStatus.CANCELLING,
            }
            assert cancel_future.result(timeout=5) is JobStatus.CANCELLING

        progress_events = service.list_events(progress_id)
        assert [event.kind for event in progress_events].count("job.progress") == 1
        assert [event.kind for event in progress_events].count("job.cancelling") == 1
        assert [event.sequence for event in progress_events] == list(
            range(1, len(progress_events) + 1)
        )
        assert service.get_job(progress_id).status is JobStatus.CANCELLING


def test_simultaneous_workers_claim_and_execute_a_queued_job_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, service = make_service(tmp_path)
    queued = service.create_job(JobType.SYSTEM_PROBE, {})
    claim_barrier = Barrier(3)
    original_claim_next = service.claim_next

    def synchronized_claim(worker_id: str) -> object:
        claim_barrier.wait(timeout=5)
        return original_claim_next(worker_id)

    monkeypatch.setattr(service, "claim_next", synchronized_claim)

    executions: list[tuple[str, str]] = []
    execution_lock = Lock()

    def record_execution(
        context: WorkerContext,
        _spec: dict[str, object],
    ) -> dict[str, str]:
        with execution_lock:
            executions.append((context.worker_id, context.job_id))
        return {"executed_by": context.worker_id}

    handlers = dict(DEFAULT_HANDLERS)
    handlers[JobType.SYSTEM_PROBE] = record_execution
    workers = [
        JobWorker(
            settings,
            service=service,
            handlers=handlers,
            worker_id=worker_id,
        )
        for worker_id in ("worker-a", "worker-b")
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker.run_once) for worker in workers]
        claim_barrier.wait(timeout=5)
        results = [future.result(timeout=5) for future in futures]

    completed = [result for result in results if result is not None]
    assert len(completed) == 1
    assert completed[0].id == queued.id
    assert completed[0].status is JobStatus.SUCCEEDED
    assert executions == [(completed[0].worker_id, queued.id)]

    persisted = service.get_job(queued.id)
    assert persisted.status is JobStatus.SUCCEEDED
    assert persisted.worker_id == completed[0].worker_id
    assert persisted.result == {"executed_by": persisted.worker_id}
    events = service.list_events(queued.id)
    assert [event.kind for event in events] == [
        "job.queued",
        "job.starting",
        "job.running",
        "job.succeeded",
    ]
    assert events[1].payload == {"worker_id": persisted.worker_id}
    assert events[2].payload == {"worker_id": persisted.worker_id}


def test_stale_reconciliation_and_artifact_checksums(tmp_path: Path) -> None:
    settings, service = make_service(tmp_path, stale_after_seconds=0.1)
    job = service.create_job(JobType.SYSTEM_PROBE, {})
    service.claim_next("worker-a")
    service.mark_running(job.id, "worker-a")
    now = utc_now()
    with service._sessions() as session, session.begin():
        persisted = session.get(type(job), job.id)
        assert persisted is not None
        persisted.heartbeat_at = now - timedelta(seconds=5)
    assert service.reconcile_stale(now=now) == [job.id]
    assert service.get_job(job.id).status is JobStatus.INTERRUPTED

    assert settings.artifact_root is not None
    artifact_path = settings.artifact_root / "reports" / "result.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
    artifact = service.register_artifact(job.id, artifact_path, kind="report")
    assert artifact.path == "reports/result.json"
    assert len(artifact.sha256) == 64
    with pytest.raises(ArtifactPathError):
        service.register_artifact(job.id, tmp_path.parent / "outside.txt")


def test_file_truth_rebuilds_a_fresh_sqlite_index_deterministically(tmp_path: Path) -> None:
    settings, service = make_service(tmp_path)
    job = service.create_job(JobType.SYSTEM_PROBE, {})
    service.claim_next("truth-worker")
    service.mark_running(job.id, "truth-worker")
    service.update_progress(job.id, "truth-worker", 0.75, {"message": "three quarters"})
    service.complete_success(job.id, "truth-worker", {"probe": "complete"})
    assert settings.artifact_root is not None
    report = settings.artifact_root / "reports" / "probe.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"probe": "complete"}\n', encoding="utf-8")
    service.register_artifact(job.id, report, kind="probe-report")
    cancelled_source = service.create_job(JobType.SYSTEM_PROBE, {})
    service.cancel(cancelled_source.id)
    retried = service.retry(cancelled_source.id)

    truth_dir = settings.artifact_root / "_control" / "jobs" / job.id
    manifest_before = (truth_dir / "manifest.json").read_bytes()
    truth_lines = (truth_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(truth_lines) == len(service.list_events(job.id))

    rebuilt_settings = ControlSettings(
        project_root=tmp_path,
        database_path=tmp_path / "rebuilt.sqlite3",
        artifact_root=settings.artifact_root,
        research_catalog_path=tmp_path / "research" / "catalog.yaml",
    )
    assert rebuilt_settings.database_path is not None
    rebuilt_engine = create_sqlite_engine(rebuilt_settings.database_path)
    initialize_database(rebuilt_engine)
    rebuilt = JobService(create_session_factory(rebuilt_engine), rebuilt_settings)

    rebuilt_job = rebuilt.get_job(job.id)
    assert rebuilt_job.status is JobStatus.SUCCEEDED
    assert rebuilt_job.progress == 1.0
    assert rebuilt_job.result == {"probe": "complete"}
    assert [event.kind for event in rebuilt.list_events(job.id)] == [
        event.kind for event in service.list_events(job.id)
    ]
    assert [artifact.sha256 for artifact in rebuilt.list_artifacts(job.id)] == [
        artifact.sha256 for artifact in service.list_artifacts(job.id)
    ]
    assert rebuilt.get_job(retried.id).retry_of_id == cancelled_source.id
    assert (truth_dir / "manifest.json").read_bytes() == manifest_before


def test_truth_reconciliation_preserves_newer_unlogged_heartbeat(tmp_path: Path) -> None:
    settings, service = make_service(tmp_path)
    job = service.create_job(JobType.SYSTEM_PROBE, {})
    service.claim_next("heartbeat-worker")
    service.mark_running(job.id, "heartbeat-worker")
    time.sleep(0.002)
    latest = service.heartbeat(job.id, "heartbeat-worker")
    assert latest.heartbeat_at is not None

    restarted = JobService(service._sessions, settings)
    restored = restarted.get_job(job.id)
    assert restored.status is JobStatus.RUNNING
    assert restored.heartbeat_at == latest.heartbeat_at
    assert restored.updated_at == latest.updated_at


def test_worker_cooperatively_cancels_active_handler(tmp_path: Path) -> None:
    settings, service = make_service(tmp_path)
    started = Event()

    def wait_for_cancel(context: object, _spec: dict[str, object]) -> None:
        started.set()
        while not context.cancelled:  # type: ignore[attr-defined]
            time.sleep(0.002)
        context.raise_if_cancelled()  # type: ignore[attr-defined]

    handlers = dict(DEFAULT_HANDLERS)
    handlers[JobType.SYSTEM_PROBE] = wait_for_cancel  # type: ignore[assignment]
    worker = JobWorker(settings, service=service, handlers=handlers, worker_id="worker-a")
    job = service.create_job(JobType.SYSTEM_PROBE, {})
    thread = Thread(target=worker.run_once)
    thread.start()
    assert started.wait(timeout=2)
    service.cancel(job.id)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert service.get_job(job.id).status is JobStatus.CANCELLED


@pytest.mark.parametrize("job_type", [JobType.TINY_TRAIN, JobType.TINY_EVAL])
def test_tiny_control_corpus_cap_boundary_and_worker_recheck(
    tmp_path: Path,
    job_type: JobType,
) -> None:
    settings, service = make_service(tmp_path)
    corpus = tmp_path / "boundary-corpus.bin"
    with corpus.open("wb") as stream:
        stream.truncate(TINY_CONTROL_CORPUS_MAX_BYTES)
    spec: dict[str, object] = {"corpus_path": corpus.name}
    if job_type is JobType.TINY_EVAL:
        spec["checkpoint_path"] = "queued-checkpoint.pt"

    queued = service.create_job(job_type, spec)
    assert queued.status is JobStatus.QUEUED

    # Grow the file after queue-time validation. The worker must reject it
    # before importing or invoking the training adapter, closing the TOCTOU gap.
    with corpus.open("r+b") as stream:
        stream.truncate(TINY_CONTROL_CORPUS_MAX_BYTES + 1)
    worker = JobWorker(settings, service=service, worker_id="corpus-cap-worker")
    failed = worker.run_once()
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.error_code == "tiny_corpus_limit_error"
    assert failed.error_message == "tiny control corpus exceeds the 1 MiB v0.0.1 limit"

    with pytest.raises(InvalidJobSpecError, match="exceeds the 1 MiB"):
        service.create_job(job_type, spec)


def test_worker_runs_tiny_train_and_eval_through_allowlisted_adapter(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    project_root = Path(__file__).resolve().parents[1]
    settings, service = make_service(tmp_path, project_root=project_root)
    worker = JobWorker(settings, service=service, worker_id="worker-train")
    train_job = service.create_job(
        JobType.TINY_TRAIN,
        {"max_steps": 1, "seed": 7, "output_name": "integration"},
    )
    trained = worker.run_once()
    assert trained is not None and trained.status is JobStatus.SUCCEEDED
    assert trained.result is not None
    checkpoint_path = str(trained.result["checkpoint_path"])
    assert {item.kind for item in service.list_artifacts(train_job.id)} >= {
        "checkpoint",
        "run-manifest",
        "training-metrics",
    }

    eval_job = service.create_job(
        JobType.TINY_EVAL,
        {"checkpoint_path": checkpoint_path, "max_batches": 1},
    )
    evaluated = worker.run_once()
    assert evaluated is not None and evaluated.status is JobStatus.SUCCEEDED
    assert {item.kind for item in service.list_artifacts(eval_job.id)} == {"evaluation-report"}
