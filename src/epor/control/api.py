"""Loopback-only FastAPI application for the EPOR local console."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from epor import __version__
from epor.models.config import load_model_config
from epor.research.catalog import load_catalog
from epor.research.models import CatalogError
from epor.research.service import verify_catalog
from epor.system import probe_system

from .database import create_session_factory, create_sqlite_engine, initialize_database
from .documents import DocumentTooLargeError, build_index, read_document
from .models import JobStatus, JobType
from .schemas import (
    JOB_SPEC_MODELS,
    ArtifactList,
    ArtifactRead,
    CapabilityRead,
    DocumentList,
    DocumentRead,
    DocumentSummary,
    ErrorEnvelope,
    EventList,
    EventRead,
    HealthRead,
    JobCreate,
    JobList,
    JobRead,
    JobTypeList,
    JobTypeRead,
    ModelFamilyList,
    ModelFamilyRead,
    ResearchCatalogRead,
    ResearchSourceRead,
)
from .security import redact, redact_text
from .service import ControlError, JobService
from .settings import ControlSettings

API_PREFIX = "/api/v1"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
JobTypeQuery = Annotated[JobType | None, Query(alias="type")]
JobLimitQuery = Annotated[int, Query(ge=1, le=500)]
EventAfterQuery = Annotated[int, Query(ge=0)]
EventLimitQuery = Annotated[int, Query(ge=1, le=1_000)]
_JOB_TYPE_COPY: dict[JobType, tuple[str, str]] = {
    JobType.SYSTEM_PROBE: (
        "System probe",
        "Refresh local CPU, memory, toolchain, and accelerator capability facts.",
    ),
    JobType.RESEARCH_SYNC: (
        "Research sync",
        "Verify the local paper archive or synchronize only catalog-allowlisted sources.",
    ),
    JobType.TINY_TRAIN: (
        "Tiny pretraining",
        "Run the deterministic CPU reference trainer with bounded debug settings.",
    ),
    JobType.TINY_EVAL: (
        "Tiny evaluation",
        "Evaluate a contained reference checkpoint against the generated fixture corpus.",
    ),
}


def _get_request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid4())


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        code=code,
        message=redact_text(message),
        details=redact(details),
        request_id=_get_request_id(request),
    )
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(envelope),
        headers={"X-Request-ID": envelope.request_id, "Cache-Control": "no-store"},
    )


def _safe_validation_details(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Return field diagnostics without reflecting request values or exceptions."""

    return [
        {key: redact(value) for key, value in error.items() if key not in {"input", "ctx", "url"}}
        for error in errors
        if isinstance(error, Mapping)
    ]


def _research_payload(settings: ControlSettings) -> ResearchCatalogRead:
    assert settings.research_catalog_path is not None
    try:
        catalog = load_catalog(settings.research_catalog_path)
        verification = {
            result.source_id: result
            for result in verify_catalog(
                settings.research_catalog_path,
                project_root=settings.project_root,
            )
        }
    except (CatalogError, OSError, ValueError):
        return ResearchCatalogRead(
            schema_version=None,
            available=False,
            sources=[],
            count=0,
            error="The tracked research catalog is unavailable or invalid.",
        )

    sources: list[ResearchSourceRead] = []
    for source in catalog.sources:
        result = verification.get(source.id)
        raw_status = str(result.status) if result is not None else "unknown"
        sync_status = raw_status.rsplit(".", maxsplit=1)[-1]
        if sync_status not in {"verified", "missing", "error"}:
            sync_status = "unknown"
        sources.append(
            ResearchSourceRead(
                id=source.id,
                title=source.title,
                organization=source.organization,
                publication_date=source.publication_date,
                media_type=source.media_type,
                canonical_url=source.canonical_url,
                tags=list(source.tags),
                evidence_classes=[item.value for item in source.evidence_classes],
                sync_status=sync_status,  # type: ignore[arg-type]
                sha256=result.sha256 if result is not None else None,
                size_bytes=result.size_bytes if result is not None else None,
                status_message=redact_text(result.message) if result is not None else "",
            )
        )
    return ResearchCatalogRead(
        schema_version=catalog.schema_version,
        available=True,
        sources=sources,
        count=len(sources),
    )


def _models_payload(settings: ControlSettings) -> ModelFamilyList:
    items: list[ModelFamilyRead] = []
    for family in ("alpha", "beta", "gamma"):
        path = settings.project_root / "configs" / "models" / f"epor-{family}.yaml"
        if not path.is_file():
            continue
        config = load_model_config(path)
        # These cards are recipes and plans.  v0.0.1 deliberately has no model metrics.
        items.append(
            ModelFamilyRead.model_validate(
                {
                    **config.model_dump(mode="json"),
                    "status": "planned",
                    "metrics": None,
                }
            )
        )
    return ModelFamilyList(items=items, count=len(items))


def _router(service: JobService, settings: ControlSettings) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=HealthRead, tags=["system"])
    async def health() -> HealthRead:
        try:
            with service._sessions() as session:
                session.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return HealthRead(version=__version__)

    @router.get("/capabilities", response_model=CapabilityRead, tags=["system"])
    async def capabilities() -> CapabilityRead:
        return CapabilityRead(
            bind_host=settings.host,
            allowed_origin=settings.allowed_origin,
            job_types=list(JobType),
            hardware=probe_system(settings.project_root).to_dict(),
        )

    @router.get("/job-types", response_model=JobTypeList, tags=["jobs"])
    async def job_types() -> JobTypeList:
        return JobTypeList(
            items=[
                JobTypeRead(
                    type=job_type,
                    title=_JOB_TYPE_COPY[job_type][0],
                    description=_JOB_TYPE_COPY[job_type][1],
                    schema=model.model_json_schema(by_alias=True),
                )
                for job_type, model in JOB_SPEC_MODELS.items()
            ]
        )

    @router.get("/research", response_model=ResearchCatalogRead, tags=["research"])
    async def research() -> ResearchCatalogRead:
        return _research_payload(settings)

    @router.get("/models", response_model=ModelFamilyList, tags=["models"])
    async def models() -> ModelFamilyList:
        return _models_payload(settings)

    @router.get("/documents", response_model=DocumentList, tags=["documents"])
    async def documents() -> DocumentList:
        items = [
            DocumentSummary(slug=item.slug, title=item.title, group=item.group)
            for item in build_index(settings.project_root).values()
        ]
        return DocumentList(items=items, count=len(items))

    @router.get("/documents/{slug:path}", response_model=DocumentRead, tags=["documents"])
    async def document(slug: str) -> DocumentRead:
        # Exact lookup in the freshly built index.  An unindexed slug is simply
        # absent, so path traversal and symlink escapes have nothing to reach.
        found = build_index(settings.project_root).get(slug)
        if found is None:
            raise HTTPException(status_code=404, detail="document not found")
        try:
            markdown = read_document(found)
        except DocumentTooLargeError as exc:
            raise HTTPException(status_code=413, detail="document exceeds the read limit") from exc
        except OSError as exc:
            raise HTTPException(status_code=404, detail="document not found") from exc
        return DocumentRead(
            slug=found.slug,
            title=found.title,
            group=found.group,
            markdown=markdown,
        )

    @router.post("/jobs", response_model=JobRead, status_code=201, tags=["jobs"])
    async def create_job(payload: JobCreate) -> JobRead:
        return JobRead.model_validate(service.create_job(payload.type, payload.spec))

    @router.get("/jobs", response_model=JobList, tags=["jobs"])
    async def list_jobs(
        status: JobStatus | None = None,
        job_type: JobTypeQuery = None,
        limit: JobLimitQuery = 100,
    ) -> JobList:
        jobs = service.list_jobs(status=status, job_type=job_type, limit=limit)
        return JobList(items=[JobRead.model_validate(job) for job in jobs], count=len(jobs))

    @router.get("/jobs/{job_id}", response_model=JobRead, tags=["jobs"])
    async def get_job(job_id: str) -> JobRead:
        return JobRead.model_validate(service.get_job(job_id))

    @router.post("/jobs/{job_id}/cancel", response_model=JobRead, tags=["jobs"])
    async def cancel_job(job_id: str) -> JobRead:
        return JobRead.model_validate(service.cancel(job_id))

    @router.post(
        "/jobs/{job_id}/retry",
        response_model=JobRead,
        status_code=201,
        tags=["jobs"],
    )
    async def retry_job(job_id: str) -> JobRead:
        return JobRead.model_validate(service.retry(job_id))

    @router.get("/jobs/{job_id}/events", response_model=EventList, tags=["jobs"])
    async def job_events(
        job_id: str,
        after: EventAfterQuery = 0,
        limit: EventLimitQuery = 500,
    ) -> EventList:
        events = service.list_events(job_id, after=after, limit=limit)
        next_after = events[-1].sequence if events else after
        return EventList(
            items=[EventRead.model_validate(event) for event in events],
            next_after=next_after,
        )

    @router.get("/jobs/{job_id}/artifacts", response_model=ArtifactList, tags=["jobs"])
    async def job_artifacts(job_id: str) -> ArtifactList:
        return ArtifactList(
            items=[
                ArtifactRead.model_validate(artifact) for artifact in service.list_artifacts(job_id)
            ]
        )

    return router


def create_app(settings: ControlSettings | None = None) -> FastAPI:
    """Create a local control app without starting a network listener.

    The caller is responsible for passing :attr:`ControlSettings.uvicorn_kwargs`
    to Uvicorn; settings reject every non-loopback bind address before startup.
    """

    active_settings = settings or ControlSettings.from_environment()
    active_settings.prepare_directories()
    assert active_settings.database_path is not None
    engine = create_sqlite_engine(active_settings.database_path)
    if active_settings.create_schema:
        initialize_database(engine)
    sessions = create_session_factory(engine)
    service = JobService(sessions, active_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="EPOR local control plane",
        version=__version__,
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    app.state.control_settings = active_settings
    app.state.engine = engine
    app.state.session_factory = sessions
    app.state.job_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[active_settings.allowed_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_identity(request: Request, call_next: Any) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(ControlError)
    async def control_error(request: Request, exc: ControlError) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            code="request_validation_error",
            message="request validation failed",
            details=_safe_validation_details(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed"
        return _error_response(
            request,
            status_code=exc.status_code,
            code="http_error",
            message=message,
            details=None if isinstance(exc.detail, str) else exc.detail,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(
            request,
            status_code=500,
            code="internal_error",
            message="an unexpected local control error occurred",
        )

    app.include_router(_router(service, active_settings))
    return app
