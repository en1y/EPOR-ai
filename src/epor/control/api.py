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
from epor.research.models import CatalogError, IntegrityStatus
from epor.research.service import verify_catalog
from epor.system import probe_system

from .authority import (
    ANONYMOUS,
    SESSION_TTL,
    Actor,
    AuthorityStore,
    Escalation,
    EscalationState,
    Principal,
)
from .database import create_session_factory, create_sqlite_engine, initialize_database
from .documents import DocumentTooLargeError, build_index, read_document
from .errors import AuthenticationRequiredError, AuthorizationDeniedError, ControlError
from .models import JobStatus, JobType
from .safety_view import covenant_payload, declared_action_payload
from .schemas import (
    JOB_SPEC_MODELS,
    ActionList,
    ArtifactList,
    ArtifactRead,
    CapabilityRead,
    CovenantRead,
    DocumentList,
    DocumentRead,
    DocumentSummary,
    ErrorEnvelope,
    EscalationDecide,
    EscalationList,
    EscalationRead,
    EventList,
    EventRead,
    HealthRead,
    IdentityRead,
    IssuedCredentialRead,
    JobCreate,
    JobList,
    JobRead,
    JobTypeList,
    JobTypeRead,
    ModelFamilyList,
    ModelFamilyRead,
    OperatorCreate,
    OperatorList,
    OperatorRead,
    ResearchCatalogRead,
    ResearchSourceRead,
    SessionCreate,
)
from .security import redact, redact_text
from .service import JobService
from .settings import ControlSettings

SESSION_COOKIE = "epor_session"
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
                integrity=source.integrity.value,
                sync_status=sync_status,  # type: ignore[arg-type]
                raw_sha256=result.raw_sha256 if result is not None else None,
                tracked_raw_sha256=source.sha256,
                tracked_content_sha256=source.content_sha256,
                content_scope=(
                    source.content_scope.value
                    if source.integrity is IntegrityStatus.CONTENT_PINNED
                    else None
                ),
                content_profile=(
                    source.content_profile.value if source.content_profile is not None else None
                ),
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


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = value.strip()
    # Oversized credentials are dropped before hashing, so a large header
    # cannot be used to make authentication do work.
    return token if 0 < len(token) <= 512 else None


def _resolve_actor(request: Request, authority: AuthorityStore) -> tuple[Actor, bool]:
    """Identify the caller, and report whether a cookie carried the identity.

    Bearer beats cookie: a CLI or scripted call that presents a credential
    explicitly must not be silently reinterpreted as whatever browser session
    happens to be open in the same profile.
    """

    token = _bearer_token(request)
    if token is not None:
        principal = authority.authenticate(token)
        return (principal.actor() if principal else ANONYMOUS), False
    cookie_actor = authority.session_actor(request.cookies.get(SESSION_COOKIE))
    if cookie_actor is not None:
        return cookie_actor, True
    return ANONYMOUS, False


def _require_actor(request: Request, authority: AuthorityStore, settings: ControlSettings) -> Actor:
    """Authenticate a mutation and enforce the browser origin boundary.

    A cookie is ambient: the browser attaches it to whatever page asks. So a
    cookie-authenticated mutation must also prove it came from the one origin
    the console is served on. A bearer credential is not ambient and carries no
    such requirement, which is what lets the CLI and scripts work with no
    Origin header at all.
    """

    actor, via_cookie = _resolve_actor(request, authority)
    if not actor.is_authenticated:
        raise AuthenticationRequiredError("this request requires a verified operator identity")
    if via_cookie and request.headers.get("Origin") != settings.allowed_origin:
        raise AuthorizationDeniedError(
            "a session-authenticated request must come from the configured console origin"
        )
    return actor


def _identity_payload(actor: Actor, authority: AuthorityStore) -> IdentityRead:
    if not actor.is_authenticated:
        return IdentityRead(role="anonymous")
    try:
        expires_at = authority.principal(actor.id).expires_at
    except ControlError:
        # A CLI actor has no stored principal, and no expiry either.
        expires_at = None
    return IdentityRead(
        role=actor.role.value,  # type: ignore[arg-type]
        principal_id=actor.id,
        label=actor.label,
        scopes=sorted(actor.scopes),
        authenticated=True,
        expires_at=expires_at,
    )


def _operator_payload(principal: Principal) -> OperatorRead:
    return OperatorRead(
        id=principal.id,
        role=principal.role.value,
        label=principal.label,
        scopes=sorted(principal.scopes),
        active=principal.active(),
        created_at=principal.created_at,
        expires_at=principal.expires_at,
        revoked_at=principal.revoked_at,
        rotated_at=principal.rotated_at,
    )


def _escalation_payload(escalation: Escalation) -> EscalationRead:
    return EscalationRead(
        id=escalation.id,
        actor_id=escalation.actor_id,
        actor_role=escalation.actor_role.value,
        action_id=escalation.action_id,
        request_digest=escalation.request_digest,
        covenant_sha256=escalation.covenant_sha256,
        binding_priority=escalation.binding_priority,
        reasons=list(escalation.reasons),
        state=escalation.effective_state().value,  # type: ignore[arg-type]
        decided_by=escalation.decided_by,
        rationale=escalation.rationale,
        created_at=escalation.created_at,
        decided_at=escalation.decided_at,
        approval_expires_at=escalation.approval_expires_at,
        consumed_at=escalation.consumed_at,
        consumed_job_id=escalation.consumed_job_id,
    )


def _router(service: JobService, settings: ControlSettings) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)
    authority = service.authority

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

    # ---- identity -----------------------------------------------------

    @router.get("/auth/me", response_model=IdentityRead, tags=["auth"])
    async def whoami(request: Request) -> IdentityRead:
        actor, _ = _resolve_actor(request, authority)
        return _identity_payload(actor, authority)

    @router.post("/auth/session", response_model=IdentityRead, tags=["auth"])
    async def open_session(request: Request, payload: SessionCreate) -> Response:
        principal = authority.authenticate(payload.token)
        if principal is None:
            # One message for an unknown, expired, and revoked credential
            # alike: which of the three it was is not the caller's business.
            raise AuthenticationRequiredError("the supplied credential is not valid")
        origin = request.headers.get("Origin")
        if origin is not None and origin != settings.allowed_origin:
            raise AuthorizationDeniedError(
                "a session may only be opened from the configured console origin"
            )
        session = authority.open_session(principal)
        body = _identity_payload(principal.actor(), authority)
        response = JSONResponse(
            content=jsonable_encoder(body),
            headers={"X-Request-ID": _get_request_id(request), "Cache-Control": "no-store"},
        )
        response.set_cookie(
            SESSION_COOKIE,
            session.id,
            max_age=int(SESSION_TTL.total_seconds()),
            httponly=True,
            samesite="strict",
            secure=False,  # loopback HTTP; the console has no TLS origin to use
            path=API_PREFIX,
        )
        return response

    @router.delete("/auth/session", status_code=204, tags=["auth"])
    async def close_session(request: Request) -> Response:
        # Ending a session is a form of stopping and is never gated; an
        # anonymous caller simply has nothing to end.
        authority.close_session(request.cookies.get(SESSION_COOKIE))
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path=API_PREFIX)
        return response

    @router.get("/auth/operators", response_model=OperatorList, tags=["auth"])
    async def list_operators() -> OperatorList:
        items = [_operator_payload(item) for item in authority.principals()]
        return OperatorList(items=items, count=len(items))

    @router.post(
        "/auth/operators",
        response_model=IssuedCredentialRead,
        status_code=201,
        tags=["auth"],
    )
    async def create_operator(request: Request, payload: OperatorCreate) -> IssuedCredentialRead:
        actor = _require_actor(request, authority, settings)
        issued = authority.create_delegation(
            label=payload.label,
            scopes=payload.scopes,
            expires_at=payload.expires_at,
            by=actor,
        )
        return IssuedCredentialRead(
            operator=_operator_payload(issued.principal), token=issued.token
        )

    @router.post(
        "/auth/operators/{principal_id}/rotate",
        response_model=IssuedCredentialRead,
        tags=["auth"],
    )
    async def rotate_operator(request: Request, principal_id: str) -> IssuedCredentialRead:
        actor = _require_actor(request, authority, settings)
        issued = authority.rotate(principal_id, by=actor)
        return IssuedCredentialRead(
            operator=_operator_payload(issued.principal), token=issued.token
        )

    @router.post(
        "/auth/operators/{principal_id}/revoke",
        response_model=OperatorRead,
        tags=["auth"],
    )
    async def revoke_operator(request: Request, principal_id: str) -> OperatorRead:
        actor = _require_actor(request, authority, settings)
        return _operator_payload(authority.revoke(principal_id, by=actor))

    @router.post("/auth/owner/rotate", response_model=IssuedCredentialRead, tags=["auth"])
    async def rotate_owner(request: Request) -> IssuedCredentialRead:
        actor = _require_actor(request, authority, settings)
        owner = authority.owner()
        if owner is None or not actor.is_owner:
            raise AuthorizationDeniedError("only the project owner may rotate the owner credential")
        issued = authority.rotate(owner.id, by=actor)
        return IssuedCredentialRead(
            operator=_operator_payload(issued.principal), token=issued.token
        )

    # ---- safety -------------------------------------------------------

    @router.get("/safety/covenant", response_model=CovenantRead, tags=["safety"])
    async def safety_covenant() -> CovenantRead:
        # Re-read rather than serve a copy cached at startup: the page exists to
        # show what is being enforced right now.
        return covenant_payload(service.reload_covenant())

    @router.get("/safety/actions", response_model=ActionList, tags=["safety"])
    async def safety_actions() -> ActionList:
        return declared_action_payload(service.reload_covenant())

    @router.get("/safety/escalations", response_model=EscalationList, tags=["safety"])
    async def list_escalations(state: EscalationState | None = None) -> EscalationList:
        items = [_escalation_payload(item) for item in authority.escalations(state=state)]
        return EscalationList(items=items, count=len(items))

    @router.get(
        "/safety/escalations/{escalation_id}",
        response_model=EscalationRead,
        tags=["safety"],
    )
    async def get_escalation(escalation_id: str) -> EscalationRead:
        return _escalation_payload(authority.escalation(escalation_id))

    @router.post(
        "/safety/escalations/{escalation_id}/decision",
        response_model=EscalationRead,
        tags=["safety"],
    )
    async def decide_escalation(
        request: Request, escalation_id: str, payload: EscalationDecide
    ) -> EscalationRead:
        actor = _require_actor(request, authority, settings)
        return _escalation_payload(
            authority.decide_escalation(
                escalation_id,
                approve=payload.approve,
                by=actor,
                rationale=payload.rationale,
            )
        )

    # ---- jobs ---------------------------------------------------------

    @router.post("/jobs", response_model=JobRead, status_code=201, tags=["jobs"])
    async def create_job(request: Request, payload: JobCreate) -> JobRead:
        actor = _require_actor(request, authority, settings)
        return JobRead.model_validate(service.create_job(payload.type, payload.spec, actor=actor))

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
    async def cancel_job(request: Request, job_id: str) -> JobRead:
        # Stopping is never weighed against the covenant. Any verified operator
        # may cancel any job; requiring a matching delegation would let the
        # third principle's self-preservation argument gate the second
        # principle's duty to accept correction, which the ordering forbids.
        _require_actor(request, authority, settings)
        return JobRead.model_validate(service.cancel(job_id))

    @router.post(
        "/jobs/{job_id}/retry",
        response_model=JobRead,
        status_code=201,
        tags=["jobs"],
    )
    async def retry_job(request: Request, job_id: str) -> JobRead:
        actor = _require_actor(request, authority, settings)
        return JobRead.model_validate(service.retry(job_id, actor=actor))

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
        # The console authenticates with a cookie, so credentialed requests are
        # permitted — from exactly one exact origin, which is why the wildcard
        # that would make this dangerous is unavailable here by construction.
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type", "X-Request-ID"],
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
