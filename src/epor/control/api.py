"""Loopback-only FastAPI application for the EPOR local console."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi import Path as FastApiPath
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
from .errors import (
    AuditItemNotFoundError,
    AuthenticationRequiredError,
    AuthorizationDeniedError,
    CapabilityUnavailableError,
    ControlError,
    ReportIntegrityError,
    ReportNotFoundError,
)
from .models import JobStatus, JobType
from .safety_view import covenant_payload, declared_action_payload
from .schemas import (
    JOB_SPEC_MODELS,
    ActionList,
    ArtifactList,
    ArtifactRead,
    AuditVerdictCreate,
    CapabilityRead,
    CovenantRead,
    DataSummaryRead,
    DocumentList,
    DocumentRead,
    DocumentSummary,
    ErrorEnvelope,
    EscalationDecide,
    EscalationList,
    EscalationRead,
    EvaluationReportList,
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
    LadderRead,
    LadderRungRead,
    ModelFamilyList,
    ModelFamilyRead,
    OperatorCreate,
    OperatorList,
    OperatorRead,
    ResearchCatalogRead,
    ResearchSourceRead,
    ScalingReportList,
    SessionCreate,
    TokenizerSummaryRead,
    TrainingSummaryRead,
)
from .security import redact, redact_text, redact_values
from .service import JobService
from .settings import ControlSettings

SESSION_COOKIE = "epor_session"
API_PREFIX = "/api/v1"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
JobTypeQuery = Annotated[JobType | None, Query(alias="type")]
JobLimitQuery = Annotated[int, Query(ge=1, le=500)]
EventAfterQuery = Annotated[int, Query(ge=0)]
EventLimitQuery = Annotated[int, Query(ge=1, le=1_000)]
ReportIdPath = Annotated[str, FastApiPath(pattern=r"^[0-9a-f]{64}$")]
AuditItemPath = Annotated[
    str, FastApiPath(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
]
AlignmentReportKind = Annotated[str, FastApiPath(pattern=r"^(?:overoptimization|g10|calibration)$")]
AuditStateQuery = Annotated[str | None, Query(pattern=r"^(?:pending|audited)$")]
JudgingRunQuery = Annotated[str | None, Query(pattern=r"^[0-9a-f]{64}$")]
ReportIdQuery = Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")]
ResamplesQuery = Annotated[int, Query(ge=1, le=10_000)]
SeedQuery = Annotated[int, Query(ge=0, le=2**32 - 1)]
_JOB_TYPE_COPY: dict[JobType, tuple[str, str]] = {
    JobType.SYSTEM_PROBE: (
        "System probe",
        "Refresh local CPU, memory, toolchain, and accelerator capability facts.",
    ),
    JobType.RESEARCH_SYNC: (
        "Research sync",
        "Verify the local paper archive or synchronize only catalog-allowlisted sources.",
    ),
    JobType.DATA_INGEST: (
        "Data ingestion",
        "Register and stream reviewed local text into immutable provenance layers.",
    ),
    JobType.DATA_BUILD: (
        "Dataset build",
        "Apply tombstones, redaction, quarantine, global deduplication, and stable splits.",
    ),
    JobType.DATA_REMOVE: (
        "Data removal",
        "Record an immutable tombstone that excludes a source or document from future builds.",
    ),
    JobType.DATA_AUDIT: (
        "Data audit",
        "Verify content hashes and summarize registrations, layers, builds, and removals.",
    ),
    JobType.TOKENIZER_TRAIN: (
        "Tokenizer training",
        "Train a deterministic byte-level BPE from one build's train split only.",
    ),
    JobType.TOKENIZER_SHARD: (
        "Token shards",
        "Encode a build into content-addressed uint16 shards that keep document boundaries.",
    ),
    JobType.TOKENIZER_EVALUATE: (
        "Tokenizer fertility",
        "Measure fertility and exact byte round-trip on one split; failures fail the job.",
    ),
    JobType.KERNEL_TRAIN: (
        "Kernel pretraining",
        "Train on a verified tokenized build in one process with exact checkpointed resume.",
    ),
    JobType.KERNEL_EVALUATE: (
        "Kernel evaluation",
        "Measure a kernel checkpoint's loss, perplexity, and MTP losses on one split.",
    ),
    JobType.EVAL_SUITE: (
        "Evaluation suite",
        "Score a kernel checkpoint on a versioned suite; code tasks stay on the owner's CLI.",
    ),
    JobType.EVAL_CONTAMINATION: (
        "Contamination scan",
        "Measure n-gram overlap of a suite's items with a training build (a lower bound).",
    ),
    JobType.SCALE_PROFILE: (
        "Host profile",
        "Measure training-step throughput and memory on this host; MFU needs a supplied peak.",
    ),
    JobType.SCALE_SWEEP: (
        "Scaling sweep",
        "Train rung x token-budget x seed points; rungs above the local bound are projection-only.",
    ),
    JobType.SCALE_MIXTURE: (
        "Mixture ablation",
        "Train one small model per domain mixture and compare per-domain validation loss.",
    ),
    JobType.ALIGN_GENERATE: (
        "Alignment candidates",
        "Sample teacher outputs, covenant-cited critique/revision, or model-written G10 items.",
    ),
    JobType.ALIGN_JUDGE: (
        "AI judge",
        "Record judge decisions as data; nothing they accept trains before a human audit.",
    ),
    JobType.ALIGN_DATASET: (
        "Alignment dataset",
        "Build an SFT or preference dataset; candidates enter only from audited judging runs.",
    ),
    JobType.ALIGN_SFT: (
        "Supervised fine-tuning",
        "Fine-tune or distill on a chat dataset through the training kernel (one process).",
    ),
    JobType.ALIGN_PREFERENCE: (
        "DPO baseline",
        "Optimize a policy on preference pairs against a frozen reference, with a stop monitor.",
    ),
    JobType.ALIGN_EVALUATE: (
        "Alignment evaluation",
        "Overoptimization monitors, judge calibration, or the G10 before/after comparison.",
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

    @router.get("/data", response_model=DataSummaryRead, tags=["data"])
    async def data_summary() -> DataSummaryRead:
        from epor.data.service import DataEngine

        assert settings.data_root is not None
        return DataSummaryRead.model_validate(DataEngine(settings.data_root).dataset_summary())

    @router.get("/tokenizers", response_model=TokenizerSummaryRead, tags=["tokenizers"])
    async def tokenizer_summary() -> TokenizerSummaryRead:
        try:
            from epor.tokenizer.summary import tokenizer_summary
        except ModuleNotFoundError as exc:
            raise CapabilityUnavailableError(
                "tokenizer support requires the optional data dependencies (uv sync --extra data)"
            ) from exc

        assert settings.data_root is not None
        return TokenizerSummaryRead.model_validate(tokenizer_summary(settings.data_root))

    @router.get("/training/runs", response_model=TrainingSummaryRead, tags=["training"])
    async def training_runs() -> TrainingSummaryRead:
        from .training_runs import training_summary

        assert settings.artifact_root is not None
        return TrainingSummaryRead.model_validate(
            training_summary(settings.artifact_root, settings.project_root)
        )

    @router.get("/evaluation/reports", response_model=EvaluationReportList, tags=["evaluation"])
    async def evaluation_reports() -> EvaluationReportList:
        from epor.evaluation.summary import list_reports

        assert settings.data_root is not None
        return EvaluationReportList.model_validate(redact_values(list_reports(settings.data_root)))

    @router.get(
        "/evaluation/reports/{report_id}", response_model=dict[str, Any], tags=["evaluation"]
    )
    async def evaluation_report(report_id: ReportIdPath) -> dict[str, Any]:
        from epor.evaluation.summary import report_view

        return redact_values(report_view(_evaluation_report(report_id)))

    @router.get("/evaluation/compare", response_model=dict[str, Any], tags=["evaluation"])
    async def evaluation_compare(
        baseline: ReportIdQuery,
        candidate: ReportIdQuery,
        resamples: ResamplesQuery = 1_000,
        seed: SeedQuery = 0,
    ) -> dict[str, Any]:
        from epor.evaluation import compare_reports

        comparison = compare_reports(
            _evaluation_report(baseline),
            _evaluation_report(candidate),
            resamples=resamples,
            seed=seed,
        )
        return redact_values(comparison.model_dump(mode="json"))

    @router.get("/scaling/ladder", response_model=LadderRead, tags=["scaling"])
    async def scaling_ladder() -> LadderRead:
        from epor.scaling.summary import ladder_table

        return LadderRead(
            rungs=[LadderRungRead.model_validate(row) for row in ladder_table(None)],
            note=(
                "Rung files are architecture contracts, not trained models; projection-only "
                "rungs are never allocated on this machine."
            ),
        )

    @router.get("/scaling/reports", response_model=ScalingReportList, tags=["scaling"])
    async def scaling_reports() -> ScalingReportList:
        from epor.scaling.summary import list_scaling_reports

        assert settings.data_root is not None
        return ScalingReportList.model_validate(
            redact_values(list_scaling_reports(settings.data_root))
        )

    @router.get(
        "/scaling/reports/{report_id}", response_model=list[dict[str, Any]], tags=["scaling"]
    )
    async def scaling_report(report_id: ReportIdPath) -> list[dict[str, Any]]:
        from epor.data.store import DataStore
        from epor.scaling import ScalingReportError
        from epor.scaling.summary import load_verified_scaling_reports, scaling_summary

        assert settings.data_root is not None
        if not DataStore(settings.data_root).scaling_report_directory(report_id).is_dir():
            raise ReportNotFoundError(f"scaling report not found: {report_id}")
        try:
            reports = load_verified_scaling_reports(settings.data_root, report_id)
        except ScalingReportError as exc:
            raise ReportIntegrityError(str(exc)) from exc
        return redact_values(
            [
                {**report.model_dump(mode="json"), "summary": scaling_summary(report)}
                for report in reports
            ]
        )

    def _evaluation_report(report_id: str) -> Any:
        from epor.data.store import DataStore
        from epor.evaluation import ReportError
        from epor.evaluation.summary import load_verified_report

        assert settings.data_root is not None
        if not DataStore(settings.data_root).evaluation_report_path(report_id).is_file():
            raise ReportNotFoundError(f"evaluation report not found: {report_id}")
        try:
            return load_verified_report(settings.data_root, report_id)
        except ReportError as exc:
            raise ReportIntegrityError(str(exc)) from exc

    # ---- alignment ----------------------------------------------------

    def _audit_ledger() -> Any:
        try:
            from epor.alignment.audit import AuditLedger
        except ModuleNotFoundError as exc:
            raise CapabilityUnavailableError(
                "alignment support requires the train-cpu and data profiles"
            ) from exc
        assert settings.safety_root is not None
        return AuditLedger(settings.safety_root)

    def _audit_item(ledger: Any, item_id: str) -> Any:
        for item in ledger.items():
            if item.id == item_id:
                return item
        raise AuditItemNotFoundError(f"audit item not found: {item_id}")

    @router.get("/alignment", response_model=dict[str, Any], tags=["alignment"])
    async def alignment_summary() -> dict[str, Any]:
        ledger = _audit_ledger()
        from epor.alignment.summary import alignment_summary as summarize

        assert settings.data_root is not None
        return redact_values(summarize(settings.data_root, ledger))

    @router.get("/alignment/audits", response_model=list[dict[str, Any]], tags=["alignment"])
    async def alignment_audits(
        state: AuditStateQuery = None, run: JudgingRunQuery = None
    ) -> list[dict[str, Any]]:
        views = [item.view() for item in _audit_ledger().items()]
        return redact_values(
            [
                view
                for view in views
                if (state is None or view["state"] == state)
                and (run is None or view["judging_run"] == run)
            ]
        )

    @router.get("/alignment/audits/{item_id}", response_model=dict[str, Any], tags=["alignment"])
    async def alignment_audit(item_id: AuditItemPath) -> dict[str, Any]:
        return redact_values(_audit_item(_audit_ledger(), item_id).view())

    @router.post(
        "/alignment/audits/{item_id}/verdict",
        response_model=dict[str, Any],
        tags=["alignment"],
    )
    async def alignment_verdict(
        request: Request, item_id: AuditItemPath, payload: AuditVerdictCreate
    ) -> dict[str, Any]:
        # A verdict is human oversight of model feedback: it needs a verified
        # identity holding the audit scope, and it is never a job.
        actor = _require_actor(request, authority, settings)
        ledger = _audit_ledger()
        _audit_item(ledger, item_id)
        service.admit(
            "align_audit",
            spec={"item_id": item_id, "verdict": payload.verdict},
            actor=actor,
            surface="api",
        )
        ledger.record_verdict(item_id, payload.verdict, actor=actor, note=payload.note)
        return redact_values(_audit_item(ledger, item_id).view())

    @router.get(
        "/alignment/reports/{kind}/{report_id}",
        response_model=dict[str, Any],
        tags=["alignment"],
    )
    async def alignment_report(
        kind: AlignmentReportKind, report_id: ReportIdPath
    ) -> dict[str, Any]:
        from epor.alignment.monitors import load_report
        from epor.alignment.records import AlignmentStore

        assert settings.data_root is not None
        store = AlignmentStore(settings.data_root)
        if not store.data.alignment_path(f"reports/{kind}/{report_id}.json").is_file():
            raise ReportNotFoundError(f"alignment report not found: {kind}/{report_id}")
        try:
            return redact_values(load_report(store, kind, report_id))
        except ValueError as exc:
            raise ReportIntegrityError(str(exc)) from exc

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
