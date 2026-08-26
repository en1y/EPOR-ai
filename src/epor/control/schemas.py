"""Versioned API and job-spec schemas for the v0.0.1 console."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import JobStatus, JobType


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SystemProbeSpec(StrictSchema):
    pass


class ResearchSyncSpec(StrictSchema):
    offline: bool = True
    source_ids: list[str] | None = Field(default=None, max_length=100)


TinyModelConfigPath = Literal[
    "configs/models/epor-tiny.yaml",
    "configs/models/epor-reference.yaml",
]

# Control jobs are deliberately fixture-scale. The standalone training CLI is
# not governed by this local-console guardrail and has its own explicit inputs.
TINY_CONTROL_CORPUS_MAX_BYTES = 1 * 1024 * 1024


class TinyTrainSpec(StrictSchema):
    config_path: TinyModelConfigPath = Field(
        default="configs/models/epor-tiny.yaml",
        alias="model_config",
        serialization_alias="model_config",
    )
    corpus_path: str = Field(
        default="fixtures/tiny_corpus.txt",
        max_length=512,
        description="Contained fixture corpus; capped at 1 MiB for control jobs.",
    )
    output_name: str = Field(default="tiny-train", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    max_steps: int = Field(default=20, ge=1, le=1_000)
    seed: int = Field(default=1337, ge=0, le=2**32 - 1)


class TinyEvalSpec(StrictSchema):
    checkpoint_path: str = Field(max_length=512)
    corpus_path: str = Field(
        default="fixtures/tiny_corpus.txt",
        max_length=512,
        description="Contained fixture corpus; capped at 1 MiB for control jobs.",
    )
    config_path: TinyModelConfigPath | None = Field(
        default=None,
        alias="model_config",
        serialization_alias="model_config",
    )
    max_batches: int | None = Field(default=None, ge=1, le=10_000)


JOB_SPEC_MODELS: dict[JobType, type[StrictSchema]] = {
    JobType.SYSTEM_PROBE: SystemProbeSpec,
    JobType.RESEARCH_SYNC: ResearchSyncSpec,
    JobType.TINY_TRAIN: TinyTrainSpec,
    JobType.TINY_EVAL: TinyEvalSpec,
}


class JobCreate(StrictSchema):
    type: JobType
    spec: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: Any = None
    request_id: str


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type: JobType
    status: JobStatus
    spec: dict[str, Any]
    result: dict[str, Any] | None
    progress: float
    error_code: str | None
    error_message: str | None
    worker_id: str | None
    retry_of_id: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    heartbeat_at: datetime | None
    cancel_requested_at: datetime | None


class JobList(BaseModel):
    items: list[JobRead]
    count: int


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    kind: str
    payload: dict[str, Any]
    created_at: datetime


class EventList(BaseModel):
    items: list[EventRead]
    next_after: int


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str | None
    created_at: datetime


class ArtifactList(BaseModel):
    items: list[ArtifactRead]


class HealthRead(BaseModel):
    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"
    version: str


class CapabilityRead(BaseModel):
    bind_host: str
    allowed_origin: str
    job_types: list[JobType]
    cancellation: bool = True
    event_transport: Literal["polling"] = "polling"
    arbitrary_commands: Literal[False] = False
    arbitrary_url_fetching: Literal[False] = False
    filesystem_browser: Literal[False] = False
    hardware: dict[str, Any]


class JobTypeRead(BaseModel):
    type: JobType
    title: str
    description: str
    schema_: dict[str, Any] = Field(alias="schema", serialization_alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class JobTypeList(BaseModel):
    items: list[JobTypeRead]


class ResearchSourceRead(BaseModel):
    id: str
    title: str
    organization: str
    publication_date: str
    media_type: str
    canonical_url: str
    tags: list[str]
    evidence_classes: list[str]
    sync_status: Literal["verified", "missing", "error", "unknown"]
    sha256: str | None = None
    size_bytes: int | None = None
    status_message: str = ""


class ResearchCatalogRead(BaseModel):
    schema_version: int | None
    available: bool
    catalog_path: str = "research/catalog.yaml"
    sources: list[ResearchSourceRead]
    count: int
    error: str | None = None


class DocumentSummary(BaseModel):
    slug: str
    title: str
    group: str


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    count: int


class DocumentRead(DocumentSummary):
    markdown: str


class CertifiedProfileRead(BaseModel):
    profile_id: str
    hardware: str
    max_context: int
    precision: str
    status: Literal["planned", "validated", "certified"]
    ram_gib: float | None = None
    vram_gib: float | None = None
    notes: str | None = None


class ModelFamilyRead(BaseModel):
    display_name: str
    slug: str
    family: Literal["gamma", "alpha", "beta"]
    architecture: str
    total_parameters: int | None
    active_parameters: int | None
    core_parameters: int | None
    configured_max_context: int
    trained_max_context: int
    validated_max_context: int
    operational_default_context: int
    certified_profiles: list[CertifiedProfileRead]
    research_features: list[str]
    status: Literal["planned", "proxy", "trained"] = "planned"
    metrics: Literal[None] = None


class ModelFamilyList(BaseModel):
    items: list[ModelFamilyRead]
    count: int


JobStatusFilter = Annotated[JobStatus | None, Field(default=None)]
