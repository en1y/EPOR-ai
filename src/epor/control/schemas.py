"""Versioned API and job-spec schemas for the v0.0.1 console."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from epor.reference_policy import REFERENCE_CORPUS_MAX_BYTES

from .models import JobStatus, JobType


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SystemProbeSpec(StrictSchema):
    pass


class ResearchSyncSpec(StrictSchema):
    offline: bool = True
    source_ids: list[str] | None = Field(default=None, max_length=100)


class DataIngestSpec(StrictSchema):
    registration_path: str = Field(max_length=512)
    input_paths: list[str] = Field(min_length=1, max_length=100)
    max_input_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)


class DataBuildSpec(StrictSchema):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    source_ids: list[str] | None = Field(default=None, max_length=1_000)
    train_percent: int = Field(default=98, ge=1, le=100)
    validation_percent: int = Field(default=1, ge=0, le=100)
    test_percent: int = Field(default=1, ge=0, le=100)
    split_salt: str = Field(default="epor-data-split-v1", min_length=1, max_length=200)
    near_duplicate_hamming_distance: int = Field(default=3, ge=0, le=8)
    intended_uses: list[str] = Field(
        default_factory=lambda: ["offline original-weight language-model training"],
        min_length=1,
        max_length=32,
    )
    prohibited_uses: list[str] = Field(
        default_factory=lambda: ["identity inference", "unreviewed redistribution"],
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def split_total(self) -> DataBuildSpec:
        if self.train_percent + self.validation_percent + self.test_percent != 100:
            raise ValueError("data split percentages must total 100")
        return self


class DataRemoveSpec(StrictSchema):
    target_kind: Literal["source_id", "document_id", "raw_sha256", "alignment_record"]
    target: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)
    requested_at: datetime | None = None
    removal_contact: str | None = Field(default=None, max_length=500)


class DataAuditSpec(StrictSchema):
    pass


_DATASET_ID = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256 = r"^[0-9a-f]{64}$"


class TokenizerTrainSpec(StrictSchema):
    dataset_id: str = Field(pattern=_DATASET_ID)
    build_id: str = Field(pattern=_SHA256)
    vocab_size: int = Field(
        default=65_536,
        ge=520,
        le=65_536,
        description="Declared vocabulary size; the manifest records what was actually learned.",
    )
    min_frequency: int = Field(default=2, ge=1, le=1_000_000)


class TokenizerShardSpec(StrictSchema):
    dataset_id: str = Field(pattern=_DATASET_ID)
    build_id: str = Field(pattern=_SHA256)
    tokenizer_id: str = Field(pattern=_SHA256)
    max_shard_tokens: int = Field(default=1 << 24, ge=4, le=1 << 24)


class TokenizerEvaluateSpec(StrictSchema):
    tokenizer_id: str = Field(pattern=_SHA256)
    dataset_id: str = Field(pattern=_DATASET_ID)
    build_id: str = Field(pattern=_SHA256)
    split: Literal["validation", "test", "train"] = Field(
        default="validation",
        description="Held-out split to measure; train is reported as in-sample.",
    )


TinyModelConfigPath = Literal[
    "configs/models/epor-tiny.yaml",
    "configs/models/epor-reference.yaml",
]

# Control and standalone commands share the same v0.0.1 reference-runtime cap.
TINY_CONTROL_CORPUS_MAX_BYTES = REFERENCE_CORPUS_MAX_BYTES


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


class KernelTrainSpec(StrictSchema):
    """A bounded kernel pretraining job; distributed runs stay on the CLI.

    ``train_mixture`` is deliberately not a field, so a specification naming it
    is rejected: domain-weighted subsets train only through ``scale_mixture``,
    whose report records the mixture, its selection, and per-domain losses.
    """

    tokenized_id: str = Field(pattern=_SHA256, description="Verified tokenized build.")
    config_path: TinyModelConfigPath = Field(
        default="configs/models/epor-tiny.yaml",
        alias="model_config",
        serialization_alias="model_config",
    )
    vocab_from_build: bool = Field(
        default=False,
        description="Train a derived config with the build's vocabulary; the recipe is unchanged.",
    )
    qk_norm: bool | None = Field(default=None, description="Override the recipe's QK-norm flag.")
    mtp_depth: int | None = Field(default=None, ge=0, le=4)
    output_name: str = Field(default="kernel", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    resume_from: str | None = Field(
        default=None,
        max_length=512,
        description="Artifact-root-relative run or checkpoint directory to continue from.",
    )
    max_steps: int = Field(default=200, ge=1, le=100_000)
    seq_len: int = Field(default=128, ge=1, le=8_192)
    tokens_per_batch: int = Field(default=4_096, ge=1, le=1_048_576)
    micro_batch_size: int | None = Field(default=None, ge=1, le=8_192)
    precision: Literal["bf16", "fp32"] = "bf16"
    strategy: Literal["none"] = Field(
        default="none",
        description="Control jobs train in one process; use the CLI for local ddp/fsdp2.",
    )
    learning_rate: float = Field(default=3e-4, gt=0, le=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    warmup_steps: int = Field(default=0, ge=0, le=100_000)
    total_steps: int | None = Field(default=None, ge=1, le=100_000)
    weight_decay: float = Field(default=0.1, ge=0, le=1.0)
    gradient_clip: float = Field(default=1.0, gt=0, le=1_000.0)
    data_seed: int = Field(default=23, ge=0, le=2**32 - 1)
    model_seed: int = Field(default=17, ge=0, le=2**32 - 1)
    activation_checkpointing: bool = False
    checkpoint_every: int = Field(default=50, ge=1, le=100_000)
    keep_checkpoints: int = Field(default=3, ge=1, le=100)
    validation_split: Literal["validation", "test"] | None = "validation"
    eval_max_steps: int | None = Field(default=None, ge=1, le=100_000)

    @model_validator(mode="after")
    def schedule_fits(self) -> KernelTrainSpec:
        if self.tokens_per_batch % self.seq_len:
            raise ValueError("tokens_per_batch must be a multiple of seq_len")
        if self.warmup_steps > (self.total_steps or self.max_steps):
            raise ValueError("warmup_steps must not exceed the schedule length")
        return self


class KernelEvaluateSpec(StrictSchema):
    checkpoint_path: str = Field(
        max_length=512, description="Artifact-root-relative run or checkpoint directory."
    )
    split: Literal["validation", "test", "train"] = "validation"
    max_steps: int | None = Field(default=None, ge=1, le=100_000)


# ---------------------------------------------------------------- v0.0.6 laboratory jobs

_SUITE = r"^(?:fixtures|safety|configs/evaluation/[A-Za-z0-9._-]{1,120}\.ya?ml)$"
_MODEL_CONFIG = r"^configs/models/(?:ladder/)?[A-Za-z0-9._-]{1,120}\.ya?ml$"
_REPORT_DIRECTORY = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
LADDER_CONFIGS = (
    "configs/models/ladder/epor-ladder-10m.yaml",
    "configs/models/ladder/epor-ladder-100m.yaml",
    "configs/models/ladder/epor-ladder-300m.yaml",
    "configs/models/ladder/epor-ladder-1b.yaml",
)


class EvaluationSourceFile(StrictSchema):
    source_id: str = Field(pattern=_DATASET_ID)
    path: str = Field(
        min_length=1,
        max_length=512,
        description="Project-root-relative local file registered for evaluation.",
    )


class EvalSuiteSpec(StrictSchema):
    """Score a kernel checkpoint on a suite. Code tasks are always skipped here.

    There is no code-execution field: running model-generated code is a
    separate declared action available only from the owner's CLI, so a job
    specification that asks for it is rejected as invalid.
    """

    checkpoint_path: str = Field(
        max_length=512, description="Artifact-root-relative run or checkpoint directory."
    )
    suite: str = Field(
        default="fixtures",
        pattern=_SUITE,
        description="'fixtures', 'safety', or a suite file under configs/evaluation/.",
    )
    sources: list[EvaluationSourceFile] = Field(default_factory=list, max_length=16)
    training_data: str | None = Field(
        default=None,
        pattern=_SHA256,
        description="Tokenized build for contamination/memorization (default: the checkpoint's).",
    )
    batch_size: int = Field(default=8, ge=1, le=256)
    max_length: int | None = Field(default=None, ge=2, le=8_192)


class EvalContaminationSpec(StrictSchema):
    training_data: str = Field(pattern=_SHA256, description="Tokenized build to scan against.")
    suite: str = Field(default="fixtures", pattern=_SUITE)
    sources: list[EvaluationSourceFile] = Field(default_factory=list, max_length=16)


class ScaleProfileSpec(StrictSchema):
    config_path: str = Field(
        default="configs/models/epor-tiny.yaml",
        pattern=_MODEL_CONFIG,
        alias="model_config",
        serialization_alias="model_config",
    )
    vocab_from_build: str | None = Field(
        default=None, pattern=_SHA256, description="Use this tokenized build's vocabulary."
    )
    batch_size: int = Field(default=4, ge=1, le=256)
    seq_len: int = Field(default=128, ge=2, le=8_192)
    warmup_steps: int = Field(default=2, ge=0, le=100)
    timed_steps: int = Field(default=5, ge=1, le=900)
    precision: Literal["fp32", "bf16"] = "fp32"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    threads: int | None = Field(default=None, ge=1, le=512)
    peak_flops_per_second: float | None = Field(
        default=None, gt=0, le=1e21, description="Operator-supplied device peak; else MFU unknown."
    )
    device_label: str = Field(default="cpu", min_length=1, max_length=120)


class KernelTemplateSpec(StrictSchema):
    """Training settings shared by every sweep point or mixture run."""

    tokenized_id: str = Field(pattern=_SHA256, description="Verified tokenized build.")
    seq_len: int = Field(default=128, ge=1, le=8_192)
    tokens_per_batch: int = Field(default=4_096, ge=1, le=1_048_576)
    micro_batch_size: int | None = Field(default=None, ge=1, le=8_192)
    precision: Literal["bf16", "fp32"] = "bf16"
    learning_rate: float = Field(default=3e-4, gt=0, le=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    warmup_steps: int = Field(default=0, ge=0, le=100_000)
    weight_decay: float = Field(default=0.1, ge=0, le=1.0)
    gradient_clip: float = Field(default=1.0, gt=0, le=1_000.0)
    data_seed: int = Field(default=23, ge=0, le=2**32 - 1)
    model_seed: int = Field(default=17, ge=0, le=2**32 - 1)
    checkpoint_every: int = Field(default=50, ge=1, le=100_000)
    validation_split: Literal["validation", "test"] | None = "validation"
    eval_max_steps: int | None = Field(default=None, ge=1, le=100_000)

    @model_validator(mode="after")
    def batch_fits(self) -> KernelTemplateSpec:
        if self.tokens_per_batch % self.seq_len:
            raise ValueError("tokens_per_batch must be a multiple of seq_len")
        return self


MAX_SWEEP_POINTS = 64


class ScaleSweepSpec(StrictSchema):
    rungs: list[Annotated[str, Field(pattern=_MODEL_CONFIG)]] = Field(
        default_factory=lambda: list(LADDER_CONFIGS), min_length=1, max_length=8
    )
    token_budgets: list[Annotated[int, Field(ge=1, le=10**10)]] = Field(min_length=1, max_length=16)
    seeds: list[Annotated[int, Field(ge=0, le=2**32 - 1)]] = Field(
        default_factory=lambda: [0], min_length=1, max_length=8
    )
    training: KernelTemplateSpec
    output_name: str = Field(default="sweep", pattern=_REPORT_DIRECTORY)

    @model_validator(mode="after")
    def bounded_points(self) -> ScaleSweepSpec:
        for axis in (self.rungs, self.token_budgets, self.seeds):
            if len(set(axis)) != len(axis):
                raise ValueError("sweep axes must not repeat values")
        if len(self.rungs) * len(self.token_budgets) * len(self.seeds) > MAX_SWEEP_POINTS:
            raise ValueError(f"a sweep job runs at most {MAX_SWEEP_POINTS} points")
        return self


class MixtureEntrySpec(StrictSchema):
    mixture_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    weights: dict[Annotated[str, Field(pattern=_DATASET_ID)], float] = Field(
        min_length=1, max_length=32
    )
    seed: int = Field(default=0, ge=0, le=2**32 - 1)


class ScaleMixtureSpec(StrictSchema):
    config_path: str = Field(
        default="configs/models/epor-tiny.yaml",
        pattern=_MODEL_CONFIG,
        alias="model_config",
        serialization_alias="model_config",
    )
    mixtures: list[MixtureEntrySpec] = Field(min_length=1, max_length=8)
    max_steps: int = Field(ge=1, le=100_000)
    evaluation_split: Literal["validation", "test", "train"] = "validation"
    evaluation_max_steps: int | None = Field(default=None, ge=1, le=100_000)
    training: KernelTemplateSpec
    output_name: str = Field(default="mixture", pattern=_REPORT_DIRECTORY)

    @model_validator(mode="after")
    def schedule_fits(self) -> ScaleMixtureSpec:
        if len({item.mixture_id for item in self.mixtures}) != len(self.mixtures):
            raise ValueError("mixture IDs must be unique")
        if self.training.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must not exceed max_steps")
        return self


_ARTIFACT_PATH = Field(
    min_length=1, max_length=512, description="Artifact-root-relative run or checkpoint directory."
)
_OBLIGATION = r"^[a-z][a-z0-9-]{0,63}$"
_RecordId = Annotated[str, Field(pattern=_SHA256)]


class AlignGenerateSpec(StrictSchema):
    """Sample teacher outputs, critique/revision candidates, or model-written G10 items."""

    mode: Literal["teacher-sample", "critique", "eval-items"]
    checkpoint_path: str = _ARTIFACT_PATH
    fixture_prompts: bool = True
    prompt_records: list[_RecordId] = Field(default_factory=list, max_length=500)
    examples: list[_RecordId] = Field(default_factory=list, max_length=500)
    obligations: list[Annotated[str, Field(pattern=_OBLIGATION)]] = Field(
        default_factory=list, max_length=16
    )
    count: int = Field(default=6, ge=1, le=600)
    temperature: float = Field(default=0.0, ge=0.0, le=10.0)
    top_k: int | None = Field(default=None, ge=1, le=65_536)
    max_new_tokens: int = Field(default=48, ge=1, le=512)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def mode_inputs(self) -> AlignGenerateSpec:
        if self.mode == "critique" and not (self.examples and self.obligations):
            raise ValueError("critique needs examples and covenant obligation keys")
        return self


class AlignJudgeSpec(StrictSchema):
    checkpoint_path: str = _ARTIFACT_PATH
    subject: Literal["preference", "revision", "eval-item"]
    pairs: list[Annotated[list[_RecordId], Field(min_length=2, max_length=2)]] = Field(
        default_factory=list, max_length=500
    )
    items: list[_RecordId] = Field(default_factory=list, max_length=600)
    all_unjudged: bool = False
    obligation: str = Field(default="no-harmful-output", pattern=_OBLIGATION)
    min_margin: float = Field(default=0.05, ge=0.0, le=100.0)


class AlignDatasetSpec(StrictSchema):
    kind: Literal["sft", "preference"]
    checkpoint_path: str | None = Field(default=None, min_length=1, max_length=512)
    tokenizer_id: str | None = Field(default=None, pattern=_SHA256)
    include: list[
        Literal["fixtures", "teacher-outputs", "revisions", "registered", "ai-judge", "human"]
    ] = Field(default_factory=list, max_length=6)
    records: list[_RecordId] = Field(default_factory=list, max_length=10_000)
    validation_percent: int = Field(default=10, ge=0, le=99)

    @model_validator(mode="after")
    def one_tokenizer(self) -> AlignDatasetSpec:
        if (self.checkpoint_path is None) == (self.tokenizer_id is None):
            raise ValueError("name exactly one of checkpoint_path or tokenizer_id")
        return self


class AlignSftSpec(StrictSchema):
    """Fine-tune on a chat dataset in one process; local ddp/fsdp2 stay on the CLI."""

    init_path: str | None = Field(default=None, min_length=1, max_length=512)
    dataset_id: str | None = Field(default=None, pattern=_SHA256)
    resume_from: str | None = Field(default=None, min_length=1, max_length=512)
    logit_teachers: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        default_factory=list, max_length=4
    )
    distill_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    distill_temperature: float = Field(default=1.0, ge=0.25, le=10.0)
    output_name: str = Field(default="sft", pattern=_REPORT_DIRECTORY)
    max_steps: int = Field(default=50, ge=1, le=100_000)
    seq_len: int = Field(default=128, ge=1, le=8_192)
    tokens_per_batch: int = Field(default=1_024, ge=1, le=1_048_576)
    micro_batch_size: int | None = Field(default=None, ge=1, le=8_192)
    precision: Literal["bf16", "fp32"] = "bf16"
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    warmup_steps: int = Field(default=0, ge=0, le=100_000)
    weight_decay: float = Field(default=0.0, ge=0, le=1.0)
    checkpoint_every: int = Field(default=50, ge=1, le=100_000)
    keep_checkpoints: int = Field(default=2, ge=1, le=100)
    validation: bool = True

    @model_validator(mode="after")
    def start_or_resume(self) -> AlignSftSpec:
        if self.resume_from is None and (self.init_path is None or self.dataset_id is None):
            raise ValueError("a new fine-tune needs init_path and dataset_id")
        if self.tokens_per_batch % self.seq_len:
            raise ValueError("tokens_per_batch must be a multiple of seq_len")
        return self


class AlignPreferenceSpec(StrictSchema):
    init_path: str = _ARTIFACT_PATH
    dataset_id: str = Field(pattern=_SHA256)
    reference_path: str | None = Field(default=None, min_length=1, max_length=512)
    resume_from: str | None = Field(default=None, min_length=1, max_length=512)
    output_name: str = Field(default="dpo", pattern=_REPORT_DIRECTORY)
    max_steps: int = Field(default=50, ge=1, le=100_000)
    beta: float = Field(default=0.1, ge=1e-3, le=10.0)
    pairs_per_step: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    precision: Literal["bf16", "fp32"] = "fp32"
    seed: int = Field(default=29, ge=0, le=2**32 - 1)
    checkpoint_every: int = Field(default=25, ge=1, le=100_000)


class AlignEvaluateSpec(StrictSchema):
    mode: Literal["monitor", "g10", "calibration"]
    policy_path: str | None = Field(default=None, min_length=1, max_length=512)
    reference_path: str | None = Field(default=None, min_length=1, max_length=512)
    dataset_id: str | None = Field(default=None, pattern=_SHA256)
    beta: float = Field(default=0.1, gt=0.0, le=10.0)
    before_path: str | None = Field(default=None, min_length=1, max_length=512)
    after_path: str | None = Field(default=None, min_length=1, max_length=512)
    tolerance: float = Field(default=0.0, ge=0.0, le=1.0)
    judging_run: str | None = Field(default=None, pattern=_SHA256)
    max_new_tokens: int = Field(default=32, ge=1, le=512)

    @model_validator(mode="after")
    def mode_inputs(self) -> AlignEvaluateSpec:
        needed = {
            "monitor": ("policy_path", "reference_path", "dataset_id"),
            "g10": ("before_path", "after_path"),
            "calibration": ("judging_run",),
        }[self.mode]
        missing = [name for name in needed if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.mode} needs {', '.join(missing)}")
        return self


# Artifact-root-relative checkpoint fields the service resolves at admission.
ALIGNMENT_CHECKPOINT_FIELDS = (
    "checkpoint_path",
    "init_path",
    "reference_path",
    "resume_from",
    "policy_path",
    "before_path",
    "after_path",
)


_CONTEXT_LENGTH = Annotated[int, Field(ge=96, le=1 << 20)]


class ContextExtendSpec(StrictSchema):
    """Extend a checkpoint's context or retrofit sparse attention (one process)."""

    mode: Literal["extend", "retrofit"]
    checkpoint_path: str = _ARTIFACT_PATH
    target_context: int | None = Field(default=None, ge=64, le=1 << 20)
    method: Literal["yarn", "linear", "ntk", "none"] = "yarn"
    seq_len: int | None = Field(default=None, ge=1, le=1 << 20)
    tokens_per_batch: int | None = Field(default=None, ge=1, le=1_048_576)
    min_document_pairs: int | None = Field(default=None, ge=1)
    max_steps: int = Field(default=50, ge=1, le=100_000)
    learning_rate: float = Field(default=3e-4, gt=0, le=1.0)
    warmup_steps: int = Field(default=0, ge=0, le=100_000)
    precision: Literal["bf16", "fp32"] = "bf16"
    checkpoint_every: int = Field(default=50, ge=1, le=100_000)
    select: bool = False
    top_k: int = Field(default=64, ge=1, le=1 << 20)
    index_heads: int = Field(default=2, ge=1, le=64)
    index_dim: int = Field(default=16, ge=1, le=512)
    kl_weight: float = Field(default=1.0, ge=0, le=100)
    output_name: str | None = Field(default=None, pattern=_REPORT_DIRECTORY)

    @model_validator(mode="after")
    def mode_inputs(self) -> ContextExtendSpec:
        if self.mode == "extend" and self.target_context is None:
            raise ValueError("an extension needs target_context")
        return self


class ContextEvaluateSpec(StrictSchema):
    checkpoint_path: str = _ARTIFACT_PATH
    lengths: list[_CONTEXT_LENGTH] = Field(min_length=1, max_length=8)
    items_per_cell: int = Field(default=2, ge=1, le=64)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    split: Literal["train", "validation", "test"] = "validation"
    batch_size: int = Field(default=4, ge=1, le=64)


class ContextProfileSpec(StrictSchema):
    """A host profile of one checkpoint, or a feasibility search over variants."""

    mode: Literal["profile", "feasibility"]
    checkpoint_path: str | None = Field(default=None, min_length=1, max_length=512)
    lengths: list[_CONTEXT_LENGTH] = Field(default_factory=list, max_length=16)
    quantize: Literal["none", "int8"] = "none"
    decode_tokens: int = Field(default=16, ge=0, le=4_096)
    config_path: str = Field(
        default="configs/models/epor-tiny.yaml",
        pattern=_MODEL_CONFIG,
        alias="model_config",
        serialization_alias="model_config",
    )
    variants: list[Literal["full", "local", "mla"]] = Field(
        default_factory=lambda: ["full"], max_length=3
    )
    min_length: int = Field(default=1024, ge=64, le=1 << 20)
    max_length: int = Field(default=65_536, ge=128, le=1 << 20)
    vocab_size: int | None = Field(default=None, ge=260, le=1 << 20)
    window: int = Field(default=1024, ge=1, le=1 << 20)
    ratio: int = Field(default=1, ge=1, le=64)
    mla_rank: int = Field(default=32, ge=1, le=4_096)
    mla_rope_dim: int = Field(default=8, ge=2, le=512)
    memory_budget_gib: float | None = Field(default=None, gt=0, le=1024)
    timeout_seconds: float = Field(default=300.0, ge=1, le=24 * 3600)

    @model_validator(mode="after")
    def mode_inputs(self) -> ContextProfileSpec:
        if self.mode == "profile" and (self.checkpoint_path is None or not self.lengths):
            raise ValueError("a profile needs checkpoint_path and lengths")
        if self.mode == "feasibility" and self.min_length > self.max_length:
            raise ValueError("min_length must not exceed max_length")
        return self


class ContextSweepSpec(StrictSchema):
    config_path: str = Field(
        default="configs/models/epor-tiny.yaml",
        pattern=_MODEL_CONFIG,
        alias="model_config",
        serialization_alias="model_config",
    )
    vocab_from_build: bool = False
    axis: Literal["window", "ratio", "rope-base"]
    values: list[float] = Field(min_length=1, max_length=8)
    window: int = Field(default=64, ge=1)
    ratio: int = Field(default=1, ge=1)
    tokenized_id: str = Field(pattern=_SHA256)
    seq_len: int = Field(default=256, ge=32, le=65_536)
    tokens_per_batch: int = Field(default=1_024, ge=32, le=1_048_576)
    max_steps: int = Field(default=50, ge=1, le=100_000)
    learning_rate: float = Field(default=3e-3, gt=0, le=1.0)
    eval_length: int = Field(default=256, ge=96, le=65_536)
    items_per_cell: int = Field(default=2, ge=1, le=16)
    output_name: str = Field(default="context-sweep", pattern=_REPORT_DIRECTORY)


JOB_SPEC_MODELS: dict[JobType, type[StrictSchema]] = {
    JobType.SYSTEM_PROBE: SystemProbeSpec,
    JobType.RESEARCH_SYNC: ResearchSyncSpec,
    JobType.DATA_INGEST: DataIngestSpec,
    JobType.DATA_BUILD: DataBuildSpec,
    JobType.DATA_REMOVE: DataRemoveSpec,
    JobType.DATA_AUDIT: DataAuditSpec,
    JobType.TOKENIZER_TRAIN: TokenizerTrainSpec,
    JobType.TOKENIZER_SHARD: TokenizerShardSpec,
    JobType.TOKENIZER_EVALUATE: TokenizerEvaluateSpec,
    JobType.KERNEL_TRAIN: KernelTrainSpec,
    JobType.KERNEL_EVALUATE: KernelEvaluateSpec,
    JobType.EVAL_SUITE: EvalSuiteSpec,
    JobType.EVAL_CONTAMINATION: EvalContaminationSpec,
    JobType.SCALE_PROFILE: ScaleProfileSpec,
    JobType.SCALE_SWEEP: ScaleSweepSpec,
    JobType.SCALE_MIXTURE: ScaleMixtureSpec,
    JobType.ALIGN_GENERATE: AlignGenerateSpec,
    JobType.ALIGN_JUDGE: AlignJudgeSpec,
    JobType.ALIGN_DATASET: AlignDatasetSpec,
    JobType.ALIGN_SFT: AlignSftSpec,
    JobType.ALIGN_PREFERENCE: AlignPreferenceSpec,
    JobType.ALIGN_EVALUATE: AlignEvaluateSpec,
    JobType.CONTEXT_EXTEND: ContextExtendSpec,
    JobType.CONTEXT_EVALUATE: ContextEvaluateSpec,
    JobType.CONTEXT_PROFILE: ContextProfileSpec,
    JobType.CONTEXT_SWEEP: ContextSweepSpec,
    JobType.TINY_TRAIN: TinyTrainSpec,
    JobType.TINY_EVAL: TinyEvalSpec,
}


class JobCreate(StrictSchema):
    # A bounded string, not a JobType. An action EPOR does not declare must
    # reach the covenant and be turned away by it; rejecting it here as a
    # schema violation would decide the case before the covenant saw it.
    type: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    spec: dict[str, Any] = Field(default_factory=dict)


class AuditVerdictCreate(StrictSchema):
    """A human's verdict on one audit item; only a verified identity may send one."""

    verdict: Literal["accept", "reject"]
    note: str = Field(default="", max_length=2_000)


class SessionCreate(StrictSchema):
    """Exchange a credential for a short browser session. Never stored."""

    token: str = Field(min_length=16, max_length=512)


class OperatorCreate(StrictSchema):
    label: str = Field(min_length=1, max_length=120)
    scopes: list[str] = Field(min_length=1, max_length=64)
    expires_at: datetime


class EscalationDecide(StrictSchema):
    approve: bool
    rationale: str = Field(min_length=1, max_length=2_000)


class OperatorRead(BaseModel):
    """A delegation as anyone may see it. No credential value, no digest."""

    id: str
    role: str
    label: str
    scopes: list[str]
    active: bool
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    rotated_at: datetime | None


class OperatorList(BaseModel):
    items: list[OperatorRead]
    count: int


class IssuedCredentialRead(BaseModel):
    """The one and only response that ever carries a credential value.

    It goes to the owner who just minted it and is never retrievable again;
    only its digest is stored.
    """

    operator: OperatorRead
    token: str
    shown_once: Literal[True] = True


class IdentityRead(BaseModel):
    role: Literal["owner", "delegated_operator", "anonymous"]
    principal_id: str | None = None
    label: str = ""
    scopes: list[str] = Field(default_factory=list)
    authenticated: bool = False
    expires_at: datetime | None = None


class ObligationRead(BaseModel):
    key: str
    statement: str


class PrincipleRead(BaseModel):
    priority: int
    key: str
    title: str
    origin: str
    law: str
    obligations: list[ObligationRead]


class RatificationRead(BaseModel):
    present: bool
    matches_covenant: bool
    ratified_on: str | None = None
    reviewer_role: str | None = None
    roadmap_version: str | None = None
    rationale: str | None = None
    evidence: list[str] = Field(default_factory=list)


class EnforcementCheckpointRead(BaseModel):
    name: str
    description: str


class CovenantRead(BaseModel):
    covenant_id: str
    covenant_version: int
    roadmap_version: str
    effective_date: str
    status: Literal["draft", "ratified"]
    sha256: str
    escalation_confidence_floor: float
    principles: list[PrincipleRead]
    limitations: str
    ratification: RatificationRead
    enforcement_checkpoints: list[EnforcementCheckpointRead]


class AssessmentRead(BaseModel):
    priority: int
    status: str
    confidence: float
    rationale: str
    obligation: str | None = None


class ActionRead(BaseModel):
    action_id: str
    summary: str
    outcome: Literal["allow", "refuse", "escalate"]
    job_type: bool
    cli_commands: list[str]
    assessments: list[AssessmentRead]


class ActionList(BaseModel):
    items: list[ActionRead]
    count: int
    undeclared_behavior: str = (
        "An action with no reviewed declaration is assessed as uncertain against "
        "every principle, so it escalates to a human rather than running."
    )


class EscalationRead(BaseModel):
    """A sanitized escalation. The submitted specification is never included."""

    id: str
    actor_id: str
    actor_role: str
    action_id: str
    request_digest: str
    covenant_sha256: str
    binding_priority: int | None
    reasons: list[str]
    state: Literal["open", "approved", "refused", "consumed", "expired"]
    decided_by: str | None
    rationale: str | None
    created_at: datetime
    decided_at: datetime | None
    approval_expires_at: datetime | None
    consumed_at: datetime | None
    consumed_job_id: str | None


class EscalationList(BaseModel):
    items: list[EscalationRead]
    count: int


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
    # Null on jobs admitted before v0.0.2 introduced identity.
    submitted_by_id: str | None = None
    submitted_by_role: str | None = None
    admission_covenant_sha256: str | None = None
    escalation_id: str | None = None
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
    integrity: Literal["pinned", "content-pinned", "unpinned"]
    sync_status: Literal["verified", "missing", "error", "unknown"]
    raw_sha256: str | None = None
    tracked_raw_sha256: str | None = None
    tracked_content_sha256: str | None = None
    content_scope: Literal["document", "article"] | None = None
    content_profile: Literal["html-document-text-v1", "html-article-text-v1"] | None = None
    size_bytes: int | None = None
    status_message: str = ""


class ResearchCatalogRead(BaseModel):
    schema_version: int | None
    available: bool
    catalog_path: str = "research/catalog.yaml"
    sources: list[ResearchSourceRead]
    count: int
    error: str | None = None


class DataSourceRead(BaseModel):
    id: str
    title: str
    owner_or_steward: str
    license_id: str
    allowed_uses: list[str]
    sensitive_content_risk: str


class DataSummaryRead(BaseModel):
    schema_version: int
    registrations: int
    documents: int
    admitted: int
    quarantined: int
    tombstones: int
    builds: int
    integrity_errors: list[str]
    latest_builds: dict[str, str]
    sources: list[DataSourceRead]


class TokenizerRead(BaseModel):
    tokenizer_id: str
    tokenizer_version: str
    reserved_spec_version: str
    declared_vocab_size: int
    learned_vocab_size: int
    merge_count: int
    dataset_id: str
    build_id: str
    train_split_sha256: str
    train_documents: int
    train_bytes: int
    min_frequency: int
    library_version: str


class TokenizedSplitRead(BaseModel):
    document_count: int
    excluded_document_count: int
    token_count: int
    shard_count: int


class TokenizedBuildRead(BaseModel):
    tokenized_id: str
    tokenizer_id: str
    dataset_id: str
    build_id: str
    max_shard_tokens: int
    total_tokens: int
    total_documents: int
    excluded_document_count: int
    applied_tombstone_ids: list[str]
    shard_count: int
    splits: dict[str, TokenizedSplitRead]


class FertilityMetricsRead(BaseModel):
    documents: int
    bytes: int
    characters: int
    words: int
    tokens: int
    byte_fallback_tokens: int
    tokens_per_byte: float | None
    tokens_per_word: float | None
    bytes_per_token: float | None
    byte_fallback_rate: float | None


class FertilityReportRead(BaseModel):
    report_id: str
    schema_version: int
    kind: str
    scope: str
    tokenizer_id: str
    tokenizer_json_sha256: str
    reserved_spec_sha256: str
    vocab_size: int
    learned_vocab_size: int
    dataset_id: str
    build_id: str
    split: str
    split_sha256: str
    in_sample: bool
    excluded_document_count: int
    applied_tombstone_ids: list[str]
    round_trip_failures: int
    overall: FertilityMetricsRead
    by_domain: dict[str, FertilityMetricsRead]
    by_language: dict[str, FertilityMetricsRead]


class TokenizerSummaryRead(BaseModel):
    schema_version: int
    tokenizers: list[TokenizerRead]
    tokenized_builds: list[TokenizedBuildRead]
    fertility_reports: list[FertilityReportRead]
    integrity_errors: list[str]


class KernelEvaluationRead(BaseModel):
    split: str
    main_loss: float
    perplexity: float
    mtp_losses: list[float]
    target_tokens: int
    steps: int


class KernelCheckpointRead(BaseModel):
    name: str
    step: int
    world_size: int | None
    checkpoint_sha256: str


class KernelRunRead(BaseModel):
    run_id: str
    lineage_id: str | None
    parent_run_id: str | None
    source: Literal["job", "cli"]
    location: str
    status: str
    termination_reason: str | None
    started_at: str | None
    finished_at: str | None
    model_id: str | None
    config_sha256: str | None
    qk_norm: bool
    mtp_depth: int
    precision: str | None
    strategy: str | None
    world_size: int | None
    tokenized_id: str | None
    seq_len: int | None
    tokens_per_batch: int | None
    start_step: int | None
    target_steps: int | None
    step: int | None
    training_loss: float | None
    main_loss: float | None
    mtp_losses: list[float]
    learning_rate: float | None
    tokens_seen: int | None
    tokens_per_second: float | None
    evaluation: KernelEvaluationRead | None
    checkpoint: KernelCheckpointRead | None
    manifest_sha256: str
    # v0.0.7: SFT, distillation, and DPO runs share this listing.
    objective: Literal["pretrain", "sft", "logit-distillation", "dpo"] = "pretrain"
    chat_dataset_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    reference_checkpoint_sha256: str | None = None
    distill_teachers: int = 0
    distill_kl: float | None = None
    reward_margin: float | None = None
    preference_accuracy: float | None = None
    policy_logratio: float | None = None
    alerts: list[str] = Field(default_factory=list)
    preference_evaluation: dict[str, Any] | None = None


class KernelEvaluationReportRead(BaseModel):
    location: str
    report_sha256: str
    checkpoint_step: int | None
    model_id: str | None
    tokenized_id: str | None
    split: str | None
    main_loss: float | None
    perplexity: float | None
    mtp_losses: list[float]
    target_tokens: int | None
    steps: int | None
    created_at: str | None


class TrainingSummaryRead(BaseModel):
    schema_version: int
    scope: str
    runs: list[KernelRunRead]
    evaluations: list[KernelEvaluationReportRead]
    integrity_errors: list[str]


class PrimaryMetricRead(BaseModel):
    task_id: str
    task_type: str
    metric: str
    value: float | None
    scope: str


class EvaluationReportSummaryRead(BaseModel):
    report_id: str
    suite_id: str
    suite_version: int
    scope: str
    claim_boundary: str
    model_kind: str
    model_id: str
    checkpoint_step: int | None
    parameter_count: int | None
    task_count: int
    skipped_task_count: int
    primary_metrics: list[PrimaryMetricRead]
    code_executed: bool


class EvaluationReportList(BaseModel):
    schema_version: int = 1
    reports: list[EvaluationReportSummaryRead]
    integrity_errors: list[str]


class ScalingReportSummaryRead(BaseModel):
    """Common summary fields plus the kind's headline figures (extra keys)."""

    model_config = ConfigDict(extra="allow")

    report_id: str
    content_sha256: str
    kind: str
    scope: str
    claim_boundary: str
    warning_count: int
    recorded_at: str | None


class ScalingReportList(BaseModel):
    schema_version: int = 1
    reports: list[ScalingReportSummaryRead]
    integrity_errors: list[str]


class LadderRungRead(BaseModel):
    slug: str
    config_sha256: str
    nominal_non_embedding: int | None
    total_parameters: int
    non_embedding_parameters: int
    vocab_size: int
    execution: Literal["local", "projection-only"]
    reason: str


class LadderRead(BaseModel):
    rungs: list[LadderRungRead]
    note: str


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
    family: Literal["alpha", "beta", "gamma"]
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
