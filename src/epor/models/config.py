"""Validated, serializable model configuration contracts.

The configuration deliberately distinguishes configured, trained, validated, and
operational context lengths.  A large configured RoPE window is not evidence that
the model has been trained or evaluated at that length.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]

# DeepSeek-V3 reports an MTP loss weight of 0.3 for its first 10T tokens and
# 0.1 afterwards (``deepseek-v3``, ``disclosed``).  EPOR starts from the first
# value; whether MTP is worth its memory at EPOR scale is decided at G3/G5.
DEFAULT_MTP_LOSS_WEIGHT = 0.3

# Architecture fields added after configurations were already hashed into run
# manifests and checkpoints.  At these values they select the original
# computation, so ``canonical_json`` omits them and every pre-existing config
# keeps its hash.  A non-default value changes the computation and therefore
# appears in, and changes, the hash.
HASH_NEUTRAL_DEFAULTS: dict[str, object] = {
    "qk_norm": False,
    "mtp_depth": 0,
    "mtp_loss_weight": DEFAULT_MTP_LOSS_WEIGHT,
}


class CertifiedProfile(BaseModel):
    """A hardware-specific context profile, planned or measured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    hardware: str = Field(min_length=1)
    max_context: PositiveInt
    precision: str = Field(min_length=1)
    status: Literal["planned", "validated", "certified"] = "planned"
    ram_gib: float | None = Field(default=None, gt=0)
    vram_gib: float | None = Field(default=None, gt=0)
    notes: str | None = None


class ModelConfig(BaseModel):
    """Canonical EPOR decoder configuration.

    ``total_parameters`` and related fields are declared research targets.  Use
    :func:`epor.models.reference.parameter_report` to obtain counts from the
    actual implementation.  Keeping both values makes architectural drift visible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    display_name: str = Field(min_length=1)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    family: Literal["alpha", "beta", "gamma", "reference"]
    architecture: Literal["dense-decoder", "moe-decoder"] = "dense-decoder"

    vocab_size: PositiveInt
    d_model: PositiveInt
    n_layers: PositiveInt
    n_heads: PositiveInt
    n_kv_heads: PositiveInt
    d_ff: PositiveInt

    rope_theta: float = Field(default=10_000.0, gt=0)
    rms_norm_eps: float = Field(default=1e-5, gt=0)
    initializer_range: float = Field(default=0.02, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    tie_embeddings: bool = True

    # Per-head RMSNorm on queries and keys before RoPE (``gemma-3-technical-report``,
    # ``disclosed``).  False is the pre-norm control block.
    qk_norm: bool = False
    # DeepSeek-V3-style sequential multi-token prediction (``deepseek-v3``,
    # ``disclosed``).  Zero disables it; each depth adds one projection and one
    # transformer block that share the embedding and output head.
    mtp_depth: NonNegativeInt = 0
    mtp_loss_weight: float = Field(default=DEFAULT_MTP_LOSS_WEIGHT, ge=0.0, le=1.0)

    pad_token_id: NonNegativeInt = 0
    bos_token_id: NonNegativeInt = 1
    eos_token_id: NonNegativeInt = 2

    configured_max_context: PositiveInt
    trained_max_context: NonNegativeInt = 0
    validated_max_context: NonNegativeInt = 0
    operational_default_context: NonNegativeInt = 0
    certified_profiles: tuple[CertifiedProfile, ...] = ()

    # Declared family targets, not measurements.
    total_parameters: PositiveInt | None = None
    active_parameters: PositiveInt | None = None
    core_parameters: PositiveInt | None = None

    # MoE fields are zero for a dense decoder.
    n_routed_experts: NonNegativeInt = 0
    n_shared_experts: NonNegativeInt = 0
    experts_per_token: NonNegativeInt = 0

    # Research metadata has no execution semantics in v0.0.1.
    elastic_width_multipliers: tuple[float, ...] = ()
    research_features: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_architecture(self) -> ModelConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")

        token_ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("pad, BOS, and EOS token IDs must be distinct")
        if max(token_ids) >= self.vocab_size:
            raise ValueError("special token IDs must be smaller than vocab_size")

        if self.trained_max_context > self.configured_max_context:
            raise ValueError("trained_max_context cannot exceed configured_max_context")
        if self.validated_max_context > self.trained_max_context:
            raise ValueError("validated_max_context cannot exceed trained_max_context")
        if self.operational_default_context > self.validated_max_context:
            raise ValueError("operational_default_context cannot exceed validated_max_context")
        profile_ids: set[str] = set()
        for profile in self.certified_profiles:
            if profile.profile_id in profile_ids:
                raise ValueError(f"certified profile ID {profile.profile_id!r} is duplicated")
            profile_ids.add(profile.profile_id)
            if profile.status in {"validated", "certified"}:
                if profile.max_context > self.validated_max_context:
                    raise ValueError(f"profile {profile.profile_id!r} exceeds validated context")
            elif profile.max_context > self.configured_max_context:
                raise ValueError(
                    f"planned profile {profile.profile_id!r} exceeds configured context"
                )

        if self.mtp_depth >= self.configured_max_context:
            raise ValueError("mtp_depth must be smaller than configured_max_context")
        if self.mtp_depth == 0 and self.mtp_loss_weight != DEFAULT_MTP_LOSS_WEIGHT:
            raise ValueError("mtp_loss_weight has no effect unless mtp_depth is positive")

        has_moe_fields = any((self.n_routed_experts, self.n_shared_experts, self.experts_per_token))
        if self.architecture == "dense-decoder" and has_moe_fields:
            raise ValueError("dense decoders must not configure MoE experts")
        if self.architecture == "moe-decoder":
            if self.n_routed_experts < 1:
                raise ValueError("MoE decoders require at least one routed expert")
            if not 1 <= self.experts_per_token <= self.n_routed_experts:
                raise ValueError("experts_per_token must be between one and n_routed_experts")

        if (
            self.active_parameters
            and self.total_parameters
            and self.active_parameters > self.total_parameters
        ):
            raise ValueError("active_parameters cannot exceed total_parameters")
        if (
            self.core_parameters
            and self.total_parameters
            and self.core_parameters > self.total_parameters
        ):
            raise ValueError("core_parameters cannot exceed total_parameters")
        if self.elastic_width_multipliers:
            if tuple(sorted(set(self.elastic_width_multipliers))) != (
                self.elastic_width_multipliers
            ):
                raise ValueError("elastic_width_multipliers must be unique and sorted")
            if self.elastic_width_multipliers[-1] != 1.0:
                raise ValueError("elastic widths must end at the full-width 1.0 model")
            if any(not 0.0 < width <= 1.0 for width in self.elastic_width_multipliers):
                raise ValueError("elastic widths must be in the interval (0, 1]")
        return self

    def canonical_json(self) -> str:
        """Return stable JSON used for config hashing and manifests.

        Fields listed in :data:`HASH_NEUTRAL_DEFAULTS` are omitted while they
        hold their computation-preserving default, so adding them did not
        change the hash of any existing configuration.
        """

        payload = self.model_dump(mode="json")
        for name, default in HASH_NEUTRAL_DEFAULTS.items():
            if payload[name] == default:
                del payload[name]
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and strictly validate a YAML or JSON model configuration."""

    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    data = json.loads(raw) if config_path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"model config {config_path} must contain a mapping")
    return ModelConfig.model_validate(data)


def validate_config(path: str | Path) -> ModelConfig:
    """Validate ``path`` and return its immutable canonical model config.

    This deliberately does not import PyTorch, so ``epor config validate`` can
    run in the CPU-safe base environment.  Parameter measurement is available
    separately through :func:`epor.models.parameter_report` when the training
    extra is installed.
    """

    return load_model_config(path)


def structural_parameter_count(config: ModelConfig) -> int:
    """Count decoder parameters from validated dimensions without allocating modules.

    Keeping this calculation free of PyTorch lets the control plane reject an
    oversized recipe before even constructing a meta-device module graph.
    """

    hidden = config.d_model
    embeddings = config.vocab_size * hidden
    output_head = 0 if config.tie_embeddings else embeddings
    return (
        embeddings
        + output_head
        + config.n_layers * _block_parameter_count(config)
        + hidden
        + mtp_parameter_count(config)
    )


def _block_parameter_count(config: ModelConfig) -> int:
    hidden = config.d_model
    head_dimension = hidden // config.n_heads
    key_value_width = config.n_kv_heads * head_dimension
    attention = 2 * hidden * hidden + 2 * hidden * key_value_width
    if config.qk_norm:
        attention += 2 * head_dimension
    normalization = 2 * hidden
    expert = 3 * hidden * config.d_ff
    if config.architecture == "moe-decoder":
        feed_forward = hidden * config.n_routed_experts + expert * (
            config.n_routed_experts + config.n_shared_experts
        )
    else:
        feed_forward = expert
    return attention + normalization + feed_forward


def mtp_parameter_count(config: ModelConfig) -> int:
    """Count the extra parameters owned by multi-token-prediction modules.

    Each depth owns two input RMSNorms, a ``2 * d_model -> d_model``
    projection, one decoder block, and an output RMSNorm.  The embedding and
    output head are shared with the main model and are not counted again.
    """

    hidden = config.d_model
    per_depth = 2 * hidden + 2 * hidden * hidden + _block_parameter_count(config) + hidden
    return config.mtp_depth * per_depth


def mtp_logit_bytes_per_token(config: ModelConfig) -> int:
    """FP32 bytes of extra training logits that MTP materializes per token.

    Each depth produces a full-vocabulary logit row per position.  At a 65,536
    vocabulary this activation term usually dominates the parameter overhead.
    """

    return config.mtp_depth * config.vocab_size * 4


def dump_model_config(config: ModelConfig, path: str | Path) -> None:
    """Atomically write a validated model configuration as YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = yaml.safe_dump(
        config.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def model_config_from_mapping(value: Any) -> ModelConfig:
    """Validate an untrusted deserialized mapping as a model configuration."""

    if not isinstance(value, dict):
        raise ValueError("model configuration must be a mapping")
    return ModelConfig.model_validate(value)
