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
        """Return stable JSON used for config hashing and manifests."""

        return json.dumps(
            self.model_dump(mode="json"),
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
    head_dimension = hidden // config.n_heads
    key_value_width = config.n_kv_heads * head_dimension
    attention = 2 * hidden * hidden + 2 * hidden * key_value_width
    normalization = 2 * hidden
    expert = 3 * hidden * config.d_ff
    if config.architecture == "moe-decoder":
        feed_forward = hidden * config.n_routed_experts + expert * (
            config.n_routed_experts + config.n_shared_experts
        )
    else:
        feed_forward = expert
    embeddings = config.vocab_size * hidden
    output_head = 0 if config.tie_embeddings else embeddings
    return (
        embeddings
        + output_head
        + config.n_layers * (attention + normalization + feed_forward)
        + hidden
    )


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
