"""Compute, storage, energy, and cost projections for a target-size training run.

A projection answers the target-proposal list in ``docs/COMPUTE_POLICY.md``
(approval ladder step 4) — tokens, GPU-hours, provider SKU/count, storage,
egress, evaluation, checkpoint cadence, failure reserve, carbon/energy where
available, and a hard budget/abort envelope — plus G5's MFU, throughput, and
loss-curve expectation.  It is arithmetic over **operator-supplied** inputs
(:class:`ProjectionInputs`, loaded from a YAML file the operator writes; see
``configs/scaling/projection-inputs.example.yaml``).  EPOR never fetches a
price or a hardware number, never provisions anything, and a projection is
not an approval: the report states that a named human must approve the exact
immutable proposal.

Every quantity is an :class:`Estimate` with a value, unit, evidence label,
basis, and the operator inputs it is missing.  A missing input makes every
quantity that depends on it ``unknown`` (``value: null``); nothing is
silently defaulted.  Two documented defaults exist, both labelled:

* ``tokens_per_parameter`` = 20 when the operator gives neither a token budget
  nor a ratio — Hoffmann et al. 2022's reported compute-optimal ratio
  (``disclosed`` there), applied to EPOR as a ``hypothesis``;
* ``failure_reserve_fraction`` = 0.25 — the EPOR policy default reserve for
  failed, restarted, and aborted work (``EPOR-adaptation``: the compute policy
  requires a reserve and requires failed runs to be counted, but no published
  source fixes the fraction).  Override it in the inputs file.

Arithmetic (all ``EPOR-adaptation`` unless noted):

* training FLOPs ``C = train_flops_per_token x D`` (:mod:`epor.scaling.flops`,
  Kaplan ``disclosed`` formula) at the operator's sequence length;
* device-hours ``= C / (peak_flops x assumed_mfu) / 3600``; wall-clock hours
  ``= device-hours / device_count``;
* evaluation ``= (D / evaluation_every_tokens) x evaluation_tokens_per_pass x
  forward FLOPs/token``, at the same effective FLOP/s;
* training-state memory ``= total parameters x 16 bytes`` (FP32 weights,
  gradients, and two AdamW moments; activations excluded);
* checkpoint size ``= total parameters x 12 bytes`` (FP32 weights and two
  AdamW moments, the kernel's checkpoint content); checkpoints written
  ``= ceil(wall-clock hours / checkpoint_every_hours)``; retained storage
  ``= min(written, keep_checkpoints) x size``;
* token-shard storage ``= D x 2 bytes`` (``uint16`` shards), an upper bound
  that assumes every training token is unique;
* cost ``= (training + evaluation device-hours) x price_per_device_hour +
  retained storage x storage_months x price + egress x price``; reserve ``=
  cost x failure_reserve_fraction``; envelope = cost + reserve;
* energy ``= device-hours x device_power_kw x pue``; carbon ``= energy x
  carbon_intensity``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from epor.models.config import ModelConfig

from .flops import estimate_flops, parameter_counts
from .ladder import rung_status
from .report import (
    CHINCHILLA_2022,
    KAPLAN_2020,
    EvidenceLabel,
    ScalingReport,
    seal_scaling_report,
)

DEFAULT_TOKENS_PER_PARAMETER = 20.0
DEFAULT_FAILURE_RESERVE_FRACTION = 0.25
TRAINING_STATE_BYTES_PER_PARAMETER = 16
CHECKPOINT_BYTES_PER_PARAMETER = 12
TOKEN_SHARD_BYTES_PER_TOKEN = 2
GIB = 1024**3

MANDATORY_ABORT_CONDITIONS = (
    "loss is non-finite or outside the approved envelope",
    "data rights change",
    "contamination or secret leakage is found",
    "checkpoint restore fails",
    "expert routing drops tokens or produces dead experts",
    "throughput makes the budget invalid",
    "safety monitoring detects an unapproved risk",
    "spend, energy, or time reaches the approved limit",
)


class _Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardwareInputs(_Inputs):
    provider: str | None = None
    device_sku: str | None = None
    device_count: int | None = Field(default=None, ge=1)
    peak_flops_per_device: float | None = Field(default=None, gt=0)
    assumed_mfu: float | None = Field(default=None, gt=0, le=1)
    mfu_source: str | None = None
    device_memory_bytes: int | None = Field(default=None, gt=0)
    device_power_kw: float | None = Field(default=None, gt=0)


class PriceInputs(_Inputs):
    currency: str | None = None
    price_per_device_hour: float | None = Field(default=None, ge=0)
    storage_price_per_gib_month: float | None = Field(default=None, ge=0)
    egress_price_per_gib: float | None = Field(default=None, ge=0)
    price_source: str | None = None


class EnergyInputs(_Inputs):
    pue: float | None = Field(default=None, ge=1)
    carbon_intensity_kg_per_kwh: float | None = Field(default=None, ge=0)


class RunInputs(_Inputs):
    sequence_length: int | None = Field(default=None, ge=1)
    token_budget: int | None = Field(default=None, ge=1)
    tokens_per_parameter: float | None = Field(default=None, gt=0)
    checkpoint_every_hours: float | None = Field(default=None, gt=0)
    keep_checkpoints: int | None = Field(default=None, ge=1)
    evaluation_every_tokens: int | None = Field(default=None, ge=1)
    evaluation_tokens_per_pass: int | None = Field(default=None, ge=0)
    egress_gib: float | None = Field(default=None, ge=0)
    storage_months: float | None = Field(default=None, gt=0)
    failure_reserve_fraction: float | None = Field(default=None, ge=0, le=5)


class ProjectionInputs(_Inputs):
    schema_version: Literal["epor-projection-inputs-v1"] = "epor-projection-inputs-v1"
    hardware: HardwareInputs = Field(default_factory=HardwareInputs)
    prices: PriceInputs = Field(default_factory=PriceInputs)
    energy: EnergyInputs = Field(default_factory=EnergyInputs)
    run: RunInputs = Field(default_factory=RunInputs)


def load_projection_inputs(path: str | Path) -> ProjectionInputs:
    """Strictly validate an operator-written inputs file (local file only)."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("projection inputs must be a mapping")
    return ProjectionInputs.model_validate(data)


class Estimate(BaseModel):
    """One projected quantity; ``value`` is ``None`` when it is unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float | None
    unit: str
    label: EvidenceLabel
    basis: str
    missing: list[str] = Field(default_factory=list)


def _input(value: float | None, name: str, unit: str) -> Estimate:
    if value is None:
        return Estimate(
            value=None, unit=unit, label="unknown", basis="operator input", missing=[name]
        )
    return Estimate(value=float(value), unit=unit, label="EPOR-adaptation", basis="operator input")


def _fixed(value: float, unit: str, label: EvidenceLabel, basis: str) -> Estimate:
    return Estimate(value=float(value), unit=unit, label=label, basis=basis)


def _derive(
    unit: str,
    basis: str,
    function: Callable[..., float],
    *dependencies: Estimate,
    label: EvidenceLabel = "EPOR-adaptation",
) -> Estimate:
    missing = sorted({name for item in dependencies for name in item.missing})
    if any(item.value is None for item in dependencies):
        return Estimate(value=None, unit=unit, label="unknown", basis=basis, missing=missing)
    labels = {item.label for item in dependencies}
    final: EvidenceLabel = "hypothesis" if "hypothesis" in labels else label
    value = function(*(item.value for item in dependencies))
    return Estimate(value=float(value), unit=unit, label=final, basis=basis)


def _dump(estimate: Estimate) -> dict[str, Any]:
    return estimate.model_dump(mode="json")


def project_training(
    target: ModelConfig,
    inputs: ProjectionInputs,
    *,
    fit_report: ScalingReport | None = None,
) -> ScalingReport:
    """Project one target run from ``inputs`` (see the module docstring); seal the report."""

    from .fit import predict_loss

    counts = parameter_counts(target)
    hardware, prices, energy, run = inputs.hardware, inputs.prices, inputs.energy, inputs.run
    status = rung_status(target)

    # Tokens.
    if run.token_budget is not None:
        tokens = _fixed(run.token_budget, "tokens", "EPOR-adaptation", "operator token budget")
        ratio = None
    else:
        ratio_value = run.tokens_per_parameter or DEFAULT_TOKENS_PER_PARAMETER
        ratio_basis = (
            "operator tokens_per_parameter"
            if run.tokens_per_parameter is not None
            else "default compute-optimal ratio (Chinchilla, disclosed) applied to EPOR"
        )
        ratio = _fixed(ratio_value, "tokens/parameter", "hypothesis", ratio_basis)
        tokens = _fixed(
            ratio_value * counts.active,
            "tokens",
            "hypothesis",
            "tokens_per_parameter x active parameters",
        )

    # FLOPs.
    sequence = _input(run.sequence_length, "run.sequence_length", "tokens")
    per_token = (
        estimate_flops(target, run.sequence_length) if run.sequence_length is not None else None
    )
    train_per_token = (
        _fixed(per_token.train_flops_per_token, "FLOP/token", "disclosed", "Kaplan 6N + attention")
        if per_token is not None
        else _derive("FLOP/token", "requires sequence length", lambda s: s, sequence)
    )
    forward_per_token = (
        _fixed(
            per_token.forward_flops_per_token, "FLOP/token", "disclosed", "Kaplan 2N + attention"
        )
        if per_token is not None
        else _derive("FLOP/token", "requires sequence length", lambda s: s, sequence)
    )
    training_flops = _derive(
        "FLOP", "train_flops_per_token x tokens", lambda f, d: f * d, train_per_token, tokens
    )

    # Throughput and device-hours.
    peak = _input(hardware.peak_flops_per_device, "hardware.peak_flops_per_device", "FLOP/s")
    mfu = _input(hardware.assumed_mfu, "hardware.assumed_mfu", "fraction")
    count = _input(hardware.device_count, "hardware.device_count", "devices")
    effective = _derive("FLOP/s", "peak x assumed MFU per device", lambda p, m: p * m, peak, mfu)
    training_hours = _derive(
        "device-hours",
        "training FLOPs / effective FLOP/s / 3600",
        lambda c, e: c / e / 3600.0,
        training_flops,
        effective,
    )
    tokens_per_second = _derive(
        "tokens/s",
        "devices x effective FLOP/s / train FLOPs per token",
        lambda n, e, f: n * e / f,
        count,
        effective,
        train_per_token,
    )

    # Evaluation.
    every = _input(run.evaluation_every_tokens, "run.evaluation_every_tokens", "tokens")
    per_pass = _input(run.evaluation_tokens_per_pass, "run.evaluation_tokens_per_pass", "tokens")
    evaluation_flops = _derive(
        "FLOP",
        "(tokens / evaluation_every_tokens) x tokens per pass x forward FLOPs/token",
        lambda d, e, p, f: math.floor(d / e) * p * f,
        tokens,
        every,
        per_pass,
        forward_per_token,
    )
    evaluation_hours = _derive(
        "device-hours",
        "evaluation FLOPs / effective FLOP/s / 3600",
        lambda c, e: c / e / 3600.0,
        evaluation_flops,
        effective,
    )
    total_hours = _derive(
        "device-hours",
        "training + evaluation",
        lambda a, b: a + b,
        training_hours,
        evaluation_hours,
    )
    wall_hours = _derive(
        "hours", "total device-hours / device count", lambda h, n: h / n, total_hours, count
    )

    # Memory.
    state_bytes = _fixed(
        counts.total * TRAINING_STATE_BYTES_PER_PARAMETER,
        "bytes",
        "EPOR-adaptation",
        "FP32 weights + gradients + two AdamW moments; activations excluded",
    )
    device_memory = _input(hardware.device_memory_bytes, "hardware.device_memory_bytes", "bytes")
    aggregate_memory = _derive(
        "bytes", "device memory x device count", lambda m, n: m * n, device_memory, count
    )
    state_fraction = _derive(
        "fraction",
        "training state / aggregate device memory (fully sharded)",
        lambda s, m: s / m,
        state_bytes,
        aggregate_memory,
    )

    # Checkpoints and storage.
    checkpoint_bytes = _fixed(
        counts.total * CHECKPOINT_BYTES_PER_PARAMETER,
        "bytes",
        "EPOR-adaptation",
        "FP32 weights + two AdamW moments per checkpoint",
    )
    cadence = _input(run.checkpoint_every_hours, "run.checkpoint_every_hours", "hours")
    keep = _input(run.keep_checkpoints, "run.keep_checkpoints", "checkpoints")
    written = _derive(
        "checkpoints",
        "ceil(wall-clock hours / cadence)",
        lambda w, c: math.ceil(w / c),
        wall_hours,
        cadence,
    )
    retained_bytes = _derive(
        "bytes",
        "min(written, keep) x checkpoint size",
        lambda w, k, b: min(w, k) * b,
        written,
        keep,
        checkpoint_bytes,
    )
    shard_bytes = _derive(
        "bytes",
        "tokens x 2 bytes (uint16 shards; upper bound, every token unique)",
        lambda d: d * TOKEN_SHARD_BYTES_PER_TOKEN,
        tokens,
    )
    storage_bytes = _derive(
        "bytes",
        "retained checkpoints + token shards",
        lambda a, b: a + b,
        retained_bytes,
        shard_bytes,
    )
    months = _input(run.storage_months, "run.storage_months", "months")
    storage_price = _input(
        prices.storage_price_per_gib_month,
        "prices.storage_price_per_gib_month",
        "currency/GiB-month",
    )
    storage_cost = _derive(
        "currency",
        "storage GiB x months x price",
        lambda b, m, p: b / GIB * m * p,
        storage_bytes,
        months,
        storage_price,
    )
    egress = _input(run.egress_gib, "run.egress_gib", "GiB")
    egress_price = _input(
        prices.egress_price_per_gib, "prices.egress_price_per_gib", "currency/GiB"
    )
    egress_cost = _derive(
        "currency", "egress GiB x price", lambda g, p: g * p, egress, egress_price
    )

    # Cost, reserve, envelope.
    hour_price = _input(
        prices.price_per_device_hour, "prices.price_per_device_hour", "currency/device-hour"
    )
    compute_cost = _derive(
        "currency", "total device-hours x price", lambda h, p: h * p, total_hours, hour_price
    )
    subtotal = _derive(
        "currency",
        "compute + storage + egress",
        lambda a, b, c: a + b + c,
        compute_cost,
        storage_cost,
        egress_cost,
    )
    reserve_fraction = (
        _fixed(run.failure_reserve_fraction, "fraction", "EPOR-adaptation", "operator input")
        if run.failure_reserve_fraction is not None
        else _fixed(
            DEFAULT_FAILURE_RESERVE_FRACTION,
            "fraction",
            "EPOR-adaptation",
            "EPOR policy default failure reserve",
        )
    )
    reserve_cost = _derive(
        "currency", "subtotal x failure reserve", lambda s, f: s * f, subtotal, reserve_fraction
    )
    total_cost = _derive(
        "currency", "subtotal + reserve", lambda s, r: s + r, subtotal, reserve_cost
    )
    reserve_hours = _derive(
        "device-hours",
        "total device-hours x (1 + reserve)",
        lambda h, f: h * (1 + f),
        total_hours,
        reserve_fraction,
    )
    reserve_wall = _derive(
        "hours",
        "wall-clock hours x (1 + reserve)",
        lambda h, f: h * (1 + f),
        wall_hours,
        reserve_fraction,
    )

    # Energy and carbon.
    power = _input(hardware.device_power_kw, "hardware.device_power_kw", "kW")
    pue = _input(energy.pue, "energy.pue", "ratio")
    energy_kwh = _derive(
        "kWh",
        "device-hours x device power x PUE",
        lambda h, p, u: h * p * u,
        total_hours,
        power,
        pue,
    )
    intensity = _input(
        energy.carbon_intensity_kg_per_kwh, "energy.carbon_intensity_kg_per_kwh", "kgCO2e/kWh"
    )
    carbon = _derive(
        "kgCO2e", "energy x carbon intensity", lambda e, c: e * c, energy_kwh, intensity
    )

    loss = (
        predict_loss(fit_report.results, float(counts.non_embedding), tokens.value)
        if fit_report is not None
        else {"value": None, "label": "unknown", "basis": "no scaling fit supplied"}
    )

    results: dict[str, Any] = {
        "target": {
            "slug": target.slug,
            "config_sha256": target.sha256,
            "architecture": target.architecture,
            "parameters": counts.model_dump(mode="json"),
            "local_execution": status.execution,
        },
        "tokens": {
            "training_tokens": _dump(tokens),
            "tokens_per_parameter": None if ratio is None else _dump(ratio),
        },
        "training_flops": {
            "sequence_length": _dump(sequence),
            "train_flops_per_token": _dump(train_per_token),
            "forward_flops_per_token": _dump(forward_per_token),
            "total": _dump(training_flops),
        },
        "gpu_hours": {
            "effective_flops_per_device": _dump(effective),
            "assumed_mfu": _dump(mfu),
            "training_device_hours": _dump(training_hours),
            "evaluation_device_hours": _dump(evaluation_hours),
            "total_device_hours": _dump(total_hours),
            "wall_clock_hours": _dump(wall_hours),
            "projected_tokens_per_second": _dump(tokens_per_second),
        },
        "provider": {
            "provider": hardware.provider,
            "device_sku": hardware.device_sku,
            "device_count": _dump(count),
            "label": "unknown" if hardware.device_sku is None else "EPOR-adaptation",
        },
        "memory": {
            "training_state_bytes": _dump(state_bytes),
            "aggregate_device_memory_bytes": _dump(aggregate_memory),
            "state_fraction_of_device_memory": _dump(state_fraction),
        },
        "storage": {
            "checkpoint_bytes": _dump(checkpoint_bytes),
            "retained_checkpoint_bytes": _dump(retained_bytes),
            "token_shard_bytes": _dump(shard_bytes),
            "total_bytes": _dump(storage_bytes),
            "cost": _dump(storage_cost),
        },
        "egress": {"gib": _dump(egress), "cost": _dump(egress_cost)},
        "evaluation": {
            "every_tokens": _dump(every),
            "tokens_per_pass": _dump(per_pass),
            "flops": _dump(evaluation_flops),
            "device_hours": _dump(evaluation_hours),
        },
        "checkpoint_cadence": {
            "every_hours": _dump(cadence),
            "keep": _dump(keep),
            "written": _dump(written),
        },
        "failure_reserve": {
            "fraction": _dump(reserve_fraction),
            "cost": _dump(reserve_cost),
            "device_hours_with_reserve": _dump(reserve_hours),
        },
        "energy_carbon": {"energy_kwh": _dump(energy_kwh), "carbon_kg": _dump(carbon)},
        "cost": {
            "currency": prices.currency,
            "compute": _dump(compute_cost),
            "subtotal": _dump(subtotal),
            "total_with_reserve": _dump(total_cost),
        },
        "budget_abort_envelope": {
            "max_spend": _dump(total_cost),
            "max_device_hours": _dump(reserve_hours),
            "max_wall_clock_hours": _dump(reserve_wall),
            "expected_final_loss": loss,
            "mandatory_abort_conditions": list(MANDATORY_ABORT_CONDITIONS),
        },
        "approval": {
            "required": True,
            "approval_id": None,
            "statement": (
                "A projection is not authorization. A named human must approve the exact "
                "immutable proposal; a material change requires a new approval."
            ),
        },
    }
    missing = sorted(
        {
            name
            for estimate in (
                tokens,
                training_flops,
                total_hours,
                wall_hours,
                storage_bytes,
                storage_cost,
                egress_cost,
                compute_cost,
                total_cost,
                energy_kwh,
                carbon,
                state_fraction,
                written,
                tokens_per_second,
            )
            for name in estimate.missing
        }
    )
    results["unknown_inputs"] = missing
    warnings = [
        "Projection only: no quantity here is measured, trained, or validated, and nothing "
        "has been provisioned."
    ]
    if missing:
        warnings.append(
            "Missing operator inputs leave dependent quantities unknown: " + ", ".join(missing)
        )
    if run.token_budget is None:
        warnings.append(
            "Token budget derived from a tokens-per-parameter ratio (hypothesis), not an "
            "operator decision."
        )
    if hardware.assumed_mfu is not None and not hardware.mfu_source:
        warnings.append("assumed_mfu has no recorded source; a measured 1B-pilot MFU is required.")
    if fit_report is not None:
        warnings.append(
            "Expected loss is extrapolated from a proxy-scale fit (hypothesis); see its caveat."
        )
    return seal_scaling_report(
        kind="compute-projection",
        scope="projection",
        inputs={
            "operator_inputs": inputs.model_dump(mode="json"),
            "target_config_sha256": target.sha256,
            "fit_report_id": None if fit_report is None else fit_report.report_id,
            "defaults": {
                "tokens_per_parameter": DEFAULT_TOKENS_PER_PARAMETER,
                "failure_reserve_fraction": DEFAULT_FAILURE_RESERVE_FRACTION,
            },
        },
        results=results,
        warnings=warnings,
        citations=[KAPLAN_2020, CHINCHILLA_2022],
    )


__all__ = [
    "CHECKPOINT_BYTES_PER_PARAMETER",
    "DEFAULT_FAILURE_RESERVE_FRACTION",
    "DEFAULT_TOKENS_PER_PARAMETER",
    "MANDATORY_ABORT_CONDITIONS",
    "EnergyInputs",
    "Estimate",
    "HardwareInputs",
    "PriceInputs",
    "ProjectionInputs",
    "RunInputs",
    "load_projection_inputs",
    "project_training",
]
