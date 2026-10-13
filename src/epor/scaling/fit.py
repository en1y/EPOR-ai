"""Deterministic scaling-law fits: L(N), L(D), and the joint Chinchilla form.

Formulas (``disclosed``):

* ``L(N) = E + A N^-alpha`` and ``L(D) = E + B D^-beta`` — power laws with an
  irreducible term (Kaplan et al. 2020, arXiv:2001.08361, with non-embedding
  ``N``; the ``E`` term as in Hoffmann et al. 2022).
* ``L(N, D) = E + A N^-alpha + B D^-beta`` (Hoffmann et al. 2022,
  arXiv:2203.15556, "Chinchilla").

Fitting procedure (``EPOR-adaptation``; Hoffmann et al. minimize a Huber loss
on log-loss residuals with L-BFGS from a grid of starts): for fixed exponents
the model is linear in ``E``, ``A`` (and ``B``), so the fit solves weighted
least squares in closed form for every exponent on a grid, then refines the
best grid cell on a finer grid.  Weights are ``1 / L^2``, i.e. relative
residuals, which approximate log-space residuals.  ``E >= 0`` and positive
coefficients are enforced (a negative ``E`` refits with ``E = 0``; exponents
giving a non-positive coefficient are rejected).  The procedure uses only the
standard library, so the same observations always give the same parameters.

Applying any of this to EPOR is a ``hypothesis``: a fit to a handful of
proxy-scale points says nothing reliable about a 1B or larger model, and the
report says so.  Warnings are explicit when there are no more points than
parameters (no residual degrees of freedom; fewer means no fit at all) and
when the points span less than one order of magnitude — which is the normal
situation for a local sweep.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from .report import CHINCHILLA_2022, KAPLAN_2020, ScalingReport, ScalingScope, seal_scaling_report

ALPHA_GRID = (0.005, 2.0, 0.005)
JOINT_GRID = (0.02, 1.2, 0.01)
REFINE_STEPS = 20


class Observation(BaseModel):
    """One (parameters, tokens, loss) observation; ``N`` is non-embedding parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: float
    tokens: float
    loss: float


def _grid(spec: tuple[float, float, float]) -> list[float]:
    low, high, step = spec
    count = round((high - low) / step)
    return [low + index * step for index in range(count + 1)]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; ``None`` when singular."""

    size = len(vector)
    rows = [[*matrix[i], vector[i]] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-300:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(size):
            if row != column:
                factor = rows[row][column] / rows[column][column]
                for k in range(column, size + 1):
                    rows[row][k] -= factor * rows[column][k]
    return [rows[i][size] / rows[i][i] for i in range(size)]


def _weighted_linear(
    columns: list[list[float]], losses: Sequence[float], weights: Sequence[float]
) -> tuple[list[float], float] | None:
    """Weighted least squares ``losses ~ sum_j c_j * columns[j]``; returns (coef, SSE)."""

    size = len(columns)
    gram = [
        [
            math.fsum(w * a * b for w, a, b in zip(weights, columns[i], columns[j], strict=True))
            for j in range(size)
        ]
        for i in range(size)
    ]
    rhs = [
        math.fsum(w * a * y for w, a, y in zip(weights, columns[i], losses, strict=True))
        for i in range(size)
    ]
    coefficients = _solve(gram, rhs)
    if coefficients is None:
        return None
    sse = math.fsum(
        w * (math.fsum(c * col[k] for c, col in zip(coefficients, columns, strict=True)) - y) ** 2
        for k, (w, y) in enumerate(zip(weights, losses, strict=True))
    )
    return coefficients, sse


def _fit_terms(
    terms: list[list[float]], losses: Sequence[float], weights: Sequence[float]
) -> tuple[float, list[float], float] | None:
    """Fit ``E + sum_j c_j * terms[j]`` with ``E >= 0`` and every ``c_j > 0``."""

    ones = [1.0] * len(losses)
    solved = _weighted_linear([ones, *terms], losses, weights)
    if solved is not None and solved[0][0] >= 0:
        coefficients, sse = solved
        energy, scales = coefficients[0], coefficients[1:]
    else:
        solved = _weighted_linear(terms, losses, weights)
        if solved is None:
            return None
        energy, (scales, sse) = 0.0, solved
    if any(scale <= 0 for scale in scales):
        return None
    return energy, scales, sse


def _span_warning(values: Sequence[float], name: str) -> list[str]:
    low, high = min(values), max(values)
    if high < 10 * low:
        return [
            f"{name} spans {high / low:.3g}x (less than one order of magnitude); the exponent "
            "is poorly constrained and extrapolation is unreliable."
        ]
    return []


def _dof_warning(points: int, parameters: int, name: str) -> list[str]:
    if points == parameters:
        return [
            f"{name}: {points} points for {parameters} parameters; zero residual degrees of "
            "freedom, so the fit interpolates exactly and its residuals carry no information."
        ]
    return []


def fit_power_law(xs: Sequence[float], losses: Sequence[float], *, variable: str) -> dict[str, Any]:
    """Fit ``L = E + A x^-alpha``; returns a JSON-ready result with warnings."""

    distinct = sorted(set(xs))
    result: dict[str, Any] = {
        "form": f"L({variable}) = E + A * {variable}^-alpha",
        "variable": variable,
        "points": len(xs),
        "distinct_values": len(distinct),
        "parameter_count": 3,
        "warnings": [],
    }
    if len(distinct) < 3 or any(x <= 0 for x in xs) or any(y <= 0 for y in losses):
        result.update(status="insufficient-points", parameters=None, residuals=None)
        result["warnings"].append(
            f"{variable}: {len(distinct)} distinct positive values; at least 3 are needed for "
            "3 parameters, so no fit was made."
        )
        return result
    reference = math.exp(math.fsum(math.log(x) for x in xs) / len(xs))
    scaled = [x / reference for x in xs]
    weights = [1.0 / (y * y) for y in losses]

    def evaluate(alpha: float) -> tuple[float, float, float] | None:
        fitted = _fit_terms([[s**-alpha for s in scaled]], losses, weights)
        if fitted is None:
            return None
        energy, (scale,), sse = fitted
        return sse, energy, scale

    best: tuple[float, float, float, float] | None = None
    for alpha in _grid(ALPHA_GRID):
        value = evaluate(alpha)
        if value is not None and (best is None or value[0] < best[0]):
            best = (value[0], alpha, value[1], value[2])
    if best is None:
        result.update(status="no-valid-fit", parameters=None, residuals=None)
        result["warnings"].append(f"{variable}: no exponent gave positive coefficients.")
        return result
    step = ALPHA_GRID[2]
    center = best[1]
    for index in range(-REFINE_STEPS, REFINE_STEPS + 1):
        alpha = center + index * step / REFINE_STEPS
        if alpha <= 0:
            continue
        value = evaluate(alpha)
        if value is not None and value[0] < best[0]:
            best = (value[0], alpha, value[1], value[2])
    _, alpha, energy, scale = best
    amplitude = scale * reference**alpha
    predicted = [energy + amplitude * x**-alpha for x in xs]
    residuals = [y - p for y, p in zip(losses, predicted, strict=True)]
    result.update(
        status="fitted",
        parameters={"E": energy, "A": amplitude, "alpha": alpha},
        residuals=residuals,
        rmse=math.sqrt(math.fsum(r * r for r in residuals) / len(residuals)),
        range={"min": min(xs), "max": max(xs)},
    )
    result["warnings"] += _dof_warning(len(distinct), 3, variable)
    result["warnings"] += _span_warning(xs, variable)
    return result


def fit_joint(
    parameters: Sequence[float], tokens: Sequence[float], losses: Sequence[float]
) -> dict[str, Any]:
    """Fit ``L = E + A N^-alpha + B D^-beta`` on observations with varied N and D."""

    pairs = sorted(set(zip(parameters, tokens, strict=True)))
    result: dict[str, Any] = {
        "form": "L(N, D) = E + A * N^-alpha + B * D^-beta",
        "points": len(losses),
        "distinct_points": len(pairs),
        "parameter_count": 5,
        "warnings": [],
    }
    if len(pairs) < 5 or len(set(parameters)) < 2 or len(set(tokens)) < 2:
        result.update(status="insufficient-points", parameters=None, residuals=None)
        result["warnings"].append(
            f"joint fit: {len(pairs)} distinct (N, D) points over {len(set(parameters))} model "
            f"sizes and {len(set(tokens))} token budgets; at least 5 points varying both axes "
            "are needed for 5 parameters, so no joint fit was made."
        )
        return result
    n_ref = math.exp(math.fsum(math.log(n) for n in parameters) / len(parameters))
    d_ref = math.exp(math.fsum(math.log(d) for d in tokens) / len(tokens))
    n_scaled = [n / n_ref for n in parameters]
    d_scaled = [d / d_ref for d in tokens]
    weights = [1.0 / (y * y) for y in losses]

    def evaluate(alpha: float, beta: float) -> tuple[float, float, float, float] | None:
        fitted = _fit_terms(
            [[n**-alpha for n in n_scaled], [d**-beta for d in d_scaled]], losses, weights
        )
        if fitted is None:
            return None
        energy, (a_scale, b_scale), sse = fitted
        return sse, energy, a_scale, b_scale

    best: tuple[float, float, float, float, float, float] | None = None
    grid = _grid(JOINT_GRID)
    for alpha in grid:
        for beta in grid:
            value = evaluate(alpha, beta)
            if value is not None and (best is None or value[0] < best[0]):
                best = (value[0], alpha, beta, value[1], value[2], value[3])
    if best is None:
        result.update(status="no-valid-fit", parameters=None, residuals=None)
        result["warnings"].append("joint fit: no exponents gave positive coefficients.")
        return result
    step = JOINT_GRID[2]
    center_alpha, center_beta = best[1], best[2]
    for i in range(-REFINE_STEPS // 2, REFINE_STEPS // 2 + 1):
        for j in range(-REFINE_STEPS // 2, REFINE_STEPS // 2 + 1):
            alpha = center_alpha + i * step / (REFINE_STEPS // 2)
            beta = center_beta + j * step / (REFINE_STEPS // 2)
            if alpha <= 0 or beta <= 0:
                continue
            value = evaluate(alpha, beta)
            if value is not None and value[0] < best[0]:
                best = (value[0], alpha, beta, value[1], value[2], value[3])
    _, alpha, beta, energy, a_scale, b_scale = best
    amplitude_a = a_scale * n_ref**alpha
    amplitude_b = b_scale * d_ref**beta
    predicted = [
        energy + amplitude_a * n**-alpha + amplitude_b * d**-beta
        for n, d in zip(parameters, tokens, strict=True)
    ]
    residuals = [y - p for y, p in zip(losses, predicted, strict=True)]
    result.update(
        status="fitted",
        parameters={"E": energy, "A": amplitude_a, "alpha": alpha, "B": amplitude_b, "beta": beta},
        residuals=residuals,
        rmse=math.sqrt(math.fsum(r * r for r in residuals) / len(residuals)),
        range={
            "parameters_min": min(parameters),
            "parameters_max": max(parameters),
            "tokens_min": min(tokens),
            "tokens_max": max(tokens),
        },
    )
    result["warnings"] += _dof_warning(len(pairs), 5, "joint fit")
    result["warnings"] += _span_warning(parameters, "N")
    result["warnings"] += _span_warning(tokens, "D")
    return result


def predict_loss(
    fit_results: dict[str, Any], parameters: float, tokens: float | None
) -> dict[str, Any]:
    """Extrapolate a fitted loss to ``(N, D)``: joint form if fitted, else L(N)."""

    joint = fit_results.get("joint") or {}
    single = fit_results.get("loss_vs_parameters") or {}
    if joint.get("status") == "fitted" and tokens is not None:
        p = joint["parameters"]
        value = p["E"] + p["A"] * parameters ** -p["alpha"] + p["B"] * tokens ** -p["beta"]
        fitted_max = max(joint["range"]["parameters_max"], 1.0)
        form, rmse = joint["form"], joint["rmse"]
    elif single.get("status") == "fitted":
        p = single["parameters"]
        value = p["E"] + p["A"] * parameters ** -p["alpha"]
        fitted_max = max(single["range"]["max"], 1.0)
        form, rmse = single["form"], single["rmse"]
    else:
        return {"value": None, "label": "unknown", "basis": "no fitted scaling law available"}
    return {
        "value": value,
        "label": "hypothesis",
        "basis": form,
        "extrapolation_factor": parameters / fitted_max,
        "fit_rmse": rmse,
        "caveat": (
            "Extrapolated from proxy-scale fits far outside their fitted range; no confidence "
            "interval is claimed and the value is not a predicted result for any EPOR model."
        ),
    }


def fit_observations(
    observations: Sequence[Observation],
    *,
    source: dict[str, Any],
    scope: ScalingScope = "proxy-scale",
) -> ScalingReport:
    """Fit L(N) (largest D), L(D) (largest N), and the joint form; seal the report."""

    if not observations:
        raise ValueError("a scaling fit needs observations")
    grouped: dict[tuple[float, float], list[float]] = {}
    for item in observations:
        grouped.setdefault((item.parameters, item.tokens), []).append(item.loss)
    aggregated = sorted((n, d, math.fsum(v) / len(v), len(v)) for (n, d), v in grouped.items())
    largest_d = max(d for _, d, _, _ in aggregated)
    largest_n = max(n for n, _, _, _ in aggregated)
    by_n = [(n, loss) for n, d, loss, _ in aggregated if d == largest_d]
    by_d = [(d, loss) for n, d, loss, _ in aggregated if n == largest_n]
    loss_vs_n = fit_power_law([n for n, _ in by_n], [y for _, y in by_n], variable="N")
    loss_vs_n["held_fixed"] = {"tokens": largest_d}
    loss_vs_d = fit_power_law([d for d, _ in by_d], [y for _, y in by_d], variable="D")
    loss_vs_d["held_fixed"] = {"parameters": largest_n}
    joint = fit_joint(
        [n for n, _, _, _ in aggregated],
        [d for _, d, _, _ in aggregated],
        [y for _, _, y, _ in aggregated],
    )
    warnings = [
        "Applying these scaling-law forms to EPOR is a hypothesis; proxy-scale fits do not "
        "establish release-scale behaviour."
    ]
    for part in (loss_vs_n, loss_vs_d, joint):
        warnings += part["warnings"]
    return seal_scaling_report(
        kind="scaling-fit",
        scope=scope,
        inputs={
            "source": source,
            "observations": [item.model_dump(mode="json") for item in observations],
            "aggregation": "mean loss over seeds per (N, D)",
            "parameter_definition": "N = non-embedding parameters",
            "procedure": (
                "grid + refined exponents, closed-form weighted least squares (weights 1/L^2) "
                "for E and coefficients; E >= 0, coefficients > 0"
            ),
            "procedure_label": "EPOR-adaptation",
        },
        results={
            "aggregated": [
                {"parameters": n, "tokens": d, "loss": y, "seeds": count}
                for n, d, y, count in aggregated
            ],
            "loss_vs_parameters": loss_vs_n,
            "loss_vs_tokens": loss_vs_d,
            "joint": joint,
            "formula_label": "disclosed",
            "application_label": "hypothesis",
        },
        warnings=warnings,
        citations=[KAPLAN_2020, CHINCHILLA_2022],
    )


def observations_from_sweep(report: ScalingReport) -> tuple[list[Observation], str]:
    """Completed sweep points as observations, and which loss was used."""

    if report.kind != "scaling-sweep":
        raise ValueError("not a scaling-sweep report")
    completed = [p for p in report.results["points"] if p["status"] == "completed"]
    if not completed:
        raise ValueError("the sweep has no completed points to fit")
    use_validation = all(p.get("validation_main_loss") is not None for p in completed)
    key = "validation_main_loss" if use_validation else "final_main_loss"
    return [
        Observation(
            parameters=float(p["parameters_non_embedding"]),
            tokens=float(p["tokens_seen"]),
            loss=float(p[key]),
        )
        for p in completed
    ], key


def fit_sweep(report: ScalingReport) -> ScalingReport:
    observations, key = observations_from_sweep(report)
    return fit_observations(
        observations,
        source={"kind": "scaling-sweep", "report_id": report.report_id, "loss": key},
        scope=report.scope,
    )


__all__ = [
    "Observation",
    "fit_joint",
    "fit_observations",
    "fit_power_law",
    "fit_sweep",
    "observations_from_sweep",
    "predict_loss",
]
