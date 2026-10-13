"""Deterministic scaling-law fits: recovery, warnings, and sealed fit reports."""

from __future__ import annotations

import pytest

from epor.scaling.fit import (
    Observation,
    fit_joint,
    fit_observations,
    fit_power_law,
    predict_loss,
)

# Off-grid exponents so the refinement step, not the coarse grid, recovers them.
E, A, ALPHA, B, BETA = 1.69, 406.4, 0.3413, 410.7, 0.2817
SIZES = [1e7, 3e7, 1e8, 3e8, 1e9]
TOKENS = [1e9, 3e9, 1e10, 3e10, 1e11]


def _loss(n: float, d: float) -> float:
    return E + A * n**-ALPHA + B * d**-BETA


def test_power_law_recovers_known_parameters() -> None:
    losses = [E + A * n**-ALPHA for n in SIZES]
    result = fit_power_law(SIZES, losses, variable="N")
    assert result["status"] == "fitted"
    fitted = result["parameters"]
    assert fitted["alpha"] == pytest.approx(ALPHA, abs=2e-3)
    assert fitted["E"] == pytest.approx(E, rel=1e-2)
    assert fitted["A"] == pytest.approx(A, rel=0.05)
    assert result["rmse"] < 1e-3
    assert result["warnings"] == []
    assert fit_power_law(SIZES, losses, variable="N") == result


def test_joint_fit_recovers_chinchilla_form() -> None:
    sizes, tokens, losses = [], [], []
    for n in SIZES:
        for d in TOKENS:
            sizes.append(n)
            tokens.append(d)
            losses.append(_loss(n, d))
    result = fit_joint(sizes, tokens, losses)
    assert result["status"] == "fitted"
    fitted = result["parameters"]
    assert fitted["alpha"] == pytest.approx(ALPHA, abs=5e-3)
    assert fitted["beta"] == pytest.approx(BETA, abs=5e-3)
    assert fitted["E"] == pytest.approx(E, rel=0.02)
    assert result["rmse"] < 2e-3


def test_fits_warn_on_too_few_points_and_small_spans() -> None:
    exact = fit_power_law([1e6, 2e6, 4e6], [3.0, 2.8, 2.7], variable="N")
    assert exact["status"] == "fitted"
    assert any("zero residual degrees of freedom" in w for w in exact["warnings"])
    assert any("less than one order of magnitude" in w for w in exact["warnings"])

    too_few = fit_power_law([1e6, 2e6], [3.0, 2.8], variable="N")
    assert too_few["status"] == "insufficient-points"
    assert too_few["parameters"] is None
    assert "no fit was made" in too_few["warnings"][0]

    joint = fit_joint([1e6, 2e6, 1e6], [1e3, 1e3, 2e3], [3.0, 2.9, 2.8])
    assert joint["status"] == "insufficient-points"


def test_fit_report_is_sealed_deterministic_and_labelled() -> None:
    observations = [
        Observation(parameters=n, tokens=d, loss=_loss(n, d)) for n in SIZES[:2] for d in TOKENS[:2]
    ]
    report = fit_observations(observations, source={"kind": "synthetic"}, scope="harness-fixture")
    again = fit_observations(observations, source={"kind": "synthetic"}, scope="harness-fixture")
    assert report.report_id == again.report_id
    assert report.kind == "scaling-fit"
    assert report.results["formula_label"] == "disclosed"
    assert report.results["application_label"] == "hypothesis"
    assert {c.key for c in report.citations} == {
        "kaplan-scaling-laws",
        "chinchilla-compute-optimal",
    }
    # Two sizes and two budgets: nothing is fittable, and the report says so.
    assert report.results["loss_vs_parameters"]["status"] == "insufficient-points"
    assert report.results["joint"]["status"] == "insufficient-points"
    assert any("hypothesis" in w for w in report.warnings)
    assert predict_loss(report.results, 1e9, 2e10)["value"] is None


def test_prediction_carries_an_extrapolation_caveat() -> None:
    observations = [Observation(parameters=n, tokens=1e10, loss=_loss(n, 1e10)) for n in SIZES]
    report = fit_observations(observations, source={"kind": "synthetic"})
    prediction = predict_loss(report.results, 1e10, None)
    assert prediction["label"] == "hypothesis"
    assert prediction["extrapolation_factor"] == pytest.approx(10.0)
    assert prediction["value"] == pytest.approx(_loss(1e10, 1e10), rel=1e-2)
    assert "not a predicted result" in prediction["caveat"]
