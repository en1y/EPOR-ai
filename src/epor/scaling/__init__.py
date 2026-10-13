"""EPOR scaling laboratory (v0.0.6): ladder, profiling, sweeps, mixtures, fits, projections.

* :mod:`~epor.scaling.ladder` — 10M/100M/300M/1B dense rungs; only rungs within
  the local kernel bound run locally, the rest are projection-only.
* :mod:`~epor.scaling.flops` — analytic parameter and FLOP counts.
* :mod:`~epor.scaling.profile` — measured host throughput, memory, and MFU.
* :mod:`~epor.scaling.sweep` — resumable rung x token-budget x seed sweeps.
* :mod:`~epor.scaling.mixture` — domain-mixture ablations with per-domain loss.
* :mod:`~epor.scaling.fit` — deterministic L(N), L(D), and joint fits.
* :mod:`~epor.scaling.projection` — compute, storage, energy, and cost projections
  from operator-supplied inputs.

Every runner returns a sealed :class:`~epor.scaling.report.ScalingReport`.
Nothing here provisions compute, calls a paid API, or fetches prices.  The
profile, sweep, and mixture runners import PyTorch lazily; the ladder, FLOPs,
fit, projection, and report modules do not need it.
"""

from __future__ import annotations

from .fit import fit_joint, fit_observations, fit_power_law, fit_sweep, predict_loss
from .flops import FlopsEstimate, ParameterCounts, estimate_flops, parameter_counts
from .ladder import LADDER_RUNGS, RungStatus, derive_rung, ladder_paths, load_ladder, rung_status
from .mixture import MixturePlan, NamedMixture, mixture_summary, run_mixture_ablation
from .profile import ProfileSettings, profile_training_step
from .projection import ProjectionInputs, load_projection_inputs, project_training
from .report import (
    ScalingReport,
    ScalingReportError,
    load_scaling_reports,
    seal_scaling_report,
    store_scaling_report,
)
from .sweep import SweepPlan, run_sweep, sweep_points

__all__ = [
    "LADDER_RUNGS",
    "FlopsEstimate",
    "MixturePlan",
    "NamedMixture",
    "ParameterCounts",
    "ProfileSettings",
    "ProjectionInputs",
    "RungStatus",
    "ScalingReport",
    "ScalingReportError",
    "SweepPlan",
    "derive_rung",
    "estimate_flops",
    "fit_joint",
    "fit_observations",
    "fit_power_law",
    "fit_sweep",
    "ladder_paths",
    "load_ladder",
    "load_projection_inputs",
    "load_scaling_reports",
    "mixture_summary",
    "parameter_counts",
    "predict_loss",
    "profile_training_step",
    "project_training",
    "run_mixture_ablation",
    "run_sweep",
    "rung_status",
    "seal_scaling_report",
    "store_scaling_report",
    "sweep_points",
]
