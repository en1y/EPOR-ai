"""Calibration of multiple-choice predictions: ECE, Brier score, reliability bins.

Definitions (fixed so numbers compare across reports):

* **Confidence** of an item is the largest normalized choice probability; the
  prediction is that choice (ties resolve to the lowest index).
* **Expected calibration error** uses ``bins`` equal-width bins over
  ``[0, 1]``; an item with confidence ``c`` lands in bin
  ``min(floor(c * bins), bins - 1)``.  ``ECE = Σ_b (n_b / N) · |acc_b - conf_b|``.
* **Brier score** is the multi-class form, ``mean_i Σ_k (p_ik - y_ik)²``,
  in ``[0, 2]`` (lower is better).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CALIBRATION_BINS = 10


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float
    upper: float
    count: int = Field(ge=0)
    mean_confidence: float | None
    accuracy: float | None


class CalibrationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["max-probability-equal-width"] = "max-probability-equal-width"
    probability_source: str
    bins: int = Field(ge=1)
    items: int = Field(ge=1)
    expected_calibration_error: float
    brier_score: float
    reliability: list[ReliabilityBin]


def softmax(scores: Sequence[float]) -> list[float]:
    """Numerically stable softmax of finite log-scores."""

    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def calibration_summary(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
    probability_source: str,
) -> CalibrationSummary:
    """ECE, Brier score, and reliability bins for per-item choice distributions."""

    if bins < 1:
        raise ValueError("calibration needs at least one bin")
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("calibration needs one label per non-empty probability row")
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    correct_sums = [0] * bins
    brier_terms: list[float] = []
    for row, label in zip(probabilities, labels, strict=True):
        if not 0 <= label < len(row):
            raise ValueError("label is outside the choice range")
        if abs(math.fsum(row) - 1.0) > 1e-6 or any(p < 0 for p in row):
            raise ValueError("each probability row must be a distribution")
        confidence = max(row)
        predicted = row.index(confidence)
        index = min(int(confidence * bins), bins - 1)
        counts[index] += 1
        confidence_sums[index] += confidence
        correct_sums[index] += int(predicted == label)
        brier_terms.append(
            math.fsum((p - (1.0 if k == label else 0.0)) ** 2 for k, p in enumerate(row))
        )
    total = len(labels)
    reliability: list[ReliabilityBin] = []
    ece_terms: list[float] = []
    for index in range(bins):
        count = counts[index]
        mean_confidence = confidence_sums[index] / count if count else None
        accuracy = correct_sums[index] / count if count else None
        if mean_confidence is not None and accuracy is not None:
            ece_terms.append(count / total * abs(accuracy - mean_confidence))
        reliability.append(
            ReliabilityBin(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )
    return CalibrationSummary(
        probability_source=probability_source,
        bins=bins,
        items=total,
        expected_calibration_error=math.fsum(ece_terms),
        brier_score=math.fsum(brier_terms) / total,
        reliability=reliability,
    )


__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "CalibrationSummary",
    "ReliabilityBin",
    "calibration_summary",
    "softmax",
]
