"""Hand-checked calibration numbers: ECE with equal-width bins and multi-class Brier."""

from __future__ import annotations

import math

import pytest

from epor.evaluation.calibration import calibration_summary, softmax

ROWS = [[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.25, 0.25, 0.5]]
LABELS = [0, 1, 0, 2]


def test_ece_and_brier_match_hand_computation() -> None:
    summary = calibration_summary(ROWS, LABELS, bins=10, probability_source="test")
    # Bins 9, 6, 7, 5 each hold one item: |1-.9| + |0-.6| + |1-.7| + |1-.5| over 4.
    assert summary.expected_calibration_error == pytest.approx(1.5 / 4)
    # (0.02 + 0.72 + 0.18 + 0.375) / 4
    assert summary.brier_score == pytest.approx(1.295 / 4)
    assert summary.bins == 10 and summary.items == 4
    assert [b.count for b in summary.reliability] == [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    six = summary.reliability[6]
    assert (six.lower, six.upper) == pytest.approx((0.6, 0.7))
    assert (six.mean_confidence, six.accuracy) == pytest.approx((0.6, 0.0))
    assert summary.reliability[0].mean_confidence is None


def test_bin_count_changes_ece_as_recorded() -> None:
    summary = calibration_summary(ROWS, LABELS, bins=2, probability_source="test")
    # Everything lands in [0.5, 1]: mean confidence 0.675, accuracy 0.75.
    assert summary.reliability[1].count == 4
    assert summary.expected_calibration_error == pytest.approx(0.075)
    # A confidence of exactly 1.0 belongs to the last bin.
    edge = calibration_summary([[1.0, 0.0]], [0], bins=4, probability_source="test")
    assert edge.reliability[3].count == 1
    assert edge.expected_calibration_error == 0.0 and edge.brier_score == 0.0


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        calibration_summary([[0.5, 0.6]], [0], probability_source="test")
    with pytest.raises(ValueError):
        calibration_summary([[0.5, 0.5]], [2], probability_source="test")
    with pytest.raises(ValueError):
        calibration_summary([], [], probability_source="test")


def test_softmax_is_stable() -> None:
    probabilities = softmax([-1000.0, -1000.0 + math.log(3.0)])
    assert probabilities == pytest.approx([0.25, 0.75])
