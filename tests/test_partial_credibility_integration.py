"""Joint integration tests for Topic 4 credibility models."""

import pytest

from therisktheory.credibility.partial_credibility import (
    integrated_partial_credibility_evaluation,
    partial_credibility_standard,
)
from therisktheory.credibility.premium import credibility_premium


def test_integrated_partial_credibility_uses_common_premium():
    result = integrated_partial_credibility_evaluation(
        exposure_count=100,
        observed_mean=200,
        collective_base=150,
    )

    standard = partial_credibility_standard(100)
    expected_weight = min(1.0, (100 / standard) ** 0.5)
    expected_premium = credibility_premium(200, 150, expected_weight)

    assert result["full_credibility_standard"] == pytest.approx(standard)
    assert result["credibility_weight_z"] == pytest.approx(expected_weight)
    assert result["adjusted_premium"] == pytest.approx(float(expected_premium))


def test_integrated_partial_credibility_returns_expected_example():
    result = integrated_partial_credibility_evaluation(100, 200, 150)

    assert result["credibility_weight_z"] == pytest.approx(0.255106728462327)
    assert result["adjusted_premium"] == pytest.approx(162.75533642311638)
