"""Tests for Topic 5: Ruin Theory."""

import numpy as np
import pytest

from therisktheory.ruin_theory.ruin_models import (
    adjustment_coefficient_exponential_severity,
    cramer_lundberg_ruin_approximation,
    lundberg_ruin_bound,
)


def test_analytical_adjustment_coefficient():
    result = adjustment_coefficient_exponential_severity(0.2, 50.0)
    assert result == pytest.approx(0.2 / 60.0)


def test_analytical_lundberg_bound():
    assert lundberg_ruin_bound(100.0, 0.01) == pytest.approx(np.exp(-1.0))


def test_vectorized_adjustment_coefficient():
    result = adjustment_coefficient_exponential_severity(
        np.array([0.1, 0.2, 0.3]), 50.0
    )
    expected = np.array([0.1 / 55.0, 0.2 / 60.0, 0.3 / 65.0])
    np.testing.assert_allclose(result, expected)


def test_vectorized_lundberg_bound():
    surplus = np.array([0.0, 50.0, 100.0, 200.0])
    result = lundberg_ruin_bound(surplus, 0.02)
    np.testing.assert_allclose(result, np.exp(-0.02 * surplus))


def test_zero_surplus_lundberg_bound():
    assert lundberg_ruin_bound(0.0, 0.05) == pytest.approx(1.0)


def test_large_surplus_lundberg_bound():
    assert lundberg_ruin_bound(10000.0, 0.05) < 1e-10


def test_lundberg_bound_decreases_with_surplus():
    values = lundberg_ruin_bound(np.array([10.0, 50.0, 100.0]), 0.01)
    assert values[0] > values[1] > values[2]


def test_cramer_lundberg_exponential_formula():
    theta = 0.2
    beta = 50.0
    surplus = 100.0
    coefficient = theta / (beta * (1.0 + theta))
    expected = np.exp(-coefficient * surplus) / (1.0 + theta)

    result = cramer_lundberg_ruin_approximation(surplus, theta, beta)
    assert result == pytest.approx(expected)


def test_poisson_intensity_cancels_from_exact_formula():
    first = cramer_lundberg_ruin_approximation(100.0, 0.2, 50.0, 1.0)
    second = cramer_lundberg_ruin_approximation(100.0, 0.2, 50.0, 10.0)
    assert first == pytest.approx(second)


def test_cramer_lundberg_vectorization():
    result = cramer_lundberg_ruin_approximation(
        np.array([0.0, 50.0, 100.0]), 0.2, 50.0
    )
    assert isinstance(result, np.ndarray)
    assert np.all((result >= 0.0) & (result <= 1.0))


@pytest.mark.parametrize(
    "theta,beta",
    [(-0.1, 50.0), (0.0, 50.0), (0.2, 0.0), (0.2, -1.0)],
)
def test_invalid_adjustment_coefficient_parameters(theta, beta):
    with pytest.raises(ValueError):
        adjustment_coefficient_exponential_severity(theta, beta)


def test_negative_surplus_is_rejected():
    with pytest.raises(ValueError):
        lundberg_ruin_bound(-10.0, 0.01)


def test_nonpositive_adjustment_coefficient_is_rejected():
    with pytest.raises(ValueError):
        lundberg_ruin_bound(10.0, 0.0)


def test_invalid_cramer_lundberg_parameters_are_rejected():
    with pytest.raises(ValueError):
        cramer_lundberg_ruin_approximation(-1.0, 0.2, 50.0)

    with pytest.raises(ValueError):
        cramer_lundberg_ruin_approximation(10.0, 0.2, 50.0, 0.0)


def test_non_finite_inputs_are_rejected():
    with pytest.raises(ValueError):
        adjustment_coefficient_exponential_severity(np.inf, 50.0)
