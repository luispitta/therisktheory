"""Unit and mathematical validation tests for collective_risk.safety_equation."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.collective_risk import (
    adjustment_coefficient,
    full_retention_adjustment,
    lundberg_equation,
)

try:
    from therisktheory.severity_aggregate import lognormal_severity_moments
except ImportError:
    lognormal_severity_moments = None


def test_full_retention_adjustment_lognormal() -> None:
    """Verify full-retention approximation with Lognormal moments."""
    mu, sigma, safety_loading = 1.0, 0.5, 0.2
    if lognormal_severity_moments is not None:
        moments = lognormal_severity_moments(mu=mu, sigma=sigma)
        mean_x = float(moments["mean"].item())
        second_moment_x = float(moments["second_raw_moment"].item())
    else:
        mean_x = float(np.exp(mu + 0.5 * sigma**2))
        second_moment_x = float(np.exp(2.0 * mu + 2.0 * sigma**2))

    approx_r = full_retention_adjustment((mean_x, second_moment_x), safety_loading)
    expected = (2.0 * safety_loading * mean_x) / second_moment_x
    assert np.isclose(approx_r, expected, rtol=1e-12)
    assert approx_r > 0.0


def test_adjustment_coefficient_exponential_analytical() -> None:
    """Verify exact adjustment coefficient for Exponential(mu) severity."""
    mu = 2.0
    mean_x = 1.0 / mu
    safety_loading = 0.2

    def mgf_exp(r: float) -> float:
        if r >= mu:
            return float("inf")
        return mu / (mu - r)

    expected_r = (safety_loading * mu) / (1.0 + safety_loading)
    computed_r = adjustment_coefficient(
        mgf_exp,
        mean_x,
        safety_loading,
        r_min=1e-9,
        r_max=mu - 1e-3,
    )
    assert np.isclose(computed_r, expected_r, rtol=1e-9)


def test_adjustment_coefficient_small_root() -> None:
    """Verify root finding works when R < 1e-6 (large mean severity)."""
    mu = 1e-5
    mean_x = 1.0 / mu
    safety_loading = 0.05

    def mgf_exp(r: float) -> float:
        if r >= mu:
            return float("inf")
        return mu / (mu - r)

    expected_r = (safety_loading * mu) / (1.0 + safety_loading)
    computed_r = adjustment_coefficient(
        mgf_exp,
        mean_x,
        safety_loading,
        r_min=1e-9,
        r_max=0.5 * mu,
    )
    assert np.isclose(computed_r, expected_r, rtol=1e-8)


def test_lundberg_equation_at_root() -> None:
    """Verify Lundberg equation residual equals zero at computed root."""
    mu = 1.0
    mean_x = 1.0
    safety_loading = 0.3

    def mgf_exp(r: float) -> float:
        if r >= mu:
            return float("inf")
        return mu / (mu - r)

    r_sol = adjustment_coefficient(
        mgf_exp,
        mean_x,
        safety_loading,
        r_min=1e-9,
        r_max=0.99,
    )
    residual = lundberg_equation(r_sol, mgf_exp, mean_x, safety_loading)
    assert np.isclose(residual, 0.0, atol=1e-10)


def test_rejects_r_max_outside_mgf_domain() -> None:
    """Verify explicit error when r_max exceeds the MGF domain of existence."""
    mu = 1.0

    def mgf_exp(r: float) -> float:
        if r >= mu:
            return float("inf")
        return mu / (mu - r)

    with pytest.raises(ValueError, match="existence domain"):
        adjustment_coefficient(mgf_exp, 1.0, 0.2, r_min=1e-6, r_max=mu)


def test_invalid_inputs() -> None:
    """Verify parameter and moment validations."""
    with pytest.raises(ValueError):
        full_retention_adjustment((10.0, 50.0), 0.2)
    with pytest.raises(ValueError):
        full_retention_adjustment((10.0, 200.0), 0.0)
    with pytest.raises(ValueError):
        adjustment_coefficient(
            lambda r: float(np.exp(r)), 1.0, 0.2, r_min=1.0, r_max=0.5
        )
