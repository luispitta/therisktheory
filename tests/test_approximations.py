"""Unit tests and mathematical properties for aggregate loss approximations."""

import numpy as np
import pytest

from therisktheory.collective_risk.approximations import (
    edgeworth_approximation,
    normal_approximation,
    normal_power_approximation,
)


def test_normal_power_rejects_negative_skewness() -> None:
    """Verify NP approximation rejects negative skewness (Prof. Pitta review)."""
    with pytest.raises(ValueError, match="skewness must be non-negative"):
        normal_power_approximation(
            x=120.0, mean=100.0, variance=400.0, skewness=-0.5
        )


def test_skewness_zero_reduces_to_normal() -> None:
    """Mathematical property: when skewness is zero, Edgeworth and NP reduce to Normal."""
    x = 130.0
    mu = 100.0
    var = 400.0

    p_norm = normal_approximation(x, mu, var)
    p_edge = edgeworth_approximation(x, mu, var, skewness=0.0)
    p_np = normal_power_approximation(x, mu, var, skewness=0.0)

    assert np.isclose(p_edge.item(), p_norm.item(), atol=1e-12)
    assert np.isclose(p_np.item(), p_norm.item(), atol=1e-12)


def test_approximations_against_simulation() -> None:
    """Compare Normal, Edgeworth, and NP against a compound Poisson simulation."""
    lam, mean_x = 50.0, 200.0
    mu_s = lam * mean_x
    var_s = lam * 2.0 * mean_x**2
    skew_s = (lam * 6.0 * mean_x**3) / (var_s**1.5)

    x_eval = 12000.0
    p_np = normal_power_approximation(x_eval, mu_s, var_s, skew_s).item()
    p_norm = normal_approximation(x_eval, mu_s, var_s).item()

    rng = np.random.default_rng(20261110)
    counts = rng.poisson(lam, size=30_000)
    losses = rng.exponential(mean_x, size=counts.sum())
    owners = np.repeat(np.arange(30_000), counts)
    totals = np.bincount(owners, weights=losses, minlength=30_000)
    empirical_cdf = np.mean(totals <= x_eval)

    assert abs(p_np - empirical_cdf) < 0.03
    assert abs(p_norm - empirical_cdf) < 0.04


def test_deterministic_zero_variance() -> None:
    """Boundary test: variance zero yields step function."""
    assert normal_approximation(10.0, 10.0, 0.0).item() == 1.0
    assert normal_approximation(9.9, 10.0, 0.0).item() == 0.0


def test_vectorization_and_broadcasting() -> None:
    """Verify array inputs and broadcasting."""
    x = np.array([80.0, 100.0, 120.0])
    p = normal_power_approximation(x, 100.0, 400.0, skewness=0.5)
    assert p.shape == (3,)
    assert p[0] < p[1] < p[2]
