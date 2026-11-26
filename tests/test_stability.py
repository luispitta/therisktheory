"""Unit tests and mathematical properties for collective stability and loading."""

import numpy as np
import pytest

from therisktheory.collective_risk.stability import (
    collective_safety_loading,
    loading_from_initial_reserve,
    stability_parameter,
    stability_sensitivity,
)


def test_stability_parameter_analytical() -> None:
    """Verify analytical example: lambda=100, mu1=10, mu2=200, theta=0.15."""
    k = stability_parameter(
        claim_rate=100.0,
        severity_mean=10.0,
        severity_second_moment=200.0,
        safety_loading=0.15,
    )
    expected = 0.15 * 10.0 * (10.0 / np.sqrt(200.0))
    np.testing.assert_allclose(k.item(), expected, rtol=1e-12)


def test_stability_sensitivity_finite_differences() -> None:
    """Verify sensitivities against finite difference numerical derivatives."""
    lam = 80.0
    mu1 = 15.0
    mu2 = 500.0
    theta = 0.20
    h = 1e-6

    sens = stability_sensitivity(lam, mu1, mu2, theta)

    # Numerical derivative w.r.t theta
    k_plus = stability_parameter(lam, mu1, mu2, theta + h).item()
    k_minus = stability_parameter(lam, mu1, mu2, theta - h).item()
    num_d_theta = (k_plus - k_minus) / (2.0 * h)
    np.testing.assert_allclose(sens.d_theta.item(), num_d_theta, rtol=1e-5)

    # Numerical derivative w.r.t lambda
    k_plus_l = stability_parameter(lam + h, mu1, mu2, theta).item()
    k_minus_l = stability_parameter(lam - h, mu1, mu2, theta).item()
    num_d_lambda = (k_plus_l - k_minus_l) / (2.0 * h)
    np.testing.assert_allclose(sens.d_lambda.item(), num_d_lambda, rtol=1e-5)

    # Unit elasticities
    assert np.isclose(sens.elasticity_theta.item(), 1.0)
    assert np.isclose(sens.elasticity_lambda.item(), 0.5)


def test_collective_safety_loading_roundtrip() -> None:
    """Mathematical identity: collective_safety_loading inverts stability."""
    target_k = 1.75
    lam = 60.0
    mu1 = 20.0
    mu2 = 600.0

    theta_req = collective_safety_loading(target_k, lam, mu1, mu2)
    k_achieved = stability_parameter(lam, mu1, mu2, theta_req)
    np.testing.assert_allclose(k_achieved.item(), target_k, rtol=1e-12)


def test_loading_from_initial_reserve_asymptotics() -> None:
    """Verify asymptotic property: theta -> 0 as initial reserve u -> infinity."""
    eps = 0.05
    mu1 = 100.0
    mu2 = 25000.0

    u_vals = np.array([500.0, 2000.0, 10000.0, 50000.0])
    thetas = loading_from_initial_reserve(u_vals, eps, mu1, mu2)

    # Must decrease monotonically
    assert np.all(np.diff(thetas) < 0.0)
    # Inverse proportionality: doubling u halves theta
    np.testing.assert_allclose(thetas[1] / thetas[0], 500.0 / 2000.0)


def test_domain_restrictions_raise() -> None:
    """Test boundary and invalid parameter domain restrictions."""
    with pytest.raises(ValueError, match="strictly positive"):
        stability_parameter(
            claim_rate=0.0, severity_mean=10.0,
            severity_second_moment=200.0, safety_loading=0.1
        )
    with pytest.raises(ValueError, match="strictly positive"):
        stability_sensitivity(
            claim_rate=10.0, severity_mean=10.0,
            severity_second_moment=200.0, safety_loading=0.0
        )
    with pytest.raises(ValueError, match="target_ruin_probability"):
        loading_from_initial_reserve(
            initial_reserve=100.0, target_ruin_probability=1.5,
            severity_mean=10.0, severity_second_moment=200.0
        )


def test_interoperability_with_collective_process() -> None:
    """Integration test connecting with Bladimir's process (PR #19).

    Uses pytest.importorskip INSIDE the test function so standalone tests
    are never skipped on isolated branches.
    """
    process_module = pytest.importorskip(
        "therisktheory.collective_risk.process",
        reason="Requires Bladimir collective risk process PR #19",
    )
    mean_claim = 200.0
    moments = [mean_claim, 2 * mean_claim**2, 6 * mean_claim**3]
    proc = process_module.collective_risk_process(
        initial_capital=1000.0,
        claim_rate=3.0,
        severity_moments=moments,
        safety_loading=0.10,
    )
    k = stability_parameter(
        proc.claim_rate,
        proc.severity_moments[0],
        proc.severity_moments[1],
        proc.safety_loading,
    )
    assert k.item() > 0.0
