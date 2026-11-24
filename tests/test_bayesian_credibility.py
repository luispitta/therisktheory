"""Unit tests and mathematical validation tests for Bayesian credibility module."""
import pytest
import numpy as np
from therisktheory.credibility import (
    bayesian_credibility_premium,
    posterior_predictive_mean,
)

def test_bayesian_credibility_premium_scalar() -> None:
    """Test scalar input calculation against hand-calculated analytical value."""
    alpha = 2.0
    beta = 1.0
    claims = 5.0
    exposure = 2.0
    # Posterior: alpha_post = 2 + 5 = 7, beta_post = 1 + 2 = 3
    # Expected premium = 7 / 3
    result = bayesian_credibility_premium(alpha, beta, claims, exposure)
    expected = 7.0 / 3.0
    assert np.isclose(result, expected)

def test_bayesian_credibility_premium_vectorized() -> None:
    """Test vectorization support with numpy arrays."""
    alpha = 2.0
    beta = 1.0
    claims = np.array([2.0, 5.0, 10.0])
    exposures = np.array([1.0, 2.0, 4.0])
    expected = np.array([4.0 / 2.0, 7.0 / 3.0, 12.0 / 5.0])
    result = bayesian_credibility_premium(alpha, beta, claims, exposures)
    np.testing.assert_allclose(result, expected)

def test_posterior_predictive_mean() -> None:
    """Test consistency of posterior predictive mean with credibility premium."""
    alpha = 3.0
    beta = 1.5
    claims = 4.0
    exposure = 2.0
    cred_prem = bayesian_credibility_premium(alpha, beta, claims, exposure)
    pred_mean = posterior_predictive_mean(alpha, beta, claims, exposure)
    assert np.isclose(cred_prem, pred_mean)

def test_invalid_parameters_raises_error() -> None:
    """Verify that non-positive parameters, negative claims, or non-finite inputs raise ValueError."""
    with pytest.raises(ValueError):
        bayesian_credibility_premium(0.0, 1.0, 5.0, 2.0)
    with pytest.raises(ValueError):
        bayesian_credibility_premium(2.0, -1.0, 5.0, 2.0)
    with pytest.raises(ValueError):
        bayesian_credibility_premium(2.0, 1.0, 5.0, 0.0)
    with pytest.raises(ValueError):
        bayesian_credibility_premium(2.0, 1.0, -1.0, 2.0)  # Negative claims test
    with pytest.raises(ValueError):
        bayesian_credibility_premium(2.0, 1.0, float("nan"), 2.0)  # Non-finite test
