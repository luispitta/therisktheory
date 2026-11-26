"""Tests and boundary validation for full credibility standards."""

import numpy as np
import pytest
from scipy.stats import norm

from therisktheory.credibility.full_credibility import (
    full_credibility_required_exposure,
    full_credibility_standard,
)

# Optional integration check with Monticelli's partial credibility module
partial_module = pytest.importorskip(
    "therisktheory.credibility.partial_credibility",
    reason="Optional test: requires Monticelli partial credibility PR #21",
)


def test_independent_norm_ppf_verification() -> None:
    """Independent calculation: compare standard against scipy.stats.norm.ppf."""
    p = 0.95
    k = 0.05
    # Two-sided confidence p means upper tail is alpha/2 = (1-p)/2
    z_scipy = norm.ppf((1.0 + p) / 2.0)
    expected_n0 = (z_scipy / k) ** 2

    n_calculated = full_credibility_standard(
        target_accuracy=k, confidence_level=p
    )
    np.testing.assert_allclose(n_calculated.item(), expected_n0, rtol=1e-12)


def test_classical_1082_claims_standard() -> None:
    """Verify classical actuarial 1,082 claims benchmark (p=0.90, k=0.05)."""
    n_0 = full_credibility_standard(target_accuracy=0.05, confidence_level=0.90)
    assert np.isclose(n_0.item(), 1082.22, atol=0.5)


def test_compatibility_with_monticelli_partial_credibility() -> None:
    """Demonstrate compatibility of assumptions and results with Monticelli (PR #21).

    1. Both models yield the exact same threshold standard n_0.
    2. When observed claims meet or exceed n_0, credibility reaches full weight (Z = 1.0).
    """
    k = 0.05
    p = 0.95
    n_full = full_credibility_standard(target_accuracy=k, confidence_level=p)

    # 1. Compare standard with Monticelli's formula
    n_monticelli = partial_module.partial_credibility_standard(
        expected_claim_count=100.0, target_accuracy=k, confidence_level=p
    )
    np.testing.assert_allclose(n_full.item(), float(n_monticelli), rtol=1e-12)

    # 2. Check full credibility transition (Z = 1.0 for n >= n_0)
    n_observed = n_full.item() + 100.0
    z_weight = min(1.0, np.sqrt(n_observed / n_full.item()))
    assert z_weight == 1.0


def test_severity_variability_increases_standard() -> None:
    """Property test: positive severity variance strictly increases required standard."""
    n_pure = full_credibility_standard(target_accuracy=0.05)
    n_compound = full_credibility_standard(
        target_accuracy=0.05,
        severity_mean=100.0,
        severity_variance=10000.0,  # CV^2 = 1.0 -> standard doubles
    )
    assert np.isclose(n_compound.item(), 2.0 * n_pure.item())


def test_required_exposure_scaling() -> None:
    """Verify exposure units scale inversely with claim frequency rate."""
    lam = 0.10
    n_std = full_credibility_standard(
        target_accuracy=0.05, confidence_level=0.95
    )
    exp_req = full_credibility_required_exposure(
        claim_frequency_rate=lam,
        target_accuracy=0.05,
        confidence_level=0.95,
    )
    assert np.isclose(exp_req.item(), n_std.item() / lam)


def test_invalid_parameters_raise() -> None:
    """Boundary and domain tests for out-of-range parameters."""
    with pytest.raises(ValueError, match="confidence_level"):
        full_credibility_standard(confidence_level=1.0)
    with pytest.raises(ValueError, match="target_accuracy"):
        full_credibility_standard(target_accuracy=-0.05)
    with pytest.raises(ValueError, match="claim_frequency_rate"):
        full_credibility_required_exposure(claim_frequency_rate=0.0)
