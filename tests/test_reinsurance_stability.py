"""Unit tests for reinsurance stability metrics, optimal reinsurance, and reciprocity."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.severity_aggregate import (
    insurer_reciprocity_metric,
    optimal_reinsurance,
    technical_result_variance,
)

# ============================================================================
# Tests for technical_result_variance
# ============================================================================


def test_technical_result_variance_scalar() -> None:
    # E[N] = 10, Var(N) = 10 (Poisson), E[X] = 100, Var(X) = 400
    # Gross Var(S) = 10 * 400 + 10 * 100^2 = 4000 + 100000 = 104000
    # alpha = 0.5 -> Var(S_ret) = 0.25 * 104000 = 26000
    var_ret = technical_result_variance(
        retention_rate=0.5,
        expected_frequency=10.0,
        variance_frequency=10.0,
        expected_severity=100.0,
        variance_severity=400.0,
    )
    assert np.asarray(var_ret).item() == pytest.approx(26000.0)


def test_technical_result_variance_boundaries() -> None:
    # alpha = 0.0 -> variance is 0.0
    var_0 = technical_result_variance(0.0, 5.0, 5.0, 50.0, 25.0)
    assert np.asarray(var_0).item() == 0.0

    # alpha = 1.0 -> full gross variance
    var_1 = technical_result_variance(1.0, 5.0, 5.0, 50.0, 25.0)
    gross_var = 5.0 * 25.0 + 5.0 * (50.0**2)
    assert np.asarray(var_1).item() == pytest.approx(gross_var)


def test_technical_result_variance_vectorized() -> None:
    alphas = np.array([0.2, 0.4, 0.6])
    res = technical_result_variance(alphas, 10.0, 10.0, 100.0, 400.0)
    expected = (alphas**2) * 104000.0
    np.testing.assert_allclose(res, expected)


# ============================================================================
# Tests for optimal_reinsurance
# ============================================================================


def test_optimal_reinsurance_target_variance() -> None:
    # Var(S) = 100000, V_target = 25000 -> alpha* = sqrt(25000 / 100000) = sqrt(0.25) = 0.5
    opt = optimal_reinsurance(
        expected_loss=1000.0,
        variance_loss=100000.0,
        reinsurer_loading=0.20,
        target_variance=25000.0,
        criterion="target_variance",
    )
    assert np.asarray(opt).item() == pytest.approx(0.5)

    # V_target >= Var(S) -> alpha* = 1.0 (no reinsurance needed)
    opt_full = optimal_reinsurance(
        expected_loss=1000.0,
        variance_loss=100000.0,
        reinsurer_loading=0.20,
        target_variance=200000.0,
        criterion="target_variance",
    )
    assert np.asarray(opt_full).item() == 1.0


def test_optimal_reinsurance_mean_variance() -> None:
    # alpha* = (lam * (1 + xi) * E[S]) / (2 * Var(S))
    # E[S] = 1000, Var(S) = 100000, xi = 0.20 -> (1 + xi) * E[S] = 1200
    # lam = 100 -> numerator = 120000
    # denominator = 200000 -> alpha* = 120000 / 200000 = 0.6
    opt = optimal_reinsurance(
        expected_loss=1000.0,
        variance_loss=100000.0,
        reinsurer_loading=0.20,
        criterion="mean_variance",
        risk_aversion=100.0,
    )
    assert np.asarray(opt).item() == pytest.approx(0.6)


def test_optimal_reinsurance_premium_budget() -> None:
    # Reinsurance cost full = (1 + xi) * E[S] = 1.25 * 1000 = 1250
    # Budget B = 500 -> (1 - alpha) <= 500 / 1250 = 0.40 -> alpha* = 1.0 - 0.40 = 0.60
    opt = optimal_reinsurance(
        expected_loss=1000.0,
        variance_loss=50000.0,
        reinsurer_loading=0.25,
        criterion="premium_budget",
        premium_budget=500.0,
    )
    assert np.asarray(opt).item() == pytest.approx(0.60)


# ============================================================================
# Tests for insurer_reciprocity_metric
# ============================================================================


def test_insurer_reciprocity_metric_identical_independent() -> None:
    # Two identical independent portfolios: Var1 = Var2 = 100k, rho = 0, q1 = 0.5
    # Post var 1: 0.25 * 100k + 0.25 * 100k = 50k (50% reduction)
    res = insurer_reciprocity_metric(
        variance_1=100000.0,
        variance_2=100000.0,
        quota_1=0.5,
        correlation=0.0,
    )
    assert np.asarray(res["post_variance_1"]).item() == pytest.approx(50000.0)
    assert np.asarray(res["post_variance_2"]).item() == pytest.approx(50000.0)
    assert np.asarray(res["variance_reduction_1"]).item() == pytest.approx(0.5)
    assert np.asarray(res["total_reciprocity_gain"]).item() == pytest.approx(0.5)


def test_insurer_reciprocity_metric_financial_balance() -> None:
    # Portfolio 1: E1 = 1000, Var1 = 40000
    # Portfolio 2: E2 = 2000, Var2 = 160000
    # q1 = 0.4 -> q2 = 0.4 * (1000 / 2000) = 0.2
    res = insurer_reciprocity_metric(
        variance_1=40000.0,
        variance_2=160000.0,
        expected_loss_1=1000.0,
        expected_loss_2=2000.0,
        quota_1=0.4,
        correlation=0.1,
    )
    assert np.asarray(res["quota_2"]).item() == pytest.approx(0.2)
    assert float(res["total_reciprocity_gain"]) > 0.0


# ============================================================================
# Domain, non-finite, and error handling tests
# ============================================================================


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_stability_non_finite_rejection(bad_val: float) -> None:
    """Verify that stability functions reject non-finite inputs (NaN and Inf)."""
    # 1. technical_result_variance
    with pytest.raises(ValueError, match="finite"):
        technical_result_variance(bad_val, 10.0, 10.0, 100.0, 400.0)
    with pytest.raises(ValueError, match="finite"):
        technical_result_variance(0.5, bad_val, 10.0, 100.0, 400.0)
    with pytest.raises(ValueError, match="finite"):
        technical_result_variance(0.5, 10.0, bad_val, 100.0, 400.0)
    with pytest.raises(ValueError, match="finite"):
        technical_result_variance(0.5, 10.0, 10.0, bad_val, 400.0)
    with pytest.raises(ValueError, match="finite"):
        technical_result_variance(0.5, 10.0, 10.0, 100.0, bad_val)

    # 2. optimal_reinsurance
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance(bad_val, 1000.0, 0.20, target_variance=500.0)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance(100.0, bad_val, 0.20, target_variance=500.0)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance(100.0, 1000.0, bad_val, target_variance=500.0)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance(100.0, 1000.0, 0.20, target_variance=bad_val)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance(100.0, 1000.0, 0.20, criterion="mean_variance", risk_aversion=bad_val)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance(100.0, 1000.0, 0.20, criterion="premium_budget", premium_budget=bad_val)

    # 3. insurer_reciprocity_metric
    with pytest.raises(ValueError, match="finite"):
        insurer_reciprocity_metric(bad_val, 100.0)
    with pytest.raises(ValueError, match="finite"):
        insurer_reciprocity_metric(100.0, bad_val)
    with pytest.raises(ValueError, match="finite"):
        insurer_reciprocity_metric(100.0, 100.0, quota_1=bad_val)
    with pytest.raises(ValueError, match="finite"):
        insurer_reciprocity_metric(100.0, 100.0, correlation=bad_val)
    with pytest.raises(ValueError, match="finite"):
        insurer_reciprocity_metric(100.0, 100.0, expected_loss_1=bad_val, expected_loss_2=100.0)


def test_stability_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="retention_rate"):
        technical_result_variance(-0.1, 5.0, 5.0, 50.0, 25.0)

    with pytest.raises(ValueError, match="Frequency moments"):
        technical_result_variance(0.5, -5.0, 5.0, 50.0, 25.0)

    with pytest.raises(ValueError, match="Severity moments"):
        technical_result_variance(0.5, 5.0, 5.0, 50.0, -25.0)

    with pytest.raises(ValueError, match="target_variance"):
        optimal_reinsurance(100.0, 400.0, 0.2, criterion="target_variance")

    with pytest.raises(ValueError, match="target_variance"):
        optimal_reinsurance(100.0, 400.0, 0.2, target_variance=-5.0, criterion="target_variance")

    with pytest.raises(ValueError, match="risk_aversion"):
        optimal_reinsurance(100.0, 400.0, 0.2, criterion="mean_variance")

    with pytest.raises(ValueError, match="risk_aversion"):
        optimal_reinsurance(100.0, 400.0, 0.2, risk_aversion=-1.0, criterion="mean_variance")

    with pytest.raises(ValueError, match="premium_budget"):
        optimal_reinsurance(100.0, 400.0, 0.2, criterion="premium_budget")

    with pytest.raises(ValueError, match="premium_budget"):
        optimal_reinsurance(100.0, 400.0, 0.2, premium_budget=-10.0, criterion="premium_budget")

    with pytest.raises(ValueError, match="expected_loss"):
        optimal_reinsurance(-10.0, 400.0, 0.2)

    with pytest.raises(ValueError, match="variance_loss"):
        optimal_reinsurance(100.0, -400.0, 0.2)

    with pytest.raises(ValueError, match="reinsurer_loading"):
        optimal_reinsurance(100.0, 400.0, -0.2)

    with pytest.raises(ValueError, match="criterion"):
        optimal_reinsurance(100.0, 400.0, 0.2, criterion="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="variances"):
        insurer_reciprocity_metric(variance_1=-100.0, variance_2=100.0)

    with pytest.raises(ValueError, match="Expected losses"):
        insurer_reciprocity_metric(
            variance_1=100.0,
            variance_2=100.0,
            expected_loss_1=-50.0,
            expected_loss_2=50.0,
        )

    with pytest.raises(ValueError, match="quota_2 > 1.0"):
        insurer_reciprocity_metric(
            variance_1=100.0,
            variance_2=100.0,
            expected_loss_1=1000.0,
            expected_loss_2=100.0,
            quota_1=0.8,
        )

    with pytest.raises(ValueError, match="quota_1"):
        insurer_reciprocity_metric(variance_1=100.0, variance_2=100.0, quota_1=1.5)

    with pytest.raises(ValueError, match="correlation"):
        insurer_reciprocity_metric(variance_1=100.0, variance_2=100.0, correlation=1.2)
