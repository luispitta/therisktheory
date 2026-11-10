"""Unit tests for free-parameter credibility, reserving models, and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.credibility import (
    credibility_reserve_estimate,
    free_parameter_credibility,
    reserve_model_diagnostics,
)

# ============================================================================
# Tests for free_parameter_credibility
# ============================================================================


def test_free_parameter_credibility_kaas_example() -> None:
    """Benchmark test against Kaas et al. (2008), Example 8.2.1 & 8.2.5.

    3 groups observed for 5 years with equal exposure:
    Group means: 100.0, 110.0, 120.0; overall mean: 110.0.
    s^2 = 108.97, a = 78.206, Z = 0.78206.
    Credibility premiums: [102.18, 110.00, 117.82].
    """
    kaas_data = np.array(
        [
            [99.3, 93.7, 103.9, 92.5, 110.6],
            [112.5, 108.3, 118.0, 99.4, 111.8],
            [129.2, 140.9, 108.3, 105.0, 116.6],
        ]
    )

    res = free_parameter_credibility(kaas_data)

    # 1. Overall mean: (100.0 + 110.0 + 120.0) / 3 = 110.0
    assert res["overall_sample_mean"] == pytest.approx(110.0)
    assert res["collective_mean"] == pytest.approx(110.0)

    # 2. Individual sample means
    expected_means = np.array([100.0, 110.0, 120.0])
    np.testing.assert_allclose(res["individual_means"], expected_means)

    # 3. Process variance: s^2 = 108.97
    assert res["process_variance"] == pytest.approx(108.97)

    # 4. Between-risk variance: a = 78.206
    assert res["variance_hypothetical_means"] == pytest.approx(78.206)

    # 5. Parameter ratio: k = 108.97 / 78.206 = 1.39337...
    expected_k = 108.97 / 78.206
    assert res["k_ratio"] == pytest.approx(expected_k)

    # 6. Credibility factor: Z = 5 / (5 + k) = 0.78206...
    expected_z = 5.0 / (5.0 + expected_k)
    np.testing.assert_allclose(res["credibility_factors"], np.full(3, expected_z))

    # 7. Credibility premiums: [102.18, 110.00, 117.82]
    expected_premiums = np.array([102.1794, 110.0, 117.8206])
    np.testing.assert_allclose(res["credibility_premiums"], expected_premiums, atol=1e-4)

    # 8. Portfolio balance: sum(P_i) == sum(X_bar_i) = 330.0
    assert np.sum(res["credibility_premiums"]) == pytest.approx(np.sum(expected_means))


def test_free_parameter_credibility_weighted_buhlmann_straub() -> None:
    """Test non-parametric credibility with unequal exposure weights."""
    # 3 risks, 3 years
    losses = np.array(
        [
            [10.0, 12.0, 14.0],
            [20.0, 22.0, 24.0],
            [30.0, 32.0, 34.0],
        ]
    )
    weights = np.array(
        [
            [100.0, 150.0, 200.0],  # P_1. = 450
            [200.0, 200.0, 200.0],  # P_2. = 600
            [50.0, 50.0, 100.0],  # P_3. = 200
        ]
    )

    res = free_parameter_credibility(losses, weights=weights)

    # Credibility factors should vary with row totals P_{i.}
    z_factors = res["credibility_factors"]
    assert len(z_factors) == 3
    # Risk 2 has highest weight (600) -> highest credibility factor
    assert z_factors[1] > z_factors[0] > z_factors[2]
    assert np.all((z_factors >= 0.0) & (z_factors <= 1.0))

    # Portfolio balance condition: sum P_{i.} * P_i == sum P_{i.} * X_bar_i
    row_weights = res["total_weights"]
    total_fitted = np.sum(row_weights * res["credibility_premiums"])
    total_observed = np.sum(row_weights * res["individual_means"])
    assert total_fitted == pytest.approx(total_observed)


def test_free_parameter_credibility_unequal_exposures_comprehensive() -> None:
    """Verify credibility behavior under wide dynamic range of unequal exposures."""
    # 4 risks observed over 3 years with extreme exposure differences:
    # Risk 1: Small boutique risk (P_{1.} = 15)
    # Risk 2: Medium commercial line (P_{2.} = 150)
    # Risk 3: Large regional portfolio (P_{3.} = 1500)
    # Risk 4: National corporate fleet (P_{4.} = 15000)
    weights = np.array(
        [
            [5.0, 5.0, 5.0],  # total = 15
            [50.0, 50.0, 50.0],  # total = 150
            [500.0, 500.0, 500.0],  # total = 1500
            [5000.0, 5000.0, 5000.0],  # total = 15000
        ]
    )
    # Different loss ratios across risks
    losses = np.array(
        [
            [0.90, 0.95, 0.85],  # mean = 0.90
            [0.70, 0.72, 0.68],  # mean = 0.70
            [0.55, 0.53, 0.57],  # mean = 0.55
            [0.40, 0.41, 0.39],  # mean = 0.40
        ]
    )

    res = free_parameter_credibility(losses, weights=weights)

    z = res["credibility_factors"]
    # 1. Monotonicity: Credibility strictly increases with exposure volume
    assert z[0] < z[1] < z[2] < z[3]

    # 2. Asymptotic properties:
    # Massive risk (P = 15000) achieves near-full credibility Z ~ 1.0
    assert z[3] > 0.95
    # Smallest risk (P = 15) has low credibility, shrinking toward collective mean
    assert z[0] < z[3]

    # 3. Shrinkage towards collective mean:
    # Premium for smallest risk is pulled toward collective mean mu_Z
    p_hat = res["credibility_premiums"]
    mu_z = res["collective_mean"]
    assert abs(p_hat[0] - mu_z) < abs(res["individual_means"][0] - mu_z)

    # 4. Exact portfolio balance condition under unequal exposures
    row_weights = res["total_weights"]
    fitted_volume = float(np.sum(row_weights * p_hat))
    observed_volume = float(np.sum(row_weights * res["individual_means"]))
    assert fitted_volume == pytest.approx(observed_volume, rel=1e-7)

    # 5. Diagnostic balance check
    diag = reserve_model_diagnostics(
        observed_values=res["individual_means"],
        credibility_estimates=p_hat,
        collective_mean=mu_z,
        credibility_factors=z,
        weights=row_weights,
        process_variance=res["process_variance"],
        hypothetical_mean_variance=res["variance_hypothetical_means"],
    )
    assert abs(diag["portfolio_imbalance"]) == pytest.approx(0.0, abs=1e-6)
    assert abs(diag["relative_imbalance"]) == pytest.approx(0.0, abs=1e-6)


def test_free_parameter_credibility_homogeneous_truncation() -> None:
    """Test truncation of between-group variance when a_raw < 0."""
    # Data with high within-variance and identical group means
    data = np.array(
        [
            [10.0, 50.0, 30.0],
            [30.0, 20.0, 40.0],
            [25.0, 35.0, 30.0],
        ]
    )
    res = free_parameter_credibility(data)

    # All group means are 30.0 -> SSB = 0, so a_raw < 0
    assert res["raw_variance_hypothetical_means"] <= 0.0
    assert res["variance_hypothetical_means"] == 0.0
    assert np.isinf(res["k_ratio"])
    # All credibility factors collapse to 0.0
    np.testing.assert_allclose(res["credibility_factors"], np.zeros(3))
    # All premiums equal collective mean
    np.testing.assert_allclose(res["credibility_premiums"], np.full(3, 30.0))


# ============================================================================
# Tests for credibility_reserve_estimate
# ============================================================================


def test_credibility_reserve_estimate_bornhuetter_ferguson() -> None:
    """Test standard Bornhuetter-Ferguson credibility bridge (Z = 1/f)."""
    # Reported claims C = 60,000; Expected ultimate M = 100,000; CDF f = 1.25
    # Emergence p = 1 / 1.25 = 0.80 -> Z = 0.80
    # CL ultimate = 60,000 * 1.25 = 75,000
    # Cred ultimate = 0.80 * 75,000 + 0.20 * 100,000 = 60,000 + 20,000 = 80,000
    # Reserve = 80,000 - 60,000 = 20,000
    res = credibility_reserve_estimate(
        reported_claims=60000.0,
        expected_claims=100000.0,
        development_factor=1.25,
        premiums=125000.0,
    )
    assert np.asarray(res["emergence_percentage"]).item() == pytest.approx(0.80)
    assert np.asarray(res["credibility_factors"]).item() == pytest.approx(0.80)
    assert np.asarray(res["chain_ladder_ultimate"]).item() == pytest.approx(75000.0)
    assert np.asarray(res["ultimate_claims"]).item() == pytest.approx(80000.0)
    assert np.asarray(res["reserve"]).item() == pytest.approx(20000.0)
    # Implied loss ratio = 80,000 / 125,000 = 0.64
    assert np.asarray(res["implied_loss_ratio"]).item() == pytest.approx(0.64)


def test_credibility_reserve_estimate_custom_credibility() -> None:
    """Test reserving with explicit credibility factors."""
    # Full credibility Z = 1.0 (Chain-Ladder)
    res_cl = credibility_reserve_estimate(
        reported_claims=50000.0,
        expected_claims=100000.0,
        development_factor=2.0,
        credibility_factor=1.0,
    )
    assert np.asarray(res_cl["ultimate_claims"]).item() == pytest.approx(100000.0)
    assert np.asarray(res_cl["reserve"]).item() == pytest.approx(50000.0)

    # Zero credibility Z = 0.0 (Prior only)
    res_prior = credibility_reserve_estimate(
        reported_claims=50000.0,
        expected_claims=120000.0,
        development_factor=2.0,
        credibility_factor=0.0,
    )
    assert np.asarray(res_prior["ultimate_claims"]).item() == pytest.approx(120000.0)
    assert np.asarray(res_prior["reserve"]).item() == pytest.approx(70000.0)


def test_credibility_reserve_estimate_vectorized() -> None:
    # 3 cohorts: accident years 2021, 2022, 2023
    c = np.array([90000.0, 60000.0, 20000.0])
    m = np.array([100000.0, 100000.0, 100000.0])
    f = np.array([1.10, 1.50, 4.00])  # emergence = [0.909, 0.667, 0.25]

    res = credibility_reserve_estimate(c, m, f)
    assert len(res["reserve"]) == 3
    # More mature years have smaller relative reserve
    assert res["reserve"][0] < res["reserve"][1]


# ============================================================================
# Tests for reserve_model_diagnostics
# ============================================================================


def test_reserve_model_diagnostics_metrics() -> None:
    obs = np.array([100.0, 110.0, 120.0, 130.0])
    fit = np.array([105.0, 110.0, 115.0, 125.0])
    z = np.array([0.7, 0.8, 0.6, 0.9])
    coll_mean = 115.0

    diag = reserve_model_diagnostics(
        observed_values=obs,
        credibility_estimates=fit,
        collective_mean=coll_mean,
        credibility_factors=z,
        process_variance=25.0,
        hypothetical_mean_variance=30.0,
    )

    assert "portfolio_imbalance" in diag
    assert "relative_imbalance" in diag
    assert "weighted_mae" in diag
    assert "weighted_rmse" in diag
    assert diag["mean_credibility"] == pytest.approx(0.75)
    assert diag["high_credibility_ratio"] == 1.0  # all Z >= 0.5

    # Theoretical MSE = (1 - Z) * a
    expected_mse = (1.0 - z) * 30.0
    np.testing.assert_allclose(diag["theoretical_mse"], expected_mse)

    # Standardized residuals check
    assert "standardized_residuals" in diag
    expected_std_resid = (obs - fit) / np.sqrt(25.0)
    np.testing.assert_allclose(diag["standardized_residuals"], expected_std_resid)


# ============================================================================
# Tests for validation, non-finite inputs, and domain bounds
# ============================================================================


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_credibility_reserve_estimate_non_finite_rejection(bad_val: float) -> None:
    """Verify that credibility_reserve_estimate strictly rejects NaN and Inf."""
    with pytest.raises(ValueError, match="finite"):
        credibility_reserve_estimate(bad_val, 100.0, 1.5)

    with pytest.raises(ValueError, match="finite"):
        credibility_reserve_estimate(50.0, bad_val, 1.5)

    with pytest.raises(ValueError, match="finite"):
        credibility_reserve_estimate(50.0, 100.0, bad_val)

    with pytest.raises(ValueError, match="finite"):
        credibility_reserve_estimate(50.0, 100.0, 1.5, credibility_factor=bad_val)

    with pytest.raises(ValueError, match="finite"):
        credibility_reserve_estimate(50.0, 100.0, 1.5, premiums=bad_val)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_reserve_model_diagnostics_non_finite_rejection(bad_val: float) -> None:
    """Verify that reserve_model_diagnostics strictly rejects NaN and Inf."""
    obs = np.array([10.0, 20.0])
    fit = np.array([12.0, 18.0])
    z = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics([bad_val, 20.0], fit, 15.0, z)

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics(obs, [bad_val, 18.0], 15.0, z)

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics(obs, fit, bad_val, z)

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics(obs, fit, 15.0, [bad_val, 0.5])

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics(obs, fit, 15.0, z, weights=[bad_val, 1.0])

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics(obs, fit, 15.0, z, process_variance=bad_val)

    with pytest.raises(ValueError, match="finite"):
        reserve_model_diagnostics(obs, fit, 15.0, z, hypothetical_mean_variance=bad_val)


def test_free_parameter_credibility_invalid_inputs() -> None:
    # 1D array
    with pytest.raises(ValueError, match="2D"):
        free_parameter_credibility(np.array([10.0, 20.0, 30.0]))

    # Less than 2 risks
    with pytest.raises(ValueError, match="at least 2 risks"):
        free_parameter_credibility(np.array([[10.0, 20.0]]))

    # Less than 2 periods
    with pytest.raises(ValueError, match="at least 2 periods"):
        free_parameter_credibility(np.array([[10.0], [20.0]]))

    # Non-finite inputs
    with pytest.raises(ValueError, match="non-finite"):
        free_parameter_credibility(np.array([[10.0, np.nan], [20.0, 30.0]]))

    # Mismatched weights shape
    with pytest.raises(ValueError, match="shape"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[1.0], [2.0]]),
        )

    # Negative weights
    with pytest.raises(ValueError, match="non-negative"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[-1.0, 1.0], [1.0, 1.0]]),
        )

    # Risk with zero total weight
    with pytest.raises(ValueError, match="strictly positive total weight"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[0.0, 0.0], [1.0, 1.0]]),
        )

    # Insufficient degrees of freedom (e.g. only 1 active period per risk)
    with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )


def test_credibility_reserve_estimate_invalid_inputs() -> None:
    # Negative reported claims
    with pytest.raises(ValueError, match="reported_claims"):
        credibility_reserve_estimate(-10.0, 100.0, 1.5)

    # Non-positive expected claims
    with pytest.raises(ValueError, match="expected_claims"):
        credibility_reserve_estimate(10.0, 0.0, 1.5)

    # Development factor < 1.0
    with pytest.raises(ValueError, match="development_factor"):
        credibility_reserve_estimate(10.0, 100.0, 0.9)

    # Credibility factor outside [0, 1]
    with pytest.raises(ValueError, match="credibility_factor"):
        credibility_reserve_estimate(10.0, 100.0, 1.5, credibility_factor=1.2)

    # Non-positive premium
    with pytest.raises(ValueError, match="premiums"):
        credibility_reserve_estimate(10.0, 100.0, 1.5, premiums=-50.0)


def test_reserve_model_diagnostics_invalid_inputs() -> None:
    # 2D array instead of 1D
    with pytest.raises(ValueError, match="1D"):
        reserve_model_diagnostics([[10.0, 20.0]], [[10.0, 20.0]], 15.0, [0.5, 0.5])

    # Mismatched lengths
    with pytest.raises(ValueError, match="identical length"):
        reserve_model_diagnostics([10.0, 20.0], [10.0], 15.0, [0.5, 0.5])

    # Factor outside [0, 1]
    with pytest.raises(ValueError, match="credibility_factors"):
        reserve_model_diagnostics([10.0], [10.0], 10.0, [-0.1])

    # Weights shape mismatch
    with pytest.raises(ValueError, match="weights must be non-negative"):
        reserve_model_diagnostics([10.0, 20.0], [10.0, 20.0], 15.0, [0.5, 0.5], weights=[1.0])

    # Negative weights
    with pytest.raises(ValueError, match="weights must be non-negative"):
        reserve_model_diagnostics([10.0, 20.0], [10.0, 20.0], 15.0, [0.5, 0.5], weights=[-1.0, 1.0])

    # Total weight zero
    with pytest.raises(ValueError, match="total weight must be strictly positive"):
        reserve_model_diagnostics([10.0, 20.0], [10.0, 20.0], 15.0, [0.5, 0.5], weights=[0.0, 0.0])

    # Negative hypothetical mean variance
    with pytest.raises(ValueError, match="hypothetical_mean_variance"):
        reserve_model_diagnostics([10.0], [10.0], 10.0, [0.5], hypothetical_mean_variance=-1.0)

    # Negative process variance
    with pytest.raises(ValueError, match="process_variance"):
        reserve_model_diagnostics([10.0], [10.0], 10.0, [0.5], process_variance=-1.0)


def test_free_parameter_credibility_degenerate_c_star(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify defensive check for degenerate exposure structure c* <= 0."""
    import therisktheory.credibility.free_parameter as fp_module

    orig_sum = fp_module.np.sum

    def patched_sum(a, *args, **kwargs):
        res = orig_sum(a, *args, **kwargs)
        if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == 2 and np.all(a == 4.0):
            return 100.0
        return res

    monkeypatch.setattr(fp_module.np, "sum", patched_sum)
    with pytest.raises(ValueError, match="c\\* <= 0"):
        free_parameter_credibility([[10.0, 20.0], [30.0, 40.0]])
