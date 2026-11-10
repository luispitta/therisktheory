"""Unit tests for De Finetti proportional reinsurance and retention optimization."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.finetti_ruin import (
    de_finetti_optimal_retention,
    de_finetti_retention_ruin_approximation,
    de_finetti_retention_ruin_bound,
    proportional_reinsurance_objective,
)

# ============================================================================
# Tests for proportional_reinsurance_objective
# ============================================================================


def test_proportional_reinsurance_objective_scalar_safety_index() -> None:
    # mu = 100, var = 400, eta = 0.15, xi = 0.20, alpha = 0.6
    # E[profit] = (0.15 - 0.40 * 0.20) * 100 = (0.15 - 0.08) * 100 = 7.0
    # Retained var = 0.6^2 * 400 = 0.36 * 400 = 144.0
    # theta = 7.0 / 144.0 = 0.04861111111111111
    val = proportional_reinsurance_objective(
        retention_rate=0.6,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.15,
        reinsurer_loading=0.20,
        criterion="safety_index",
    )
    assert np.asarray(val).item() == pytest.approx(7.0 / 144.0)


def test_proportional_reinsurance_objective_criteria() -> None:
    # Expected profit: (eta - (1 - alpha)*xi) * mu
    profit = proportional_reinsurance_objective(
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=200.0,
        insurer_loading=0.10,
        reinsurer_loading=0.20,
        criterion="expected_profit",
    )
    # (0.10 - 0.5 * 0.20) * 100 = 0.0
    assert np.asarray(profit).item() == pytest.approx(0.0)

    # Retained variance: alpha^2 * var
    var_res = proportional_reinsurance_objective(
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=200.0,
        insurer_loading=0.10,
        reinsurer_loading=0.20,
        criterion="retained_variance",
    )
    assert np.asarray(var_res).item() == pytest.approx(0.25 * 200.0)

    # Mean-variance: profit - 0.5 * k * var
    mv = proportional_reinsurance_objective(
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=200.0,
        insurer_loading=0.20,
        reinsurer_loading=0.20,
        criterion="mean_variance",
        risk_aversion=0.01,
    )
    # profit = (0.20 - 0.10) * 100 = 10.0
    # penalty = 0.5 * 0.01 * 50 = 0.25
    assert np.asarray(mv).item() == pytest.approx(9.75)


def test_proportional_reinsurance_objective_vectorized() -> None:
    alphas = np.array([0.2, 0.5, 0.8])
    res = proportional_reinsurance_objective(
        retention_rate=alphas,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.25,
        reinsurer_loading=0.25,
        criterion="expected_profit",
    )
    # profit = (0.25 - (1 - alpha)*0.25) * 100 = alpha * 0.25 * 100 = alpha * 25
    expected = alphas * 25.0
    np.testing.assert_allclose(res, expected)


def test_zero_retention_regimes() -> None:
    # 1. Arbitrage regime: eta > xi -> lim_{alpha -> 0+} theta(alpha) = +inf
    theta_arbitrage = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.20,
        reinsurer_loading=0.10,
        criterion="safety_index",
    )
    assert np.isposinf(np.asarray(theta_arbitrage).item())

    # 2. Equal positive loadings: eta == xi > 0 -> +inf
    theta_equal = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.15,
        reinsurer_loading=0.15,
        criterion="safety_index",
    )
    assert np.isposinf(np.asarray(theta_equal).item())

    # 3. Equal zero loadings: eta == xi == 0 -> 0.0
    theta_zero = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.0,
        reinsurer_loading=0.0,
        criterion="safety_index",
    )
    assert np.asarray(theta_zero).item() == 0.0

    # 4. Standard costly reinsurance: xi > eta -> lim_{alpha -> 0+} theta(alpha) = -inf
    theta_costly = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.10,
        reinsurer_loading=0.20,
        criterion="safety_index",
    )
    assert np.isneginf(np.asarray(theta_costly).item())


# ============================================================================
# Tests for de_finetti_optimal_retention & Boundary Contrast
# ============================================================================


def test_professor_observation_counterexample() -> None:
    """Test the exact case from the instructor review: mu=1, var=1, eta=0.20, xi=0.10.

    When eta=0.20 > xi=0.10 (reinsurance cheaper than primary), ceding 100% of risk
    yields a guaranteed arbitrage drift of (0.20 - 0.10)*1 = 0.10 with 0 retained variance.
    The safety index theta(0) evaluates to +inf. At alpha=0.10, theta(0.10) evaluates to 11.0.
    Thus alpha*=0.0 strictly achieves the supremum (+inf >= 11.0).
    """
    mu = 1.0
    var = 1.0
    eta = 0.20
    xi = 0.10

    # 1. Optimal retention selects alpha* = 0.0
    alpha_star = de_finetti_optimal_retention(
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.asarray(alpha_star).item() == 0.0

    # 2. Objective function at alpha* = 0.0 evaluates to +inf
    theta_star = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.isposinf(np.asarray(theta_star).item())

    # 3. Objective function at alpha = 0.10 evaluates to 11.0
    theta_01 = proportional_reinsurance_objective(
        retention_rate=0.10,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.asarray(theta_01).item() == pytest.approx(11.0)

    # 4. Check that theta(alpha*) >= theta(alpha) for all alpha in (0, 1]
    grid_alphas = np.linspace(0.01, 1.0, 100)
    grid_thetas = proportional_reinsurance_objective(
        retention_rate=grid_alphas,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.all(grid_thetas < np.asarray(theta_star).item())


def test_boundary_contrast_interior_optimum() -> None:
    """Test regime 2: eta < xi < 2*eta, where alpha* in (0, 1) strictly beats boundaries."""
    mu = 100.0
    var = 400.0
    eta = 0.15
    xi = 0.20  # eta < xi < 2*eta (0.15 < 0.20 < 0.30)

    # Stationary point alpha* = 2 * (1 - 0.15 / 0.20) = 0.5
    alpha_star = de_finetti_optimal_retention(
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    opt_val = np.asarray(alpha_star).item()
    assert opt_val == pytest.approx(0.5)

    # Objective at optimum: theta(0.5) = (0.15 - 0.10)*100 / (0.25 * 400) = 5.0 / 100 = 0.05
    theta_opt = proportional_reinsurance_objective(
        retention_rate=opt_val,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.asarray(theta_opt).item() == pytest.approx(0.05)

    # Left boundary alpha = 0: xi > eta -> -inf
    theta_0 = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.isneginf(np.asarray(theta_0).item())

    # Right boundary alpha = 1: theta(1) = 0.15 * 100 / 400 = 15.0 / 400 = 0.0375
    theta_1 = proportional_reinsurance_objective(
        retention_rate=1.0,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.asarray(theta_1).item() == pytest.approx(0.0375)

    # Optimum strictly beats both boundaries
    assert np.asarray(theta_opt).item() > np.asarray(theta_1).item()
    assert np.asarray(theta_opt).item() > np.asarray(theta_0).item()


def test_boundary_contrast_expensive_reinsurance() -> None:
    """Test regime 3: xi >= 2*eta, where upper boundary alpha* = 1.0 is optimal."""
    mu = 100.0
    var = 400.0
    eta = 0.10
    xi = 0.30  # xi >= 2*eta (0.30 >= 0.20)

    # Optimal retention should be 1.0 (reinsurance is too expensive)
    alpha_star = de_finetti_optimal_retention(
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.asarray(alpha_star).item() == pytest.approx(1.0)

    # Objective at alpha = 1.0
    theta_1 = proportional_reinsurance_objective(
        retention_rate=1.0,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    # E[profit] = 0.10 * 100 = 10.0; retained var = 400; theta = 10 / 400 = 0.025
    assert np.asarray(theta_1).item() == pytest.approx(0.025)

    # Left boundary alpha = 0 is -inf
    theta_0 = proportional_reinsurance_objective(
        retention_rate=0.0,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.isneginf(np.asarray(theta_0).item())

    # Any interior alpha yields strictly lower theta than alpha = 1.0
    interior_alphas = np.linspace(0.1, 0.9, 9)
    interior_thetas = proportional_reinsurance_objective(
        retention_rate=interior_alphas,
        expected_loss=mu,
        variance_loss=var,
        insurer_loading=eta,
        reinsurer_loading=xi,
        criterion="safety_index",
    )
    assert np.all(interior_thetas < np.asarray(theta_1).item())


def test_de_finetti_optimal_retention_mean_variance() -> None:
    # alpha* = clip(xi * mu / (k * var), 0.0, 1.0)
    # xi = 0.20, mu = 100, k = 0.001, var = 40000
    # xi * mu = 20.0, k * var = 40.0 -> alpha* = 0.5
    opt_mv = de_finetti_optimal_retention(
        expected_loss=100.0,
        variance_loss=40000.0,
        insurer_loading=0.15,
        reinsurer_loading=0.20,
        criterion="mean_variance",
        risk_aversion=0.001,
    )
    assert np.asarray(opt_mv).item() == pytest.approx(0.5)

    # Extreme risk aversion -> very low retention
    opt_high_k = de_finetti_optimal_retention(
        expected_loss=100.0,
        variance_loss=40000.0,
        insurer_loading=0.15,
        reinsurer_loading=0.20,
        criterion="mean_variance",
        risk_aversion=1.0,
    )
    assert np.asarray(opt_high_k).item() == pytest.approx(0.0005)


def test_de_finetti_optimal_retention_vectorized() -> None:
    etas = np.array([0.15, 0.10, 0.18])
    xis = np.array([0.20, 0.20, 0.20])
    opts = de_finetti_optimal_retention(
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=etas,
        reinsurer_loading=xis,
        criterion="safety_index",
    )
    # alpha* = 2 * (1 - eta / 0.20)
    # 0.15 -> 2 * (1 - 0.75) = 0.5
    # 0.10 -> 2 * (1 - 0.50) = 1.0
    # 0.18 -> 2 * (1 - 0.90) = 0.2
    np.testing.assert_allclose(opts, np.array([0.5, 1.0, 0.2]))


# ============================================================================
# Tests for de_finetti_retention_ruin_approximation & Alias
# ============================================================================


def test_de_finetti_retention_ruin_approximation_properties() -> None:
    # At u = 0, exp(0) = 1.0
    bound_u0 = de_finetti_retention_ruin_approximation(
        initial_capital=0.0,
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.15,
        reinsurer_loading=0.20,
    )
    assert np.asarray(bound_u0).item() == pytest.approx(1.0)

    # For large capital, ruin probability decays to zero
    bound_large_u = de_finetti_retention_ruin_approximation(
        initial_capital=1000.0,
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.15,
        reinsurer_loading=0.20,
    )
    assert float(bound_large_u) < 1e-4

    # Negative net drift (loss making retention) gives ruin prob = 1.0
    bound_neg = de_finetti_retention_ruin_approximation(
        initial_capital=100.0,
        retention_rate=0.1,  # (0.10 - 0.9 * 0.30) = 0.10 - 0.27 = -0.17 < 0
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.10,
        reinsurer_loading=0.30,
    )
    assert np.asarray(bound_neg).item() == pytest.approx(1.0)


def test_de_finetti_retention_ruin_infinite_theta() -> None:
    """When theta = +inf (arbitrage limit), ruin probability for u > 0 is 0.0."""
    res_pos_u = de_finetti_retention_ruin_approximation(
        initial_capital=10.0,
        retention_rate=0.0,
        expected_loss=1.0,
        variance_loss=1.0,
        insurer_loading=0.20,
        reinsurer_loading=0.10,
    )
    assert np.asarray(res_pos_u).item() == 0.0

    res_zero_u = de_finetti_retention_ruin_approximation(
        initial_capital=0.0,
        retention_rate=0.0,
        expected_loss=1.0,
        variance_loss=1.0,
        insurer_loading=0.20,
        reinsurer_loading=0.10,
    )
    assert np.asarray(res_zero_u).item() == 1.0


def test_degenerate_zero_reserve_and_zero_drift() -> None:
    """Verify degenerate boundary conditions: zero initial reserve and zero drift."""
    # 1. Zero initial reserve u = 0 with positive drift -> ruin is certain (psi = 1.0)
    ruin_u0 = de_finetti_retention_ruin_approximation(
        initial_capital=0.0,
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.20,
        reinsurer_loading=0.20,
    )
    assert np.asarray(ruin_u0).item() == 1.0

    # 2. Zero drift theta = 0 with positive initial reserve u > 0 -> psi = 1.0
    # Expected profit = (eta - (1 - alpha) * xi) * mu
    # For alpha = 0.5, eta = 0.10, xi = 0.20: (0.10 - 0.5 * 0.20) * mu = 0.0 -> theta = 0.0
    ruin_drift0 = de_finetti_retention_ruin_approximation(
        initial_capital=100.0,
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.10,
        reinsurer_loading=0.20,
    )
    assert np.asarray(ruin_drift0).item() == 1.0

    # 3. Degenerate null reserve AND null drift (u = 0, theta = 0) -> psi = 1.0
    ruin_both0 = de_finetti_retention_ruin_approximation(
        initial_capital=0.0,
        retention_rate=0.5,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=0.10,
        reinsurer_loading=0.20,
    )
    assert np.asarray(ruin_both0).item() == 1.0

    # 4. Vectorized degenerate boundary cases
    u_vec = np.array([0.0, 50.0, 0.0, 100.0])
    alpha_vec = np.array([0.5, 0.5, 0.5, 0.5])
    eta_vec = np.array([0.20, 0.10, 0.10, 0.30])
    res_vec = de_finetti_retention_ruin_approximation(
        initial_capital=u_vec,
        retention_rate=alpha_vec,
        expected_loss=100.0,
        variance_loss=400.0,
        insurer_loading=eta_vec,
        reinsurer_loading=0.20,
    )
    assert res_vec[0] == 1.0  # u = 0
    assert res_vec[1] == 1.0  # theta = 0
    assert res_vec[2] == 1.0  # u = 0 and theta = 0
    assert 0.0 < res_vec[3] < 1.0  # u > 0, theta > 0


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_rejection_of_non_finite_inputs(bad_val: float) -> None:
    """Verify that all public functions reject non-finite inputs (NaN and Inf)."""
    # 1. proportional_reinsurance_objective
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_objective(bad_val, 100.0, 400.0, 0.15, 0.20)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_objective(0.5, bad_val, 400.0, 0.15, 0.20)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_objective(0.5, 100.0, bad_val, 0.15, 0.20)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_objective(0.5, 100.0, 400.0, bad_val, 0.20)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_objective(0.5, 100.0, 400.0, 0.15, bad_val)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_objective(
            0.5, 100.0, 400.0, 0.15, 0.20, criterion="mean_variance", risk_aversion=bad_val
        )

    # 2. de_finetti_optimal_retention
    with pytest.raises(ValueError, match="finite"):
        de_finetti_optimal_retention(bad_val, 400.0, 0.15, 0.20)
    with pytest.raises(ValueError, match="finite"):
        de_finetti_optimal_retention(100.0, bad_val, 0.15, 0.20)
    with pytest.raises(ValueError, match="finite"):
        de_finetti_optimal_retention(100.0, 400.0, bad_val, 0.20)
    with pytest.raises(ValueError, match="finite"):
        de_finetti_optimal_retention(100.0, 400.0, 0.15, bad_val)
    with pytest.raises(ValueError, match="finite"):
        de_finetti_optimal_retention(
            100.0, 400.0, 0.15, 0.20, criterion="mean_variance", risk_aversion=bad_val
        )

    # 3. de_finetti_retention_ruin_approximation
    with pytest.raises(ValueError, match="finite"):
        de_finetti_retention_ruin_approximation(bad_val, 0.5, 100.0, 400.0, 0.15, 0.20)


def test_alias_backward_compatibility() -> None:
    """Verify de_finetti_retention_ruin_bound is identical to approximation."""
    assert de_finetti_retention_ruin_bound is de_finetti_retention_ruin_approximation

    val_bound = de_finetti_retention_ruin_bound(50.0, 0.5, 100.0, 400.0, 0.15, 0.20)
    val_approx = de_finetti_retention_ruin_approximation(50.0, 0.5, 100.0, 400.0, 0.15, 0.20)
    assert np.asarray(val_bound).item() == np.asarray(val_approx).item()


# ============================================================================
# Tests for domain and error handling
# ============================================================================


@pytest.mark.parametrize(
    ("alpha", "mu", "var", "eta", "xi"),
    [
        (-0.1, 100.0, 400.0, 0.1, 0.2),
        (1.1, 100.0, 400.0, 0.1, 0.2),
        (0.5, 0.0, 400.0, 0.1, 0.2),
        (0.5, -10.0, 400.0, 0.1, 0.2),
        (0.5, 100.0, 0.0, 0.1, 0.2),
        (0.5, 100.0, -400.0, 0.1, 0.2),
        (0.5, 100.0, 400.0, -0.1, 0.2),
        (0.5, 100.0, 400.0, 0.1, -0.2),
    ],
)
def test_objective_invalid_inputs(
    alpha: float, mu: float, var: float, eta: float, xi: float
) -> None:
    with pytest.raises(ValueError):
        proportional_reinsurance_objective(alpha, mu, var, eta, xi)


@pytest.mark.parametrize(
    ("mu", "var", "eta", "xi"),
    [
        (0.0, 400.0, 0.1, 0.2),
        (-10.0, 400.0, 0.1, 0.2),
        (100.0, 0.0, 0.1, 0.2),
        (100.0, -400.0, 0.1, 0.2),
        (100.0, 400.0, -0.1, 0.2),
        (100.0, 400.0, 0.1, -0.2),
    ],
)
def test_optimal_retention_invalid_inputs(mu: float, var: float, eta: float, xi: float) -> None:
    with pytest.raises(ValueError):
        de_finetti_optimal_retention(mu, var, eta, xi)


def test_optimal_retention_invalid_criterion_and_risk_aversion() -> None:
    with pytest.raises(ValueError, match="criterion"):
        de_finetti_optimal_retention(
            100.0,
            400.0,
            0.1,
            0.2,
            criterion="unknown",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="risk_aversion"):
        de_finetti_optimal_retention(100.0, 400.0, 0.1, 0.2, criterion="mean_variance")

    with pytest.raises(ValueError, match="risk_aversion"):
        de_finetti_optimal_retention(
            100.0, 400.0, 0.1, 0.2, criterion="mean_variance", risk_aversion=-0.1
        )


def test_invalid_criterion_and_missing_risk_aversion() -> None:
    with pytest.raises(ValueError, match="criterion"):
        proportional_reinsurance_objective(
            0.5,
            100.0,
            400.0,
            0.1,
            0.2,
            criterion="unknown",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="risk_aversion"):
        proportional_reinsurance_objective(0.5, 100.0, 400.0, 0.1, 0.2, criterion="mean_variance")

    with pytest.raises(ValueError, match="risk_aversion"):
        proportional_reinsurance_objective(
            0.5, 100.0, 400.0, 0.1, 0.2, criterion="mean_variance", risk_aversion=-0.5
        )


def test_ruin_bound_invalid_capital() -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        de_finetti_retention_ruin_approximation(
            initial_capital=-50.0,
            retention_rate=0.5,
            expected_loss=100.0,
            variance_loss=400.0,
            insurer_loading=0.15,
            reinsurer_loading=0.20,
        )
