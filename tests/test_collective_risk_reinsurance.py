"""Unit tests for reinsurance effects on the collective risk process and ruin."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.collective_risk import (
    compare_de_finetti_lundberg_retention,
    excess_of_loss_reinsurance_process,
    optimal_reinsurance_retention,
    proportional_reinsurance_process,
    reinsurance_adjustment_coefficient,
    reinsurance_capital_relief,
    reinsurance_ruin_bound,
)
from therisktheory.collective_risk.reinsurance_effects import _excess_of_loss_lundberg_equation

# ============================================================================
# Tests for proportional_reinsurance_process
# ============================================================================


def test_proportional_reinsurance_process_scalar() -> None:
    # lam = 100, mu1 = 2, var = 4, theta = 0.20, xi = 0.30, alpha = 0.8
    # mu1_ret = 0.8 * 2 = 1.6
    # var_ret = 0.8^2 * 4 = 2.56
    # mu2_ret = 2.56 + 1.6^2 = 2.56 + 2.56 = 5.12
    # c_gross = 1.20 * 100 * 2 = 240
    # c_re = 1.30 * 100 * (1 - 0.8) * 2 = 1.30 * 100 * 0.4 = 52
    # c_ret = 240 - 52 = 188
    # theta_ret = (0.20 - 0.06)/0.80 = 0.175
    # drift = 188 - 100 * 1.6 = 28
    # alpha_min = 1 - 0.20 / 0.30 = 1/3 ~ 0.33333333
    res = proportional_reinsurance_process(
        claim_rate=100.0,
        claim_mean=2.0,
        claim_variance=4.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        retention_rate=0.80,
    )
    assert np.asarray(res["retained_claim_mean"]).item() == pytest.approx(1.6)
    assert np.asarray(res["retained_claim_variance"]).item() == pytest.approx(2.56)
    assert np.asarray(res["retained_claim_second_moment"]).item() == pytest.approx(5.12)
    assert np.asarray(res["retained_premium_rate"]).item() == pytest.approx(188.0)
    assert np.asarray(res["retained_safety_loading"]).item() == pytest.approx(0.175)
    assert np.asarray(res["expected_annual_drift"]).item() == pytest.approx(28.0)
    assert np.asarray(res["minimum_viable_retention"]).item() == pytest.approx(1.0 / 3.0)
    assert bool(np.asarray(res["is_solvent_drift"]).item()) is True


def test_proportional_reinsurance_process_vectorized() -> None:
    alphas = np.array([0.5, 0.7, 0.9])
    res = proportional_reinsurance_process(
        claim_rate=50.0,
        claim_mean=10.0,
        claim_variance=25.0,
        safety_loading=0.15,
        reinsurer_loading=0.25,
        retention_rate=alphas,
    )
    expected_means = alphas * 10.0
    np.testing.assert_allclose(res["retained_claim_mean"], expected_means)

    expected_vars = (alphas**2) * 25.0
    np.testing.assert_allclose(res["retained_claim_variance"], expected_vars)


def test_proportional_reinsurance_process_boundary_cases() -> None:
    # Full retention alpha = 1.0 (gross process)
    res_full = proportional_reinsurance_process(10.0, 5.0, 10.0, 0.20, 0.30, 1.0)
    assert np.asarray(res_full["retained_safety_loading"]).item() == pytest.approx(0.20)
    assert np.asarray(res_full["retained_premium_rate"]).item() == pytest.approx(1.20 * 50.0)

    # Insolvent retention below alpha_min (alpha = 0.2 < 1/3)
    res_insolvent = proportional_reinsurance_process(100.0, 2.0, 4.0, 0.20, 0.30, 0.20)
    assert float(np.asarray(res_insolvent["retained_safety_loading"]).item()) < 0.0
    assert bool(np.asarray(res_insolvent["is_solvent_drift"]).item()) is False

    # Zero retention alpha = 0.0 with theta < xi -> -inf loading
    res_zero = proportional_reinsurance_process(10.0, 5.0, 10.0, 0.20, 0.30, 0.0)
    assert np.isneginf(np.asarray(res_zero["retained_safety_loading"]).item())

    # Zero retention alpha = 0.0 with theta >= xi -> +inf loading
    res_zero_arb = proportional_reinsurance_process(10.0, 5.0, 10.0, 0.30, 0.20, 0.0)
    assert np.isposinf(np.asarray(res_zero_arb["retained_safety_loading"]).item())


# ============================================================================
# Tests for excess_of_loss_reinsurance_process
# ============================================================================


def test_excess_of_loss_reinsurance_process_scalar() -> None:
    # Exponential claims with mean 100 (beta = 0.01), priority d = 150
    res = excess_of_loss_reinsurance_process(
        claim_rate=10.0,
        claim_mean=100.0,
        priority=150.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
    )
    exp_factor = np.exp(-1.5)
    assert np.asarray(res["retained_claim_mean"]).item() == pytest.approx(
        100.0 * (1.0 - exp_factor)
    )
    assert np.asarray(res["ceded_claim_mean"]).item() == pytest.approx(100.0 * exp_factor)

    # Invariant: retained mean + ceded mean == gross mean
    gross_recovered = (
        np.asarray(res["retained_claim_mean"]).item() + np.asarray(res["ceded_claim_mean"]).item()
    )
    assert gross_recovered == pytest.approx(100.0)


# ============================================================================
# Tests for reinsurance_adjustment_coefficient
# ============================================================================


def test_reinsurance_adjustment_coefficient_proportional() -> None:
    # Exponential claims with beta = 1.0 (mean = 1.0), theta = 0.20, xi = 0.30
    # At alpha = 1.0 (gross): R = theta / (1 + theta) = 0.20 / 1.20 = 1/6 ~ 0.16666667
    r_gross = reinsurance_adjustment_coefficient(
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        retention_param=1.0,
        treaty_type="proportional",
    )
    assert np.asarray(r_gross).item() == pytest.approx(1.0 / 6.0)

    # At alpha = 0.70 (Prof. Pitta's observation case):
    # num = 1.0 * (0.20 - 0.30*(1 - 0.70)) = 0.11
    # den = 0.70 * (0.70 + 0.20 - 0.30*(1 - 0.70)) = 0.70 * 0.81 = 0.567
    # R = 0.11 / 0.567 ~ 0.194003527
    r_07 = reinsurance_adjustment_coefficient(
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        retention_param=0.70,
        treaty_type="proportional",
    )
    assert np.asarray(r_07).item() == pytest.approx(0.19400352733686066)

    # At alpha = 0.8:
    r_08 = reinsurance_adjustment_coefficient(
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        retention_param=0.80,
        treaty_type="proportional",
    )
    assert np.asarray(r_08).item() == pytest.approx(0.14 / (0.80 * 0.94))

    # Insolvent retention alpha <= alpha_min (alpha = 0.20 < 1/3) -> R = 0.0
    r_insolvent = reinsurance_adjustment_coefficient(
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        retention_param=0.20,
        treaty_type="proportional",
    )
    assert np.asarray(r_insolvent).item() == 0.0


def test_reinsurance_adjustment_coefficient_excess_of_loss() -> None:
    # Exponential claims with mean 1.0 (beta = 1.0), theta = 0.20, xi = 0.25
    # For very large d -> converges to gross adjustment coefficient R_gross = 1/6
    r_large_d = reinsurance_adjustment_coefficient(
        claim_rate=10.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_param=50.0,
        treaty_type="excess_of_loss",
    )
    assert np.asarray(r_large_d).item() == pytest.approx(1.0 / 6.0, rel=1e-3)

    # For intermediate d: priority covers large risks
    r_mid = reinsurance_adjustment_coefficient(
        claim_rate=10.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_param=5.0,
        treaty_type="excess_of_loss",
    )
    assert float(np.asarray(r_mid).item()) > 0.0

    # Very small d where net drift is negative -> R = 0.0
    # theta = 0.10, xi = 0.90, d = 0.01 -> massive ceded claims with expensive loading
    r_small_d = reinsurance_adjustment_coefficient(
        claim_rate=1.0,
        claim_mean=100.0,
        safety_loading=0.10,
        reinsurer_loading=0.90,
        retention_param=0.01,
        treaty_type="excess_of_loss",
    )
    assert np.asarray(r_small_d).item() == 0.0

    # Test removable singularity handling in _excess_of_loss_lundberg_equation at r = beta
    beta_val = 1.2
    priority_val = 3.0
    val_at_beta = _excess_of_loss_lundberg_equation(
        r=beta_val,
        beta=beta_val,
        priority=priority_val,
        claim_rate=2.0,
        reinsured_premium_rate=2.5,
    )
    # At r = beta, mgf = 1 + beta * priority, so lambda * beta * priority - c_ret * beta
    expected_at_beta = 2.0 * beta_val * priority_val - 2.5 * beta_val
    assert val_at_beta == pytest.approx(expected_at_beta)


# ============================================================================
# Tests for optimal_reinsurance_retention
# ============================================================================


def test_optimal_reinsurance_retention_proportional() -> None:
    # When xi > theta (e.g. theta = 0.20, xi = 0.35), find optimal alpha
    opt = optimal_reinsurance_retention(
        claim_rate=5.0,
        claim_mean=10.0,
        safety_loading=0.20,
        reinsurer_loading=0.35,
        treaty_type="proportional",
    )
    alpha_star = opt["optimal_retention"]
    assert opt["minimum_viable_retention"] < alpha_star <= 1.0
    assert opt["optimal_adjustment_coefficient"] > 0.0
    assert opt["retained_safety_loading"] > 0.0


def test_optimal_reinsurance_retention_excess_of_loss() -> None:
    opt = optimal_reinsurance_retention(
        claim_rate=5.0,
        claim_mean=50.0,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        treaty_type="excess_of_loss",
    )
    d_star = opt["optimal_retention"]
    assert d_star > 0.0
    assert opt["optimal_adjustment_coefficient"] > 0.0


# ============================================================================
# Tests for reinsurance_ruin_bound
# ============================================================================


def test_reinsurance_ruin_bound_exponential() -> None:
    # Capital u = 50, mean = 1, theta = 0.20, xi = 0.25, alpha = 0.80
    res = reinsurance_ruin_bound(
        initial_capital=50.0,
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_rate=0.80,
    )
    assert 0.0 <= float(res["reinsured_ruin_probability"]) <= 1.0
    assert 0.0 <= float(res["gross_ruin_probability"]) <= 1.0

    # Boundary condition u = 0 yields ruin probability 1.0
    res_u0 = reinsurance_ruin_bound(
        initial_capital=0.0,
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_rate=0.80,
    )
    assert np.asarray(res_u0["reinsured_ruin_probability"]).item() == 1.0
    assert np.asarray(res_u0["gross_ruin_probability"]).item() == 1.0


def test_reinsurance_ruin_bound_methods() -> None:
    # Check that Lundberg upper bound is >= Cramér-Lundberg exact probability
    res_exact = reinsurance_ruin_bound(
        initial_capital=20.0,
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_rate=0.85,
        method="cramer_lundberg_exact",
    )
    res_bound = reinsurance_ruin_bound(
        initial_capital=20.0,
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_rate=0.85,
        method="lundberg_upper_bound",
    )
    assert float(res_bound["reinsured_ruin_probability"]) >= float(
        res_exact["reinsured_ruin_probability"]
    )


# ============================================================================
# Tests for reinsurance_capital_relief
# ============================================================================


def test_reinsurance_capital_relief_metrics() -> None:
    # Ruin tolerance eps = 0.01 (1%)
    relief = reinsurance_capital_relief(
        ruin_tolerance=0.01,
        claim_rate=1.0,
        claim_mean=100.0,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        retention_rate=0.80,
    )
    assert "gross_required_capital" in relief
    assert "reinsured_required_capital" in relief
    assert "capital_relief" in relief
    assert "percentage_capital_relief" in relief

    u_gross = float(relief["gross_required_capital"])
    u_re = float(relief["reinsured_required_capital"])
    diff = float(relief["capital_relief"])
    assert diff == pytest.approx(u_gross - u_re)


# ============================================================================
# Tests for compare_de_finetti_lundberg_retention
# ============================================================================


def test_compare_de_finetti_lundberg_retention() -> None:
    comp = compare_de_finetti_lundberg_retention(
        claim_mean=10.0,
        claim_variance=100.0,
        safety_loading=0.20,
        reinsurer_loading=0.35,
    )
    assert "de_finetti_optimal_retention" in comp
    assert "lundberg_optimal_retention" in comp
    assert "retention_difference" in comp
    assert "commentary" in comp
    assert 0.0 < comp["de_finetti_optimal_retention"] <= 1.0
    assert 0.0 < comp["lundberg_optimal_retention"] <= 1.0


# ============================================================================
# Non-finite and domain error tests
# ============================================================================


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_collective_reinsurance_non_finite_rejection(bad_val: float) -> None:
    """Verify that all public functions reject NaN and Inf."""
    # 1. proportional_reinsurance_process
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_process(bad_val, 2.0, 4.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_process(100.0, bad_val, 4.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_process(100.0, 2.0, bad_val, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_process(100.0, 2.0, 4.0, bad_val, 0.30)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_process(100.0, 2.0, 4.0, 0.20, bad_val)
    with pytest.raises(ValueError, match="finite"):
        proportional_reinsurance_process(100.0, 2.0, 4.0, 0.20, 0.30, retention_rate=bad_val)

    # 2. excess_of_loss_reinsurance_process
    with pytest.raises(ValueError, match="finite"):
        excess_of_loss_reinsurance_process(bad_val, 100.0, 150.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        excess_of_loss_reinsurance_process(10.0, bad_val, 150.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        excess_of_loss_reinsurance_process(10.0, 100.0, bad_val, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        excess_of_loss_reinsurance_process(10.0, 100.0, 150.0, bad_val, 0.30)
    with pytest.raises(ValueError, match="finite"):
        excess_of_loss_reinsurance_process(10.0, 100.0, 150.0, 0.20, bad_val)

    # 3. reinsurance_adjustment_coefficient
    with pytest.raises(ValueError, match="finite"):
        reinsurance_adjustment_coefficient(bad_val, 1.0, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_adjustment_coefficient(1.0, bad_val, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_adjustment_coefficient(1.0, 1.0, bad_val, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_adjustment_coefficient(1.0, 1.0, 0.20, bad_val, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_adjustment_coefficient(1.0, 1.0, 0.20, 0.30, bad_val)

    # 4. optimal_reinsurance_retention
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance_retention(bad_val, 10.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance_retention(5.0, bad_val, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance_retention(5.0, 10.0, bad_val, 0.30)
    with pytest.raises(ValueError, match="finite"):
        optimal_reinsurance_retention(5.0, 10.0, 0.20, bad_val)

    # 5. reinsurance_ruin_bound
    with pytest.raises(ValueError, match="finite"):
        reinsurance_ruin_bound(bad_val, 1.0, 1.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_ruin_bound(10.0, bad_val, 1.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_ruin_bound(10.0, 1.0, bad_val, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_ruin_bound(10.0, 1.0, 1.0, bad_val, 0.30)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_ruin_bound(10.0, 1.0, 1.0, 0.20, bad_val)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_ruin_bound(10.0, 1.0, 1.0, 0.20, 0.30, retention_rate=bad_val)

    # 6. reinsurance_capital_relief
    with pytest.raises(ValueError, match="finite"):
        reinsurance_capital_relief(bad_val, 1.0, 100.0, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_capital_relief(0.01, bad_val, 100.0, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_capital_relief(0.01, 1.0, bad_val, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, bad_val, 0.30, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, 0.20, bad_val, 0.80)
    with pytest.raises(ValueError, match="finite"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, 0.20, 0.30, bad_val)

    # 7. compare_de_finetti_lundberg_retention
    with pytest.raises(ValueError, match="finite"):
        compare_de_finetti_lundberg_retention(bad_val, 100.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        compare_de_finetti_lundberg_retention(10.0, bad_val, 0.20, 0.30)
    with pytest.raises(ValueError, match="finite"):
        compare_de_finetti_lundberg_retention(10.0, 100.0, bad_val, 0.30)
    with pytest.raises(ValueError, match="finite"):
        compare_de_finetti_lundberg_retention(10.0, 100.0, 0.20, bad_val)
    with pytest.raises(ValueError, match="finite"):
        compare_de_finetti_lundberg_retention(10.0, 100.0, 0.20, 0.30, claim_rate=bad_val)


def test_collective_reinsurance_domain_errors() -> None:
    # 1. proportional_reinsurance_process domain checks
    with pytest.raises(ValueError, match="claim_rate"):
        proportional_reinsurance_process(-10.0, 2.0, 4.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="claim_mean"):
        proportional_reinsurance_process(10.0, -2.0, 4.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="claim_variance"):
        proportional_reinsurance_process(10.0, 2.0, -4.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="safety_loading"):
        proportional_reinsurance_process(10.0, 2.0, 4.0, -0.20, 0.30)
    with pytest.raises(ValueError, match="reinsurer_loading"):
        proportional_reinsurance_process(10.0, 2.0, 4.0, 0.20, -0.30)
    with pytest.raises(ValueError, match="retention_rate"):
        proportional_reinsurance_process(10.0, 2.0, 4.0, 0.20, 0.30, retention_rate=1.5)
    with pytest.raises(ValueError, match="retention_rate"):
        proportional_reinsurance_process(10.0, 2.0, 4.0, 0.20, 0.30, retention_rate=-0.1)

    # 2. excess_of_loss_reinsurance_process domain checks
    with pytest.raises(ValueError, match="claim_rate"):
        excess_of_loss_reinsurance_process(-10.0, 100.0, 150.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="claim_mean"):
        excess_of_loss_reinsurance_process(10.0, -100.0, 150.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="priority"):
        excess_of_loss_reinsurance_process(10.0, 100.0, -50.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="safety_loading"):
        excess_of_loss_reinsurance_process(10.0, 100.0, 150.0, -0.20, 0.30)
    with pytest.raises(ValueError, match="reinsurer_loading"):
        excess_of_loss_reinsurance_process(10.0, 100.0, 150.0, 0.20, -0.30)
    with pytest.raises(ValueError, match="Unsupported distribution"):
        excess_of_loss_reinsurance_process(10.0, 100.0, 50.0, 0.20, 0.30, distribution="gamma")  # type: ignore[arg-type]

    # 3. reinsurance_adjustment_coefficient domain checks
    with pytest.raises(ValueError, match="claim_rate"):
        reinsurance_adjustment_coefficient(-1.0, 1.0, 0.20, 0.30, 0.8)
    with pytest.raises(ValueError, match="claim_mean"):
        reinsurance_adjustment_coefficient(1.0, -1.0, 0.20, 0.30, 0.8)
    with pytest.raises(ValueError, match="safety_loading"):
        reinsurance_adjustment_coefficient(1.0, 1.0, -0.20, 0.30, 0.8)
    with pytest.raises(ValueError, match="reinsurer_loading"):
        reinsurance_adjustment_coefficient(1.0, 1.0, 0.20, -0.30, 0.8)
    with pytest.raises(ValueError, match="Unsupported distribution"):
        reinsurance_adjustment_coefficient(1.0, 1.0, 0.20, 0.30, 0.8, distribution="pareto")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="treaty_type"):
        reinsurance_adjustment_coefficient(1.0, 1.0, 0.20, 0.30, 0.8, treaty_type="stop_loss")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        reinsurance_adjustment_coefficient(1.0, 1.0, 0.20, 0.30, -0.1, treaty_type="proportional")
    with pytest.raises(ValueError, match="priority"):
        reinsurance_adjustment_coefficient(
            1.0, 1.0, 0.20, 0.30, -10.0, treaty_type="excess_of_loss"
        )

    # 4. optimal_reinsurance_retention domain checks
    with pytest.raises(ValueError, match="strictly positive"):
        optimal_reinsurance_retention(-1.0, 10.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="reinsurer_loading"):
        optimal_reinsurance_retention(1.0, 10.0, 0.20, -0.30)
    with pytest.raises(ValueError, match="treaty_type"):
        optimal_reinsurance_retention(1.0, 10.0, 0.20, 0.30, treaty_type="unknown")  # type: ignore[arg-type]

    # 5. reinsurance_ruin_bound domain checks
    with pytest.raises(ValueError, match="initial_capital"):
        reinsurance_ruin_bound(-10.0, 1.0, 1.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="claim_rate"):
        reinsurance_ruin_bound(10.0, -1.0, 1.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="claim_mean"):
        reinsurance_ruin_bound(10.0, 1.0, -1.0, 0.20, 0.30)
    with pytest.raises(ValueError, match="safety_loading"):
        reinsurance_ruin_bound(10.0, 1.0, 1.0, -0.20, 0.30)
    with pytest.raises(ValueError, match="reinsurer_loading"):
        reinsurance_ruin_bound(10.0, 1.0, 1.0, 0.20, -0.30)
    with pytest.raises(ValueError, match="retention_rate"):
        reinsurance_ruin_bound(10.0, 1.0, 1.0, 0.20, 0.30, retention_rate=1.2)

    # 6. reinsurance_capital_relief domain checks
    with pytest.raises(ValueError, match="ruin_tolerance"):
        reinsurance_capital_relief(1.5, 1.0, 100.0, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="ruin_tolerance"):
        reinsurance_capital_relief(-0.05, 1.0, 100.0, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="strictly positive"):
        reinsurance_capital_relief(0.01, -1.0, 100.0, 0.20, 0.30, 0.80)
    with pytest.raises(ValueError, match="reinsurer_loading"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, 0.20, -0.30, 0.80)
    with pytest.raises(ValueError, match="retention_rate"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, 0.20, 0.30, -0.10)
    with pytest.raises(ValueError, match="retention_rate"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, 0.20, 0.30, 1.50)
    with pytest.raises(ValueError, match="net profit condition"):
        reinsurance_capital_relief(0.01, 1.0, 100.0, 0.20, 0.40, 0.10)

    # 7. compare_de_finetti_lundberg_retention domain checks
    with pytest.raises(ValueError, match="strictly positive"):
        compare_de_finetti_lundberg_retention(-10.0, 100.0, 0.20, 0.35)
    with pytest.raises(ValueError, match="strictly positive"):
        compare_de_finetti_lundberg_retention(10.0, -100.0, 0.20, 0.35)
    with pytest.raises(ValueError, match="strictly positive"):
        compare_de_finetti_lundberg_retention(10.0, 100.0, -0.20, 0.35)
    with pytest.raises(ValueError, match="strictly positive"):
        compare_de_finetti_lundberg_retention(10.0, 100.0, 0.20, -0.35)
    with pytest.raises(ValueError, match="strictly positive"):
        compare_de_finetti_lundberg_retention(10.0, 100.0, 0.20, 0.35, claim_rate=-1.0)


# ============================================================================
# Tests verifying exact root substitution into original Lundberg equation
# ============================================================================


def test_lundberg_equation_residues_proportional() -> None:
    """Verify that root R yields residue zero in the original Lundberg equation."""
    test_cases = [
        # (claim_rate, claim_mean, safety_loading, reinsurer_loading, alpha)
        (1.0, 1.0, 0.20, 0.30, 0.70),  # Prof. Pitta's observation case
        (5.0, 2.0, 0.15, 0.25, 0.60),
        (10.0, 50.0, 0.30, 0.40, 0.85),
        (2.0, 0.5, 0.10, 0.12, 0.95),
    ]

    for lam, mu1, theta, xi, alpha in test_cases:
        r_val = float(
            np.asarray(
                reinsurance_adjustment_coefficient(
                    claim_rate=lam,
                    claim_mean=mu1,
                    safety_loading=theta,
                    reinsurer_loading=xi,
                    retention_param=alpha,
                    treaty_type="proportional",
                )
            ).item()
        )
        assert r_val > 0.0

        # Lundberg fundamental equation: lambda * (m_Xret(R) - 1) - c_ret * R = 0
        beta = 1.0 / mu1
        c_ret = (1.0 + theta) * lam * mu1 - (1.0 + xi) * lam * (1.0 - alpha) * mu1
        # X_ret = alpha * X ~ Exp(beta / alpha)
        mgf = beta / (beta - alpha * r_val)
        residue = lam * (mgf - 1.0) - c_ret * r_val
        assert residue == pytest.approx(0.0, abs=1e-12)


def test_lundberg_equation_residues_excess_of_loss() -> None:
    """Verify that numerical XL root R yields residue zero in Lundberg equation."""
    test_cases = [
        # (claim_rate, claim_mean, safety_loading, reinsurer_loading, priority)
        (10.0, 1.0, 0.20, 0.25, 5.0),
        (5.0, 2.0, 0.25, 0.30, 3.0),
        (2.0, 10.0, 0.15, 0.20, 25.0),
    ]

    for lam, mu1, theta, xi, priority in test_cases:
        r_val = float(
            np.asarray(
                reinsurance_adjustment_coefficient(
                    claim_rate=lam,
                    claim_mean=mu1,
                    safety_loading=theta,
                    reinsurer_loading=xi,
                    retention_param=priority,
                    treaty_type="excess_of_loss",
                )
            ).item()
        )
        assert r_val > 0.0

        beta = 1.0 / mu1
        mu1_ced = mu1 * np.exp(-beta * priority)
        c_ret = (1.0 + theta) * lam * mu1 - (1.0 + xi) * lam * mu1_ced
        residue = _excess_of_loss_lundberg_equation(r_val, beta, priority, lam, c_ret)
        assert residue == pytest.approx(0.0, abs=1e-10)


def test_optimal_reinsurance_retention_narrow_domain() -> None:
    """Verify optimal retention when alpha_min is very close to 1.0 (Prof. Pitta's test)."""
    # Prof. Pitta's observation case: optimal_reinsurance_retention(1, 1, 0.00005, 1)
    # alpha_min = 1 - 0.00005 / 1.0 = 0.99995. Admissible interval is (0.99995, 1.0]
    opt = optimal_reinsurance_retention(
        claim_rate=1.0,
        claim_mean=1.0,
        safety_loading=0.00005,
        reinsurer_loading=1.0,
        treaty_type="proportional",
    )
    alpha_star = opt["optimal_retention"]
    assert opt["minimum_viable_retention"] < alpha_star <= 1.0
    assert opt["optimal_adjustment_coefficient"] > 0.0
    assert opt["retained_safety_loading"] > 0.0

    # Test compare_de_finetti_lundberg_retention with the same narrow domain
    comp = compare_de_finetti_lundberg_retention(
        claim_mean=1.0,
        claim_variance=1.0,
        safety_loading=0.00005,
        reinsurer_loading=1.0,
        claim_rate=1.0,
    )
    assert opt["minimum_viable_retention"] < comp["de_finetti_optimal_retention"] <= 1.0
    assert opt["minimum_viable_retention"] < comp["lundberg_optimal_retention"] <= 1.0
