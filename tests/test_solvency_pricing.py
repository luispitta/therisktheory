"""Tests and boundary conditions for solvency and ruin functions."""

import numpy as np
import pytest

from therisktheory.solvency_pricing.expected_profit import expected_profit
from therisktheory.solvency_pricing.finite_horizon_ruin_estimate import (
    finite_horizon_ruin_estimate,
)
from therisktheory.solvency_pricing.portfolio_equilibrium import (
    portfolio_equilibrium,
)


def test_expected_profit_mathematical_identity() -> None:
    """Verify safety loading identity: E[Profit] = theta * E[Claims]."""
    e_claims = 5000.0
    theta = 0.15
    premium = (1.0 + theta) * e_claims
    profit = expected_profit(premium, e_claims)
    assert np.isclose(profit, theta * e_claims)


def test_expected_profit_allows_negative_result() -> None:
    """Boundary test: Claims exceeding premium results in deficit."""
    premium = 1000.0
    claims = 1800.0
    profit = expected_profit(premium, claims)
    assert profit == -800.0
    assert profit < 0.0


def test_expected_profit_with_expenses() -> None:
    """Verify operational expenses deduction."""
    profit = expected_profit(
        premium=2000.0,
        expected_claims=1200.0,
        expenses=300.0,
    )
    assert profit == 500.0


def test_portfolio_equilibrium_negative_accumulated_profit() -> None:
    """Ensure accumulated technical losses are permitted."""
    u = 1500.0
    negative_profit = -400.0
    surplus = portfolio_equilibrium(
        initial_capital=u,
        accumulated_profit=negative_profit,
    )
    assert surplus == 1100.0


def test_portfolio_equilibrium_insolvency_balance() -> None:
    """Deficit test: Severe loss depletes capital below zero."""
    u = 500.0
    negative_profit = -800.0
    surplus = portfolio_equilibrium(
        initial_capital=u,
        accumulated_profit=negative_profit,
    )
    assert surplus == -300.0


def test_portfolio_equilibrium_no_double_deduction() -> None:
    """Verify net profit does not discount claims twice by default."""
    u = 2000.0
    net_profit = 350.0
    surplus = portfolio_equilibrium(
        initial_capital=u,
        accumulated_profit=net_profit,
    )
    assert surplus == 2350.0


def test_portfolio_equilibrium_break_even() -> None:
    """Boundary test: Zero initial capital and zero net profit."""
    surplus = portfolio_equilibrium(
        initial_capital=0.0,
        accumulated_profit=0.0,
    )
    assert surplus == 0.0


def test_portfolio_equilibrium_invalid_negative_capital() -> None:
    """Verify that negative initial capital raises ValueError."""
    with pytest.raises(ValueError):
        portfolio_equilibrium(
            initial_capital=-100.0,
            accumulated_profit=50.0,
        )


def test_finite_ruin_sensitivity_to_claim_intensity() -> None:
    """Verify ruin probability increases with higher frequency lambda."""
    u = 50.0
    c = 15.0
    mean_x = 1.0
    t = 10.0

    psi_low = finite_horizon_ruin_estimate(
        initial_capital=u,
        premium_rate=c,
        claim_intensity=5.0,
        mean_claim_size=mean_x,
        time_horizon=t,
    )
    psi_high = finite_horizon_ruin_estimate(
        initial_capital=u,
        premium_rate=c,
        claim_intensity=12.0,
        mean_claim_size=mean_x,
        time_horizon=t,
    )

    assert psi_high > psi_low


def test_finite_ruin_boundary_zero_capital() -> None:
    """Boundary case: u = 0 implies immediate ruin (psi = 1.0)."""
    psi = finite_horizon_ruin_estimate(
        initial_capital=0.0,
        premium_rate=10.0,
        claim_intensity=2.0,
        mean_claim_size=3.0,
        time_horizon=5.0,
    )
    assert np.isclose(psi, 1.0)


def test_finite_ruin_boundary_zero_time() -> None:
    """Boundary case: T = 0 with u > 0 implies psi = 0.0."""
    psi = finite_horizon_ruin_estimate(
        initial_capital=100.0,
        premium_rate=10.0,
        claim_intensity=2.0,
        mean_claim_size=3.0,
        time_horizon=0.0,
    )
    assert np.isclose(psi, 0.0)


def test_finite_ruin_vectorization() -> None:
    """Test multi-dimensional array support and monotonicity."""
    u_vec = np.array([10.0, 50.0, 100.0])
    psi = finite_horizon_ruin_estimate(
        initial_capital=u_vec,
        premium_rate=20.0,
        claim_intensity=2.0,
        mean_claim_size=5.0,
        time_horizon=5.0,
    )
    assert psi.shape == (3,)
    assert psi[0] >= psi[1] >= psi[2]
