"""Unit tests for reinsurance treaties and solvency evaluation functions."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.solvency_pricing import (
    excess_of_loss,
    quota_share,
    reinsured_solvent_position,
    stop_loss,
    surplus_share,
)

# ============================================================================
# Tests for quota_share
# ============================================================================


def test_quota_share_scalar() -> None:
    retained, ceded = quota_share(1000.0, 0.6)
    assert np.asarray(retained).item() == pytest.approx(600.0)
    assert np.asarray(ceded).item() == pytest.approx(400.0)


def test_quota_share_vectorized() -> None:
    losses = np.array([500.0, 1000.0, 2000.0])
    retained, ceded = quota_share(losses, 0.75)
    np.testing.assert_allclose(retained, np.array([375.0, 750.0, 1500.0]))
    np.testing.assert_allclose(ceded, np.array([125.0, 250.0, 500.0]))


def test_quota_share_broadcasting() -> None:
    losses = np.array([[100.0, 200.0], [300.0, 400.0]])
    retention = np.array([0.5, 0.8])
    retained, ceded = quota_share(losses, retention)
    expected_retained = np.array([[50.0, 160.0], [150.0, 320.0]])
    expected_ceded = np.array([[50.0, 40.0], [150.0, 80.0]])
    np.testing.assert_allclose(retained, expected_retained)
    np.testing.assert_allclose(ceded, expected_ceded)


def test_quota_share_conservation_property() -> None:
    rng = np.random.default_rng(42)
    losses = rng.uniform(10.0, 10000.0, size=50)
    rates = rng.uniform(0.0, 1.0, size=50)
    retained, ceded = quota_share(losses, rates)
    np.testing.assert_allclose(retained + ceded, losses)


def test_quota_share_boundaries() -> None:
    losses = np.array([1000.0])
    # Full retention
    ret_1, ced_1 = quota_share(losses, 1.0)
    assert ret_1.item() == 1000.0
    assert ced_1.item() == 0.0

    # Full cession
    ret_0, ced_0 = quota_share(losses, 0.0)
    assert ret_0.item() == 0.0
    assert ced_0.item() == 1000.0


@pytest.mark.parametrize(
    ("losses", "rate"),
    [
        (-100.0, 0.5),
        (100.0, -0.1),
        (100.0, 1.05),
    ],
)
def test_quota_share_invalid_inputs(losses: float, rate: float) -> None:
    with pytest.raises(ValueError):
        quota_share(losses, rate)


# ============================================================================
# Tests for surplus_share
# ============================================================================


def test_surplus_share_below_retention_line() -> None:
    # Sum insured <= line -> 100% retained
    retained, ceded = surplus_share(500.0, sum_insured=5000.0, retention_line=10000.0)
    assert np.asarray(retained).item() == pytest.approx(500.0)
    assert np.asarray(ceded).item() == pytest.approx(0.0)


def test_surplus_share_within_capacity() -> None:
    # Sum insured = 40000, Line = 10000, 3 lines capacity (max cession 30000)
    # Total capacity = 10000 + 30000 = 40000 -> 25% retained, 75% ceded
    retained, ceded = surplus_share(
        4000.0, sum_insured=40000.0, retention_line=10000.0, max_lines=3.0
    )
    assert np.asarray(retained).item() == pytest.approx(1000.0)
    assert np.asarray(ceded).item() == pytest.approx(3000.0)


def test_surplus_share_exceeding_capacity() -> None:
    # Sum insured = 60000, Line = 10000, 2 lines max cession (20000)
    # Ceded ratio = 20000 / 60000 = 1/3; Retained ratio = 2/3
    retained, ceded = surplus_share(
        6000.0, sum_insured=60000.0, retention_line=10000.0, max_lines=2.0
    )
    assert np.asarray(retained).item() == pytest.approx(4000.0)
    assert np.asarray(ceded).item() == pytest.approx(2000.0)


def test_surplus_share_conservation_property() -> None:
    losses = np.array([1000.0, 2500.0, 7000.0])
    sums_insured = np.array([10000.0, 25000.0, 100000.0])
    retained, ceded = surplus_share(losses, sums_insured, retention_line=10000.0, max_lines=4.0)
    np.testing.assert_allclose(retained + ceded, losses)


@pytest.mark.parametrize(
    ("loss", "si", "line", "m"),
    [
        (-10.0, 1000.0, 500.0, 2.0),
        (10.0, 0.0, 500.0, 2.0),
        (10.0, -100.0, 500.0, 2.0),
        (10.0, 1000.0, 0.0, 2.0),
        (10.0, 1000.0, -500.0, 2.0),
        (10.0, 1000.0, 500.0, -1.0),
    ],
)
def test_surplus_share_invalid_inputs(loss: float, si: float, line: float, m: float) -> None:
    with pytest.raises(ValueError):
        surplus_share(loss, si, line, m)


# ============================================================================
# Tests for excess_of_loss
# ============================================================================


def test_excess_of_loss_unlimited() -> None:
    claims = np.array([200.0, 500.0, 1200.0])
    retained, ceded = excess_of_loss(claims, priority=500.0)
    np.testing.assert_allclose(retained, np.array([200.0, 500.0, 500.0]))
    np.testing.assert_allclose(ceded, np.array([0.0, 0.0, 700.0]))


def test_excess_of_loss_with_limit() -> None:
    # Priority = 500, Limit = 1000 (cover is [500, 1500])
    claims = np.array([300.0, 800.0, 1500.0, 2200.0])
    retained, ceded = excess_of_loss(claims, priority=500.0, limit=1000.0)
    # Claim 300: ceded 0, ret 300
    # Claim 800: ceded 300, ret 500
    # Claim 1500: ceded 1000, ret 500
    # Claim 2200: ceded 1000, ret 1200 (500 + 700 above cover)
    np.testing.assert_allclose(ceded, np.array([0.0, 300.0, 1000.0, 1000.0]))
    np.testing.assert_allclose(retained, np.array([300.0, 500.0, 500.0, 1200.0]))
    np.testing.assert_allclose(retained + ceded, claims)


@pytest.mark.parametrize(
    ("claim", "priority", "limit"),
    [
        (-50.0, 100.0, None),
        (50.0, -100.0, None),
        (50.0, 100.0, 0.0),
        (50.0, 100.0, -10.0),
    ],
)
def test_excess_of_loss_invalid_inputs(claim: float, priority: float, limit: float | None) -> None:
    with pytest.raises(ValueError):
        excess_of_loss(claim, priority, limit)


# ============================================================================
# Tests for stop_loss
# ============================================================================


def test_stop_loss_unlimited() -> None:
    portfolios = np.array([30000.0, 50000.0, 85000.0])
    retained, ceded = stop_loss(portfolios, deductible=50000.0)
    np.testing.assert_allclose(retained, np.array([30000.0, 50000.0, 50000.0]))
    np.testing.assert_allclose(ceded, np.array([0.0, 0.0, 35000.0]))


def test_stop_loss_with_limit() -> None:
    portfolios = np.array([40000.0, 70000.0, 120000.0])
    retained, ceded = stop_loss(portfolios, deductible=50000.0, limit=40000.0)
    # Portfolio 40k: ceded 0, ret 40k
    # Portfolio 70k: ceded 20k, ret 50k
    # Portfolio 120k: ceded 40k, ret 80k (50k + 30k above limit)
    np.testing.assert_allclose(ceded, np.array([0.0, 20000.0, 40000.0]))
    np.testing.assert_allclose(retained, np.array([40000.0, 50000.0, 80000.0]))
    np.testing.assert_allclose(retained + ceded, portfolios)


@pytest.mark.parametrize(
    ("losses", "deductible", "limit"),
    [
        (-1000.0, 5000.0, None),
        (1000.0, -5000.0, None),
        (1000.0, 5000.0, 0.0),
        (1000.0, 5000.0, -100.0),
    ],
)
def test_stop_loss_invalid_inputs(losses: float, deductible: float, limit: float | None) -> None:
    with pytest.raises(ValueError):
        stop_loss(losses, deductible, limit)


# ============================================================================
# Tests for reinsured_solvent_position
# ============================================================================


def test_reinsured_solvent_position_scalar_solvent() -> None:
    pos = reinsured_solvent_position(
        initial_capital=500.0, retained_premium=300.0, retained_losses=600.0
    )
    assert pos["net_surplus"].item() == pytest.approx(200.0)
    assert pos["is_solvent"].item() is True
    assert pos["deficit"].item() == pytest.approx(0.0)


def test_reinsured_solvent_position_scalar_insolvent() -> None:
    pos = reinsured_solvent_position(
        initial_capital=500.0, retained_premium=300.0, retained_losses=1100.0
    )
    assert pos["net_surplus"].item() == pytest.approx(-300.0)
    assert pos["is_solvent"].item() is False
    assert pos["deficit"].item() == pytest.approx(300.0)


def test_reinsured_solvent_position_vectorized() -> None:
    pos = reinsured_solvent_position(
        initial_capital=1000.0,
        retained_premium=500.0,
        retained_losses=np.array([1200.0, 1500.0, 1800.0]),
    )
    # Surplus: 1000 + 500 - [1200, 1500, 1800] = [300, 0, -300]
    np.testing.assert_allclose(pos["net_surplus"], np.array([300.0, 0.0, -300.0]))
    np.testing.assert_array_equal(pos["is_solvent"], np.array([True, True, False]))
    np.testing.assert_allclose(pos["deficit"], np.array([0.0, 0.0, 300.0]))


@pytest.mark.parametrize(
    ("capital", "premium", "losses"),
    [
        (-100.0, 50.0, 100.0),
        (100.0, -50.0, 100.0),
        (100.0, 50.0, -100.0),
    ],
)
def test_reinsured_solvent_position_invalid_inputs(
    capital: float, premium: float, losses: float
) -> None:
    with pytest.raises(ValueError):
        reinsured_solvent_position(capital, premium, losses)


# ============================================================================
# Joint Integration Tests with Bracamonte's pure_premium
# ============================================================================


def test_joint_integration_pure_premium_and_quota_share() -> None:
    """Joint integration test: combining pure_premium (Bracamonte) with reinsurance (Valera).

    Validates the end-to-end pricing and risk transfer chain:
    frequency + severity -> pure premium -> quota share treaty -> solvent position.
    """
    from therisktheory.solvency_pricing import pure_premium

    # 1. Collective risk decomposition
    expected_frequency = np.array([10.0, 50.0, 100.0])
    mean_claim_size = np.array([500.0, 200.0, 1000.0])

    # Pure premium from Bracamonte's module: E[S] = E[N] * E[X]
    expected_aggregate_loss = pure_premium(expected_frequency, mean_claim_size)
    # Expected: [5000, 10000, 100000]
    np.testing.assert_allclose(expected_aggregate_loss, np.array([5000.0, 10000.0, 100000.0]))

    # 2. Risk transfer via quota share (Valera)
    retention_rate = 0.70
    retained_loss, ceded_loss = quota_share(expected_aggregate_loss, retention_rate=retention_rate)

    # Conservation of expected risk
    np.testing.assert_allclose(retained_loss + ceded_loss, expected_aggregate_loss)
    np.testing.assert_allclose(retained_loss, 0.70 * expected_aggregate_loss)
    np.testing.assert_allclose(ceded_loss, 0.30 * expected_aggregate_loss)

    # 3. Solvency evaluation
    # Insurer collects 15% loading, reinsurer charges 20% loading
    insurer_premium = 1.15 * expected_aggregate_loss
    reinsurance_cost = 0.30 * 1.20 * expected_aggregate_loss
    net_retained_premium = insurer_premium - reinsurance_cost

    solvency = reinsured_solvent_position(
        initial_capital=5000.0,
        retained_premium=net_retained_premium,
        retained_losses=retained_loss,
    )

    # Check terminal surplus: u + P_ret - S_ret
    expected_surplus = 5000.0 + net_retained_premium - retained_loss
    np.testing.assert_allclose(solvency["net_surplus"], expected_surplus)
    assert np.all(solvency["is_solvent"])


def test_joint_integration_pure_premium_and_stop_loss() -> None:
    """Joint integration test: pure_premium baseline combined with stop_loss treaty."""
    from therisktheory.solvency_pricing import pure_premium

    # Expected annual claims: 200 claims * $250 = $50,000 pure premium
    base_pure_prem = float(pure_premium(200.0, 250.0).item())
    assert base_pure_prem == pytest.approx(50000.0)

    # Realized loss scenarios around the pure premium
    realized_scenarios = np.array([35000.0, 50000.0, 65000.0, 100000.0])

    # Stop loss attaches at pure premium level (deductible = E[S]) with limit 30000
    ret_loss, ced_loss = stop_loss(realized_scenarios, deductible=base_pure_prem, limit=30000.0)

    # At $35k: ceded = 0, retained = 35k
    # At $50k: ceded = 0, retained = 50k
    # At $65k: ceded = 15k, retained = 50k
    # At $100k: ceded = 30k (capped), retained = 70k
    np.testing.assert_allclose(ced_loss, np.array([0.0, 0.0, 15000.0, 30000.0]))
    np.testing.assert_allclose(ret_loss, np.array([35000.0, 50000.0, 50000.0, 70000.0]))
    np.testing.assert_allclose(ret_loss + ced_loss, realized_scenarios)


@pytest.mark.parametrize(
    "invalid_val",
    [np.nan, np.inf, -np.inf],
)
def test_reinsurance_rejects_non_finite_inputs(invalid_val: float) -> None:
    """Verify that all reinsurance treaty functions reject NaN, +Inf, -Inf."""
    with pytest.raises(ValueError):
        quota_share(invalid_val, 0.5)
    with pytest.raises(ValueError):
        quota_share(100.0, invalid_val)

    with pytest.raises(ValueError):
        surplus_share(invalid_val, 1000.0, 500.0)
    with pytest.raises(ValueError):
        surplus_share(100.0, invalid_val, 500.0)

    with pytest.raises(ValueError):
        excess_of_loss(invalid_val, 100.0)
    with pytest.raises(ValueError):
        excess_of_loss(100.0, invalid_val)
    with pytest.raises(ValueError):
        excess_of_loss(100.0, 50.0, limit=invalid_val)

    with pytest.raises(ValueError):
        stop_loss(invalid_val, 100.0)
    with pytest.raises(ValueError):
        stop_loss(100.0, invalid_val)
    with pytest.raises(ValueError):
        stop_loss(100.0, 50.0, limit=invalid_val)

    with pytest.raises(ValueError):
        reinsured_solvent_position(invalid_val, 100.0, 50.0)
    with pytest.raises(ValueError):
        reinsured_solvent_position(100.0, invalid_val, 50.0)
    with pytest.raises(ValueError):
        reinsured_solvent_position(100.0, 100.0, invalid_val)
