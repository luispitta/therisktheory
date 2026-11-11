"""Tests for De Finetti reserve funds and premium models."""

import numpy as np
import pytest

from therisktheory.de_finetti.reserves import (
    de_finetti_premium,
    de_finetti_profit,
    reserve_fund,
)


def test_de_finetti_premium():
    """Test the expected value premium calculation."""
    result = de_finetti_premium(200.0, 0.15)
    np.testing.assert_allclose(result, 230.0)


def test_de_finetti_premium_vectorization():
    """Test premium calculation with array inputs."""
    result = de_finetti_premium(
        np.array([100.0, 200.0]),
        0.10,
    )
    np.testing.assert_allclose(result, [110.0, 220.0])


@pytest.mark.parametrize(
    ("severity", "loading"),
    [
        (-1.0, 0.10),
        (100.0, -0.10),
        (np.nan, 0.10),
        (100.0, np.inf),
    ],
)
def test_de_finetti_premium_rejects_invalid_inputs(severity, loading):
    """Reject negative or non-finite premium parameters."""
    with pytest.raises((TypeError, ValueError)):
        de_finetti_premium(severity, loading)


def test_de_finetti_premium_rejects_incompatible_shapes():
    """Reject arrays that cannot be broadcast together."""
    with pytest.raises(ValueError):
        de_finetti_premium(
            np.array([100.0, 200.0]),
            np.array([0.10, 0.20, 0.30]),
        )


def test_de_finetti_premium_detects_overflow():
    """Reject a premium calculation that exceeds float64 range."""
    with pytest.raises(OverflowError):
        de_finetti_premium(np.finfo(np.float64).max, 1.0)


def test_de_finetti_profit():
    """Test periodic profit, including a negative result."""
    premiums = np.array([100.0, 100.0])
    claims = np.array([60.0, 120.0])

    result = de_finetti_profit(premiums, claims)

    np.testing.assert_allclose(result, [40.0, -20.0])


def test_de_finetti_profit_vectorization():
    """Test profit calculation with broadcastable inputs."""
    result = de_finetti_profit(
        np.array([100.0, 200.0]),
        50.0,
    )
    np.testing.assert_allclose(result, [50.0, 150.0])


@pytest.mark.parametrize(
    ("premiums", "claims"),
    [
        (np.array([-1.0]), np.array([2.0])),
        (np.array([1.0]), np.array([-2.0])),
        (np.array([np.nan]), np.array([2.0])),
        (np.array([1.0]), np.array([np.inf])),
    ],
)
def test_de_finetti_profit_rejects_invalid_inputs(premiums, claims):
    """Reject negative or non-finite premium and claim amounts."""
    with pytest.raises((TypeError, ValueError)):
        de_finetti_profit(premiums, claims)


def test_de_finetti_profit_rejects_incompatible_shapes():
    """Reject premium and claim arrays with incompatible shapes."""
    with pytest.raises(ValueError):
        de_finetti_profit(
            np.array([100.0, 200.0]),
            np.array([10.0, 20.0, 30.0]),
        )


def test_reserve_fund_trajectory():
    """Test the accumulated reserve after each period."""
    result = reserve_fund(
        100.0,
        np.array([50.0, 50.0, 50.0]),
        np.array([40.0, 60.0, 30.0]),
    )

    np.testing.assert_allclose(result, [110.0, 100.0, 120.0])


def test_reserve_fund_allows_negative_states():
    """Allow negative reserves to represent ruin states."""
    result = reserve_fund(
        50.0,
        np.array([0.0, 0.0, 0.0]),
        np.array([20.0, 40.0, 10.0]),
    )

    np.testing.assert_allclose(result, [30.0, -10.0, -20.0])


def test_reserve_fund_rejects_negative_initial_reserve():
    """Reject a negative initial reserve."""
    with pytest.raises(ValueError):
        reserve_fund(-10.0, np.array([20.0]), np.array([5.0]))


def test_reserve_fund_rejects_invalid_sequences():
    """Reject negative or non-finite premium and claim sequences."""
    with pytest.raises((TypeError, ValueError)):
        reserve_fund(
            100.0,
            np.array([20.0, np.nan]),
            np.array([5.0, 10.0]),
        )


def test_reserve_fund_rejects_incompatible_shapes():
    """Reject premium and claim sequences with different lengths."""
    with pytest.raises(ValueError):
        reserve_fund(
            100.0,
            np.array([20.0, 30.0]),
            np.array([5.0, 10.0, 15.0]),
        )


def test_reserve_fund_rejects_non_scalar_initial_reserve():
    """Reject an initial reserve that is not scalar."""
    with pytest.raises(ValueError):
        reserve_fund(
            np.array([100.0, 200.0]),
            np.array([20.0, 30.0]),
            np.array([5.0, 10.0]),
        )


@pytest.mark.parametrize(
    ("premiums", "claims"),
    [
        (np.array([[20.0, 30.0]]), np.array([[5.0, 10.0]])),
        (np.array([20.0, 30.0]), np.array([[5.0, 10.0]])),
        (np.array([[20.0, 30.0]]), np.array([5.0, 10.0])),
    ],
)
def test_reserve_fund_rejects_non_1d_sequences(premiums, claims):
    """Reject multidimensional premium and claim sequences."""
    with pytest.raises(ValueError):
        reserve_fund(100.0, premiums, claims)


def test_reserve_fund_accepts_empty_sequences():
    """Return an empty trajectory when no periods are supplied."""
    result = reserve_fund(
        100.0,
        np.array([], dtype=float),
        np.array([], dtype=float),
    )

    assert result.shape == (0,)


def test_reserve_fund_detects_overflow():
    """Reject a reserve trajectory that exceeds float64 range."""
    with pytest.raises(OverflowError):
        reserve_fund(
            np.finfo(np.float64).max,
            np.array([np.finfo(np.float64).max]),
            np.array([0.0]),
        )
