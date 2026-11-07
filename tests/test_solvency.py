import numpy as np
import pytest

from therisktheory.solvency import guarantee_capital, solvency_ratio


def test_guarantee_capital_analytical():
    """Tests exact analytical evaluation with specific initial conditions."""
    # Case: Initial Capital = 80, Max Loss = 100, Safety Margin = 0.10 (10%)
    # Target = 100 * 1.10 = 110 -> Required Additional Capital = 110 - 80 = 30
    res = guarantee_capital(
        initial_capital=80.0, max_loss=100.0, safety_margin=0.10
    )
    assert res == pytest.approx(30.0)


def test_guarantee_capital_sufficient_reserves():
    """Tests that no additional capital is needed if reserves exceed target."""
    res = guarantee_capital(
        initial_capital=120.0, max_loss=100.0, safety_margin=0.10
    )
    assert res == pytest.approx(0.0)


def test_solvency_ratio_analytical():
    """Tests non-dimensional ratio calculation."""
    res = solvency_ratio(available_capital=150.0, required_capital=100.0)
    assert res == pytest.approx(1.5)


def test_vectorized_inputs():
    """Tests functions with NumPy arrays."""
    capitals = np.array([50.0, 80.0, 120.0])
    losses = np.array([100.0, 100.0, 100.0])

    ratios = solvency_ratio(capitals, losses)
    np.testing.assert_allclose(ratios, np.array([0.5, 0.8, 1.2]))

    extra_cap = guarantee_capital(capitals, losses, safety_margin=0.10)
    np.testing.assert_allclose(extra_cap, np.array([60.0, 30.0, 0.0]))


def test_invalid_inputs_exceptions():
    """Tests error handling for out-of-domain inputs."""
    with pytest.raises(ValueError, match="non-negative"):
        guarantee_capital(initial_capital=-10.0, max_loss=100.0)

    with pytest.raises(ValueError, match="greater than zero"):
        solvency_ratio(available_capital=100.0, required_capital=0.0)
