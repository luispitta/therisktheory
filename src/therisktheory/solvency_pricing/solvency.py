
"""Insurer solvency evaluation and guarantee capital calculations.

Topic 1: Solvency, premiums, retention, and reinsurance.

The capital formulas in this module are project conventions:
    Required capital = maximum loss * (1 + safety margin).
    Guarantee capital = max(0, required capital - initial capital).
    Solvency ratio = available capital / required capital.

The portfolio evaluation is a simplified one-period model. It assumes
that the monetary premium loading increases available capital. It does
not model realized claims, expenses, taxes, or investment returns.
"""

from __future__ import annotations

import numpy as np

from therisktheory import _validation
from therisktheory.solvency_pricing.loadings import (
    loaded_premium,
    safety_loading as calculate_safety_loading,
)
from therisktheory.solvency_pricing.premiums import pure_premium


def required_capital(
    max_loss: float | np.ndarray,
    safety_margin: float | np.ndarray = 0.05,
) -> np.ndarray:
    """Calculate the total capital required for a specified maximum loss.

    Formula:
        C_r = L_max * (1 + m)

    Inputs must be finite and non-negative. NumPy broadcasting applies.
    """
    loss = _validation.as_nonnegative_finite_array(max_loss, "max_loss")
    margin = _validation.as_nonnegative_finite_array(
        safety_margin, "safety_margin"
    )

    _validation.check_broadcastable(
        max_loss=loss,
        safety_margin=margin,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(loss * (1.0 + margin), dtype=np.float64)

    return _validation.check_finite_result(result, "required_capital")


def guarantee_capital(
    initial_capital: float | np.ndarray,
    max_loss: float | np.ndarray,
    safety_margin: float | np.ndarray = 0.05,
) -> np.ndarray:
    """Calculate the additional capital needed to meet the requirement.

    Formula:
        C_g = max(0, C_r - C_0)
    """
    initial = _validation.as_nonnegative_finite_array(
        initial_capital, "initial_capital"
    )
    required = required_capital(max_loss, safety_margin)

    _validation.check_broadcastable(
        initial_capital=initial,
        required_capital=required,
    )

    result = np.asarray(np.maximum(0.0, required - initial))

    return _validation.check_finite_result(result, "guarantee_capital")


def solvency_ratio(
    available_capital: float | np.ndarray,
    required_capital: float | np.ndarray,
) -> np.ndarray:
    """Calculate available capital divided by required capital.

    A zero required-capital denominator is rejected because the ratio
    is undefined in this implementation.
    """
    available = _validation.as_nonnegative_finite_array(
        available_capital, "available_capital"
    )
    required = _validation.as_nonnegative_finite_array(
        required_capital, "required_capital"
    )

    _validation.check_broadcastable(
        available_capital=available,
        required_capital=required,
    )

    if np.any(required == 0.0):
        raise ValueError("required_capital must be greater than zero")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = np.asarray(available / required, dtype=np.float64)

    return _validation.check_finite_result(result, "solvency_ratio")


def evaluate_portfolio_solvency(
    initial_capital: float | np.ndarray,
    expected_frequency: float | np.ndarray,
    average_severity: float | np.ndarray,
    safety_loading: float | np.ndarray = 0.15,
    max_loss: float | np.ndarray = 100.0,
    safety_margin: float | np.ndarray = 0.05,
) -> dict[str, np.ndarray]:
    """Evaluate premiums and solvency under a simplified one-period model.

    The expected aggregate loss is calculated using the pure-premium
    function. The expected-value premium principle determines the
    loaded premium. Its monetary loading is added to initial capital.

    The resulting available capital is compared with required capital.
    This is a simplified project model, not a regulatory solvency formula.
    """
    initial = _validation.as_nonnegative_finite_array(
        initial_capital, "initial_capital"
    )
    frequency = _validation.as_nonnegative_finite_array(
        expected_frequency, "expected_frequency"
    )
    severity = _validation.as_nonnegative_finite_array(
        average_severity, "average_severity"
    )
    rate = _validation.as_nonnegative_finite_array(
        safety_loading, "safety_loading"
    )

    _validation.check_broadcastable(
        initial_capital=initial,
        expected_frequency=frequency,
        average_severity=severity,
        safety_loading=rate,
    )

    expected_loss = pure_premium(frequency, severity)
    premium = loaded_premium(expected_loss, rate)

    monetary_loading = calculate_safety_loading(
        expected_loss,
        premium,
    )

    available = np.asarray(initial + monetary_loading)
    required = required_capital(max_loss, safety_margin)
    guarantee = guarantee_capital(
        initial,
        max_loss,
        safety_margin,
    )
    ratio = solvency_ratio(available, required)

    return {
        "pure_premium": expected_loss,
        "loaded_premium": premium,
        "safety_loading": monetary_loading,
        "total_available_capital": _validation.check_finite_result(
            available,
            "total_available_capital",
        ),
        "required_capital": required,
        "guarantee_capital": guarantee,
        "solvency_ratio": ratio,
    }
