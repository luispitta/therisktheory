"""Solvency and guarantee capital functions for TheRiskTheory."""

import numpy as np

from therisktheory.solvency_pricing.premiums import (
    expected_aggregate_loss,
    expected_value_premium,
)


def guarantee_capital(
    frequency: float | np.ndarray,
    mean_severity: float | np.ndarray,
    loading: float,
    solvency_factor: float,
) -> np.ndarray:
    """Calculate required guarantee capital based on expected losses and safety loadings.

    Args:
        frequency: Expected claim count (E[N]).
        mean_severity: Expected claim size (E[X]).
        loading: Safety loading coefficient (theta).
        solvency_factor: Risk margin factor for capital requirement.

    Returns:
        np.ndarray: Required guarantee capital.
    """
    exp_loss = expected_aggregate_loss(frequency, mean_severity)
    premium = expected_value_premium(exp_loss, loading)
    capital = solvency_factor * premium
    return np.asarray(capital, dtype=np.float64)


def solvency_ratio(
    capital: float | np.ndarray, premium_income: float | np.ndarray
) -> np.ndarray:
    """Compute the insurer's solvency ratio (Capital / Premium Income).

    Args:
        capital: Available capital.
        premium_income: Total premium income.

    Returns:
        np.ndarray: Solvency ratio.
    """
    cap_arr = np.asarray(capital, dtype=np.float64)
    prem_arr = np.asarray(premium_income, dtype=np.float64)

    if np.any(prem_arr <= 0):
        raise ValueError("Premium income must be strictly positive.")

    return cap_arr / prem_arr
