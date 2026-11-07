"""Solvency and guarantee capital models for TheRiskTheory.

Topic 1: Solvency and Guarantee Capital.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np


def guarantee_capital(
    initial_capital: float | np.ndarray,
    maximum_aggregate_claim: float | np.ndarray,
    safety_margin: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Calculate the required guarantee capital for insurer solvency.

    Mathematical Definition:
        The guarantee capital (or solvency capital requirement) U_g is defined as:
        U_g = max(0, M - u) * (1 + m)
        where:
            - M is the maximum aggregate claim expected (or quantile-based peak loss).
            - u is the initial reserve or capital level.
            - m is a proportional safety margin (expressed as a decimal fraction, e.g., 0.1 for 10%).

    Args:
        initial_capital: Initial reserve or capital level (u). Must be non-negative.
        maximum_aggregate_claim: Peak or maximum aggregate claim expected (M). Must be non-negative.
        safety_margin: Proportional safety margin factor (m) applied as a multiplier.

    Returns:
        np.ndarray: Calculated guarantee capital level.

    Raises:
        ValueError: If inputs contain negative values, NaNs, or infinities.
    """
    u_arr = np.asarray(initial_capital, dtype=np.float64)
    max_claim = np.asarray(maximum_aggregate_claim, dtype=np.float64)
    margin = np.asarray(safety_margin, dtype=np.float64)

    # Validations for NaN, infinite, or negative values
    if (
        np.any(np.isnan(u_arr))
        or np.any(np.isnan(max_claim))
        or np.any(np.isnan(margin))
    ):
        raise ValueError("Inputs cannot contain NaN (Not a Number) values.")
    if (
        np.any(np.isinf(u_arr))
        or np.any(np.isinf(max_claim))
        or np.any(np.isinf(margin))
    ):
        raise ValueError("Inputs cannot contain infinite values.")
    if np.any(u_arr < 0.0) or np.any(max_claim < 0.0) or np.any(margin < -1.0):
        raise ValueError(
            "Capital and claims must be non-negative; safety margin must be > -1."
        )

    capital = np.maximum(0.0, max_claim - u_arr) * (1.0 + margin)
    return np.asarray(capital, dtype=np.float64)


def solvency_margin(
    premium_income: float | np.ndarray,
    claim_payments: float | np.ndarray,
    solvency_coefficient: float | np.ndarray,
) -> np.ndarray:
    """Calculate the statutory or risk-based solvency margin based on specific parameters.

    Mathematical Definition:
        The solvency margin M_s is derived from a regulatory or portfolio reference base:
        M_s = max(P, C) * k
        where:
            - P is the total premium income.
            - C is the total incurred claim payments.
            - k is the risk or solvency coefficient defined by the specific model or framework.

    Args:
        premium_income: Total collected premiums (P). Must be non-negative.
        claim_payments: Total incurred claims (C). Must be non-negative.
        solvency_coefficient: Coefficient factor (k) reflecting portfolio risk parameters.

    Returns:
        np.ndarray: Required solvency margin.

    Raises:
        ValueError: If inputs contain negative values, NaNs, or infinities.
    """
    prem_arr = np.asarray(premium_income, dtype=np.float64)
    claim_arr = np.asarray(claim_payments, dtype=np.float64)
    coeff = np.asarray(solvency_coefficient, dtype=np.float64)

    # Validations for NaN, infinite, or negative values
    if (
        np.any(np.isnan(prem_arr))
        or np.any(np.isnan(claim_arr))
        or np.any(np.isnan(coeff))
    ):
        raise ValueError("Inputs cannot contain NaN values.")
    if (
        np.any(np.isinf(prem_arr))
        or np.any(np.isinf(claim_arr))
        or np.any(np.isinf(coeff))
    ):
        raise ValueError("Inputs cannot contain infinite values.")
    if np.any(prem_arr < 0.0) or np.any(claim_arr < 0.0) or np.any(coeff < 0.0):
        raise ValueError(
            "Premiums, claims, and solvency coefficients must be non-negative."
        )

    base = np.maximum(prem_arr, claim_arr)
    margin = base * coeff
    return np.asarray(margin, dtype=np.float64)
