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

    Args:
        initial_capital: Initial reserve or capital level (u).
        maximum_aggregate_claim: Peak or maximum aggregate claim expected.
        safety_margin: Additional safety buffer percentage or fixed amount.

    Returns:
        np.ndarray: Calculated guarantee capital level.
    """
    u_arr = np.asarray(initial_capital, dtype=np.float64)
    max_claim = np.asarray(maximum_aggregate_claim, dtype=np.float64)
    margin = np.asarray(safety_margin, dtype=np.float64)

    capital = np.maximum(0.0, max_claim - u_arr) * (1.0 + margin)
    return np.asarray(capital, dtype=np.float64)


def solvency_margin(
    premium_income: float | np.ndarray,
    claim_payments: float | np.ndarray,
    solvency_coefficient: float | np.ndarray = 0.2,
) -> np.ndarray:
    """Calculate the statutory or risk-based solvency margin.

    Args:
        premium_income: Total collected premiums.
        claim_payments: Total incurred claims.
        solvency_coefficient: Regulatory solvency coefficient.

    Returns:
        np.ndarray: Required solvency margin.
    """
    prem_arr = np.asarray(premium_income, dtype=np.float64)
    claim_arr = np.asarray(claim_payments, dtype=np.float64)
    coeff = np.asarray(solvency_coefficient, dtype=np.float64)

    base = np.maximum(prem_arr, claim_arr)
    margin = base * coeff
    return np.asarray(margin, dtype=np.float64)