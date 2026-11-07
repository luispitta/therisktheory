"""Solvency and guarantee capital models for non-life insurance portfolios.

This module provides actuarial functions to evaluate insurer solvency, calculate
required monetary guarantee capital, and determine non-dimensional solvency ratios.
"""

import numpy as np


def guarantee_capital(
    initial_capital: float | np.ndarray,
    max_loss: float | np.ndarray,
    safety_margin: float = 0.10,
) -> float | np.ndarray:
    """Calculates the monetary guarantee (additional capital) required for solvency.

    References:
        - Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory.
        - Daykin, C. D., Pentikäinen, T., & Pesonen, E. (1994). Practical Risk Theory for Actuaries.
    """
    initial_cap_arr = np.asarray(initial_capital, dtype=float)
    max_loss_arr = np.asarray(max_loss, dtype=float)

    if np.any(initial_cap_arr < 0) or np.any(max_loss_arr < 0):
        raise ValueError("Initial capital and max loss must be non-negative.")
    if safety_margin < 0:
        raise ValueError("Safety margin rate must be non-negative.")

    required_target = max_loss_arr * (1.0 + safety_margin)
    additional_capital = np.maximum(0.0, required_target - initial_cap_arr)

    if np.ndim(initial_capital) == 0 and np.ndim(max_loss) == 0:
        return float(additional_capital)
    return additional_capital


def solvency_ratio(
    available_capital: float | np.ndarray,
    required_capital: float | np.ndarray,
) -> float | np.ndarray:
    """Calculates the solvency ratio (adimensional quotient).

    References:
        - Daykin, C. D., Pentikäinen, T., & Pesonen, E. (1994). Practical Risk Theory for Actuaries.
    """
    avail_arr = np.asarray(available_capital, dtype=float)
    req_arr = np.asarray(required_capital, dtype=float)

    if np.any(avail_arr < 0):
        raise ValueError("Available capital cannot be negative.")
    if np.any(req_arr <= 0):
        raise ValueError("Required capital must be strictly greater than zero.")

    ratio = avail_arr / req_arr

    if np.ndim(available_capital) == 0 and np.ndim(required_capital) == 0:
        return float(ratio)
    return ratio


def evaluate_portfolio_solvency(
    initial_capital: float,
    expected_frequency: float,
    average_severity: float,
    safety_loading: float,
    max_loss: float,
    safety_margin: float = 0.10,
) -> dict:
    """Evaluates the complete solvency position of an insurance portfolio."""
    # Cálculo local directo para evitar ciclos de importación
    p_pure = expected_frequency * average_severity
    p_commercial = p_pure * (1.0 + safety_loading)
    total_available = initial_capital + p_commercial

    ratio = solvency_ratio(total_available, max_loss)
    guarantee_cap = guarantee_capital(initial_capital, max_loss, safety_margin)

    return {
        "pure_premium": float(p_pure),
        "loaded_premium": float(p_commercial),
        "total_available_capital": float(total_available),
        "solvency_ratio": float(ratio),
        "guarantee_capital_required": float(guarantee_cap),
    }
