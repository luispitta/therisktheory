"""Solvency and guarantee capital models for non-life insurance portfolios.

This module provides actuarial functions to evaluate insurer solvency, calculate
required monetary guarantee capital, and determine non-dimensional solvency ratios.
It integrates seamlessly with premium models provided by peer modules.
"""

from typing import Union
import numpy as np

# =====================================================================
# INTEGRACIÓN CON COMPAÑEROS (TEMA 1)
# Importa las funciones de Bracamonte (primas puras) y García (recargos).
# Si sus módulos aún se están fusionando, el sistema usará valores por defecto.
# =====================================================================
try:
    from therisktheory.premiums import pure_premium  # Bracamonte
    from therisktheory.loadings import loaded_premium  # García
except ImportError:
    # Definiciones de respaldo (fallbacks) para pruebas locales aisladas
    def pure_premium(expected_frequency: float, average_severity: float) -> float:
        return expected_frequency * average_severity

    def loaded_premium(pure_prem: float, safety_loading: float) -> float:
        return pure_prem * (1.0 + safety_loading)


def guarantee_capital(
    initial_capital: Union[float, np.ndarray],
    max_loss: Union[float, np.ndarray],
    safety_margin: float = 0.10,
) -> Union[float, np.ndarray]:
    """Calculates the monetary guarantee or additional capital required.

    Determines the extra capital needed to cover the expected maximum loss
    at a given confidence level, accounting for an explicit safety margin.

    Args:
        initial_capital (Union[float, np.ndarray]): Available initial reserves or capital.
        max_loss (Union[float, np.ndarray]): Maximum expected loss (e.g., VaR / quantile).
        safety_margin (float, optional): Safety margin rate as a decimal (e.g., 0.10 for 10%).
            Defaults to 0.10.

    Returns:
        Union[float, np.ndarray]: The monetary guarantee/additional capital required.

    Raises:
        ValueError: If any input contains negative values or invalid margins.

    References:
        Beard, R., Pentikäinen, T., & Pesonen, E. — Risk Theory.
        Daykin, C., Pentikäinen, T., & Pesonen, E. — Practical Risk Theory for Actuaries.
    """
    initial_cap_arr = np.asarray(initial_capital, dtype=float)
    max_loss_arr = np.asarray(max_loss, dtype=float)

    if np.any(initial_cap_arr < 0) or np.any(max_loss_arr < 0):
        raise ValueError("Initial capital and max loss must be non-negative.")
    if safety_margin < 0:
        raise ValueError("Safety margin rate must be non-negative.")

    # Target capital required to maintain solvency under peak exposure
    required_target = max_loss_arr * (1.0 + safety_margin)
    additional_capital = np.maximum(0.0, required_target - initial_cap_arr)

    if np.ndim(initial_capital) == 0 and np.ndim(max_loss) == 0:
        return float(additional_capital)
    return additional_capital


def solvency_ratio(
    available_capital: Union[float, np.ndarray],
    required_capital: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculates the solvency ratio (adimensional quotient).

    Evaluates the ratio of total available capital (including initial reserves and premiums)
    against the required solvency capital or maximum exposure.

    Args:
        available_capital (Union[float, np.ndarray]): Total available capital/assets.
        required_capital (Union[float, np.ndarray]): Required capital or loss exposure.

    Returns:
        Union[float, np.ndarray]: The solvency ratio (available / required).

    Raises:
        ValueError: If required capital is non-positive or available capital is negative.

    References:
        Daykin, C., Pentikäinen, T., & Pesonen, E. — Practical Risk Theory for Actuaries.
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
    """Evaluates the complete solvency position of an insurance portfolio by

    integrating Bracamonte's pure premium and Garcia's loaded premium.

    Args:
        initial_capital (float): Insurer's initial capital.
        expected_frequency (float): Expected number of claims (Bracamonte).
        average_severity (float): Average claim severity (Bracamonte).
        safety_loading (float): Safety loading factor (Garcia).
        max_loss (float): Maximum probable loss or quantile limit.
        safety_margin (float): Regulatory safety margin rate.

    Returns:
        dict: Dictionary containing intermediate premiums, available capital,
              solvency ratio, and required guarantee capital.
    """
    # 1. Obtener prima pura (Bracamonte)
    p_pure = pure_premium(expected_frequency, average_severity)

    # 2. Obtener prima comercial / recargada (García)
    p_commercial = loaded_premium(p_pure, safety_loading)

    # 3. Calcular capital total disponible (Capital Inicial + Primas)
    total_available = initial_capital + p_commercial

    # 4. Calcular métricas de solvencia (Tus funciones)
    ratio = solvency_ratio(total_available, max_loss)
    guarantee_cap = guarantee_capital(initial_capital, max_loss, safety_margin)

    return {
        "pure_premium": float(p_pure),
        "loaded_premium": float(p_commercial),
        "total_available_capital": float(total_available),
        "solvency_ratio": float(ratio),
        "guarantee_capital_required": float(guarantee_cap),
    }