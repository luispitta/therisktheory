"""Pure premium and premium principles for TheRiskTheory.

Topic 1: Solvency and Pricing.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np


def pure_premium(
    frequency: float | np.ndarray,
    mean_severity: float | np.ndarray,
) -> np.ndarray:
    """Calculate the pure premium (net premium) from frequency and severity.

    Args:
        frequency: Expected claim frequency (lambda).
        mean_severity: Expected claim severity (mu).

    Returns:
        np.ndarray: Calculated pure premium.
    """
    freq_arr = np.asarray(frequency, dtype=np.float64)
    sev_arr = np.asarray(mean_severity, dtype=np.float64)
    return np.asarray(freq_arr * sev_arr, dtype=np.float64)


def expected_value_premium(
    pure_premium_val: float | np.ndarray,
    loading: float | np.ndarray = 0.1,
) -> np.ndarray:
    """Calculate the premium using the expected value principle with safety loading.

    Args:
        pure_premium_val: Net or pure premium amount.
        loading: Safety loading factor (e.g., 0.1 for 10%).

    Returns:
        np.ndarray: Gross premium with safety loading.
    """
    net_arr = np.asarray(pure_premium_val, dtype=np.float64)
    load_arr = np.asarray(loading, dtype=np.float64)
    return np.asarray(net_arr * (1.0 + load_arr), dtype=np.float64)