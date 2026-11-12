"""Partial credibility models, full credibility standards, and adjusted premiums.

Topic 4: Credibility Theory and Reserves.
Author: Maria Lucia Monticelli D’Elias
Reference: Bühlmann, H. — Mathematical Methods in Risk Theory; Klugman, S. et al. — Loss Models.
"""

import numpy as np
from scipy.special import erfinv

# Bracamonte aporta la f?rmula com?n de prima de credibilidad.
# Se mantiene una alternativa local mientras su m?dulo no est? integrado.
try:
    from .premium import credibility_premium as bracamonte_credibility_premium
except ImportError:
    bracamonte_credibility_premium = None

def partial_credibility_standard(
    expected_claim_count: float | np.ndarray,
    target_accuracy: float = 0.05,
    confidence_level: float = 0.95,
) -> float | np.ndarray:
    """Calculate the full credibility standard from accuracy and confidence.

    Theoretical Context:
        The classical full credibility standard r_0 ensures that the observed sample mean 
        is within k percent of the true mean with probability equal to the confidence level.
        Formula based on Normal approximation:
            r_0 = ( z_{1 - alpha/2} / k )^2

    Args:
        expected_claim_count: Expected number of claims or exposure base. Must be non-negative.
        target_accuracy: Maximum allowable error margin k (e.g., 0.05 for 5%). Must be in (0, 1).
        confidence_level: Statistical confidence level (e.g., 0.95). Must be in (0, 1).

    Returns:
        float | np.ndarray: Required standard for full credibility.
    """
    n_exp = np.asarray(expected_claim_count, dtype=np.float64)
    k = float(target_accuracy)
    p = float(confidence_level)

    if np.any(n_exp < 0.0):
        raise ValueError("Expected claim count must be non-negative.")
    if not (0.0 < k < 1.0):
        raise ValueError("Target accuracy k must be strictly between 0 and 1.")
    if not (0.0 < p < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")

    # Quantile estimation using SciPy's error function inverse (erfinv)
    z_score = np.sqrt(2.0) * erfinv(p)
    standard = (z_score / k) ** 2
    
    return float(standard) if n_exp.ndim == 0 else np.full_like(n_exp, standard)


def partial_credibility_premium(
    collective_premium: float | np.ndarray,
    observed_experience_mean: float | np.ndarray,
    credibility_weight: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the partial credibility premium combining collective prior and observed experience.

    Theoretical Context:
        The partial credibility premium P_C is given by the classic linear credibility formula:
            P_C = Z * X_bar + (1 - Z) * Mu
        where:
            - Z is the credibility weight in [0, 1] (supplied by Garcia's module).
            - X_bar is the observed portfolio experience mean.
            - Mu is the collective/prior baseline premium (supplied by Bracamonte's module).

    Args:
        collective_premium: Prior or collective premium baseline (Mu). Must be non-negative.
        observed_experience_mean: Observed sample mean of claims (X_bar).
        credibility_weight: Credibility factor Z assigned to the portfolio data, in [0, 1].

    Returns:
        float | np.ndarray: The resulting credibility-adjusted premium.
    """
    mu = np.asarray(collective_premium, dtype=np.float64)
    x_bar = np.asarray(observed_experience_mean, dtype=np.float64)
    z = np.asarray(credibility_weight, dtype=np.float64)

    if np.any(mu < 0.0) or np.any(x_bar < 0.0):
        raise ValueError("Premiums and experience means must be non-negative.")
    if np.any(z < 0.0) or np.any(z > 1.0):
        raise ValueError("Credibility weight Z must be between 0 and 1 shapes/values.")

    premium_c = z * x_bar + (1.0 - z) * mu
    return np.asarray(premium_c, dtype=np.float64)


def integrated_partial_credibility_evaluation(
    exposure_count: float,
    observed_mean: float,
    collective_base: float,
    target_accuracy: float = 0.05,
) -> dict[str, float]:
    """Integration wrapper combining Bracamonte's/Garcia's concepts with partial credibility.

    Args:
        exposure_count: Number of risks or observed exposures.
        observed_mean: Observed portfolio mean experience.
        collective_base: Collective baseline premium.
        target_accuracy: Target error margin for full credibility standard.

    Returns:
        dict[str, float]: Comprehensive dictionary with standards, weights, and final premium.
    """
    r_0 = float(partial_credibility_standard(exposure_count, target_accuracy))
    
    # Limited fluctuation credibility; this is not the Buhlmann weight.
    # Buhlmann requires EPV and VHM, which this wrapper does not receive.
    z = float(min(1.0, np.sqrt(exposure_count / r_0))) if r_0 > 0 else 0.0

    if bracamonte_credibility_premium is not None:
        # Bracamonte's signature: (individual_mean, collective_mean, weight).
        premium_result = bracamonte_credibility_premium(
            observed_mean, collective_base, z
        )
        final_premium = float(np.asarray(premium_result))
    else:
        final_premium = float(
            partial_credibility_premium(collective_base, observed_mean, z)
        )

    return {
        "full_credibility_standard": r_0,
        "credibility_weight_z": z,
        "adjusted_premium": final_premium,
    }