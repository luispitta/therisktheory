"""Full credibility standards and required exposure models.

Topic 4: Credibility Theory and Reserves.
Author: Adrianna Passanisi

References:
    - Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
      Modern Actuarial Risk Theory: Using R. 2nd ed. Springer.
      Chapter 8: Credibility Theory, Section 8.1.
    - Klugman, S. A., Panjer, H. H., and Willmot, G. E. (2012).
      Loss Models: From Data to Decisions. 4th ed. Wiley.
      Chapter 16: Classical Credibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfinv

from therisktheory._validation import (
    as_nonnegative_finite_array,
    as_probability_array,
    as_real_array,
    check_broadcastable,
    check_finite_result,
)

if TYPE_CHECKING:
    import numpy.typing as npt

_SQRT_2 = np.sqrt(2.0)


def _compute_z_score(confidence_level: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Calculate normal z-quantile for two-sided confidence level p in (0, 1)."""
    p = as_probability_array(confidence_level, "confidence_level")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    return _SQRT_2 * erfinv(p)


def full_credibility_standard(
    target_accuracy: npt.ArrayLike = 0.05,
    confidence_level: npt.ArrayLike = 0.95,
    severity_mean: npt.ArrayLike | None = None,
    severity_variance: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Compute required expected claim count for full credibility.

    Under classical limited fluctuation credibility, full credibility (Z = 1)
    requires that observed experience is within +/- k of true mean with
    probability p = 1 - alpha:
        P(|X - mu| <= k * mu) >= p

    For pure Poisson claim counts:
        n_0 = (z_{1 - alpha/2} / k)^2

    When severity is stochastic with mean E[X] and variance Var(X), the
    standard for aggregate claims incorporates the squared coefficient
    of variation C_X^2 = Var(X) / (E[X])^2:
        n_S = n_0 * (1 + Var(X) / (E[X])^2)

    Args:
        target_accuracy: Maximum allowable relative error margin k in (0, 1).
            Defaults to 0.05 (5% margin).
        confidence_level: Statistical confidence level p in (0, 1).
            Defaults to 0.95 (95% confidence).
        severity_mean: Optional expected individual claim severity E[X] > 0.
        severity_variance: Optional variance of claim severity Var(X) >= 0.

    Returns:
        Expected number of claims required for full credibility as float64 array.

    Raises:
        TypeError: If inputs cannot be converted to real numbers.
        ValueError: If k or p are not in (0, 1), or severity moments are invalid.
    """
    k_arr = as_real_array(target_accuracy, "target_accuracy")
    if np.any(k_arr <= 0.0) or np.any(k_arr >= 1.0):
        raise ValueError("target_accuracy must be strictly between 0 and 1.")

    z_score = _compute_z_score(confidence_level)
    check_broadcastable(target_accuracy=k_arr, z=z_score)
    k_b, z_b = np.broadcast_arrays(k_arr, z_score)

    n_0 = (z_b / k_b) ** 2

    # Severity correction factor (1 + Var(X) / E[X]^2)
    if severity_mean is not None or severity_variance is not None:
        if severity_mean is None or severity_variance is None:
            raise ValueError(
                "Both severity_mean and severity_variance must be provided."
            )
        mu_x = as_nonnegative_finite_array(severity_mean, "severity_mean")
        if np.any(mu_x <= 0.0):
            raise ValueError("severity_mean must be strictly positive.")
        var_x = as_nonnegative_finite_array(
            severity_variance, "severity_variance"
        )
        check_broadcastable(n0=n_0, mu_x=mu_x, var_x=var_x)
        n_0, mu_x_b, var_x_b = np.broadcast_arrays(n_0, mu_x, var_x)

        cv_sq = var_x_b / (mu_x_b**2)
        n_0 = n_0 * (1.0 + cv_sq)

    return check_finite_result(n_0, "full_credibility_standard")


def full_credibility_required_exposure(
    claim_frequency_rate: npt.ArrayLike,
    target_accuracy: npt.ArrayLike = 0.05,
    confidence_level: npt.ArrayLike = 0.95,
    severity_mean: npt.ArrayLike | None = None,
    severity_variance: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Compute required number of exposure units to achieve full credibility.

    Given an expected claim arrival rate lambda per exposure unit
    (e.g., claims per vehicle-year), the required exposure E_0 is:
        E_0 = n_standard / lambda

    Args:
        claim_frequency_rate: Expected claim frequency per exposure unit (lambda > 0).
        target_accuracy: Relative error margin k in (0, 1). Defaults to 0.05.
        confidence_level: Confidence level p in (0, 1). Defaults to 0.95.
        severity_mean: Optional expected individual claim severity E[X] > 0.
        severity_variance: Optional variance of claim severity Var(X) >= 0.

    Returns:
        Required exposure units as a float64 NumPy array.

    Raises:
        ValueError: If claim_frequency_rate <= 0 or parameters are out of domain.
    """
    lam = as_nonnegative_finite_array(
        claim_frequency_rate, "claim_frequency_rate"
    )
    if np.any(lam <= 0.0):
        raise ValueError("claim_frequency_rate must be strictly positive.")

    n_std = full_credibility_standard(
        target_accuracy=target_accuracy,
        confidence_level=confidence_level,
        severity_mean=severity_mean,
        severity_variance=severity_variance,
    )

    check_broadcastable(lam=lam, n_std=n_std)
    lam_b, n_std_b = np.broadcast_arrays(lam, n_std)

    required_exp = n_std_b / lam_b
    return check_finite_result(
        required_exp, "full_credibility_required_exposure"
    )
