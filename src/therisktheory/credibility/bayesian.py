"""Bayesian credibility models under Gamma-Poisson conjugacy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def _validate_positive_scalar_or_array(
    value: npt.ArrayLike, name: str
) -> npt.NDArray[np.float64]:
    """Validate that input is finite and strictly positive (> 0)."""
    arr = as_nonnegative_finite_array(value, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive (> 0)")
    return arr


def bayesian_credibility_premium(
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    claims: npt.ArrayLike,
    exposure: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Bayesian credibility pure premium under a Gamma-Poisson model.

    Given a Gamma(alpha, beta) prior for the underlying claim frequency parameter
    lambda and Poisson-distributed observed claim counts X over exposure t, the
    conjugate posterior distribution is:
        lambda | X ~ Gamma(alpha + X, beta + t)

    The Bayesian credibility premium under squared-error loss is the posterior mean:
        P_Bayes = E[lambda | X] = (alpha + X) / (beta + t)
                = Z * (X / t) + (1 - Z) * (alpha / beta)
    where Z = t / (beta + t) is the Bayesian credibility factor.

    Args:
        alpha: Prior Gamma shape parameter (alpha > 0).
        beta: Prior Gamma rate parameter (beta > 0).
        claims: Observed claim counts (claims >= 0).
        exposure: Observed exposure units (exposure > 0).

    Returns:
        np.ndarray: Bayesian credibility premium as a float64 array (0-d for scalars).

    Raises:
        TypeError: If any input is not made of real numbers.
        ValueError: If alpha <= 0, beta <= 0, exposure <= 0, claims < 0,
            any input is NaN/infinite, or shapes cannot be broadcast together.
        OverflowError: If the computed result exceeds the float64 range.

    References:
        Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer, Ch. 4.
        Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). Loss Models:
            From Data to Decisions (5th ed.). Wiley, Ch. 16.
    """
    alpha_arr = _validate_positive_scalar_or_array(alpha, "alpha")
    beta_arr = _validate_positive_scalar_or_array(beta, "beta")
    claims_arr = as_nonnegative_finite_array(claims, "claims")
    exp_arr = _validate_positive_scalar_or_array(exposure, "exposure")

    check_broadcastable(
        alpha=alpha_arr,
        beta=beta_arr,
        claims=claims_arr,
        exposure=exp_arr,
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        posterior_alpha = alpha_arr + claims_arr
        posterior_beta = beta_arr + exp_arr
        direct = posterior_alpha / posterior_beta
        # Halve all terms only if either sum overflows. The ratio is unchanged.
        overflow = ~np.isfinite(posterior_alpha) | ~np.isfinite(posterior_beta)
        scaled = (alpha_arr / 2 + claims_arr / 2) / (
            beta_arr / 2 + exp_arr / 2
        )
        result = np.asarray(np.where(overflow, scaled, direct), dtype=np.float64)

    return check_finite_result(result, "bayesian_credibility_premium")


def posterior_predictive_mean(
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    claims: npt.ArrayLike,
    exposure: npt.ArrayLike,
    future_exposure: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    """Compute the posterior predictive mean claim frequency for a future period.

    In the Gamma-Poisson conjugate model, the posterior predictive distribution
    for future claims X_new over future_exposure t_new is Negative Binomial
    (Gamma-Poisson mixture) with expected value:
        E[X_new | X] = t_new * E[lambda | X] = t_new * (alpha + X) / (beta + t)

    When future_exposure = 1.0 (default unit exposure), this equals the
    Bayesian credibility premium E[lambda | X].

    Args:
        alpha: Prior Gamma shape parameter (alpha > 0).
        beta: Prior Gamma rate parameter (beta > 0).
        claims: Observed claim counts (claims >= 0).
        exposure: Observed exposure units (exposure > 0).
        future_exposure: Future exposure units for prediction (default 1.0, > 0).

    Returns:
        np.ndarray: Posterior predictive expected claims as a float64 array.

    Raises:
        TypeError: If any input is not made of real numbers.
        ValueError: If parameters/exposures <= 0, claims < 0, non-finite inputs,
            or incompatible broadcast shapes.
        OverflowError: If the result overflows float64.

    References:
        Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). Loss Models:
            From Data to Decisions (5th ed.). Wiley, Ch. 16.
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). SOA.
    """
    fut_exp_arr = _validate_positive_scalar_or_array(
        future_exposure, "future_exposure"
    )
    cred_premium = bayesian_credibility_premium(alpha, beta, claims, exposure)

    check_broadcastable(
        credibility_premium=cred_premium,
        future_exposure=fut_exp_arr,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(fut_exp_arr * cred_premium, dtype=np.float64)

    return check_finite_result(result, "posterior_predictive_mean")

# Backward-compatible names from the prior integration snapshot.
from therisktheory.solvency_pricing._retention_compat import (  # noqa: E402, F401
    deflate_losses,
    inflate_losses,
    retained_loss,
    retention_capacity,
)
