"""Stability parameters, sensitivity, and safety loadings in collective risk.

Topic 5: Collective risk process, Lundberg, and reinsurance.
Author: Adrianna Passanisi

References:
    - Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer.
      Chapter 6: The Risk Process, Section 6.4: Stability criteria.
    - Beard, R., Pentikäinen, T., and Pesonen, E. (1984). Risk Theory.
      3rd ed. Chapman & Hall. Section 3.2: The stability criterion,
      and Section 3.6: Fluctuation of basic probabilities.
    - Daykin, C., Pentikäinen, T., and Pesonen, E. (1994).
      Practical Risk Theory for Actuaries. Chapman & Hall.
      Section 3.2 & Section 8.3: Safety loading and initial reserve.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from therisktheory._validation import (
    as_nonnegative_finite_array,
    as_probability_array,
    check_broadcastable,
    check_finite_result,
)

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class StabilitySensitivity:
    """Analytical sensitivities and elasticities of the stability parameter.

    Attributes:
        stability: Base stability parameter K = E[G] / sqrt(Var(G)).
        d_theta: Derivative w.r.t. safety loading dK/dtheta.
        d_lambda: Derivative w.r.t. claim frequency rate dK/dlambda.
        elasticity_theta: Constant elasticity w.r.t. safety loading (equals 1.0).
        elasticity_lambda: Constant elasticity w.r.t. frequency (equals 0.5).
    """

    stability: npt.NDArray[np.float64]
    d_theta: npt.NDArray[np.float64]
    d_lambda: npt.NDArray[np.float64]
    elasticity_theta: npt.NDArray[np.float64]
    elasticity_lambda: npt.NDArray[np.float64]


def stability_parameter(
    claim_rate: npt.ArrayLike,
    severity_mean: npt.ArrayLike,
    severity_second_moment: npt.ArrayLike,
    safety_loading: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Compute the classical collective stability parameter.

    Defined as the ratio of expected annual result to standard deviation:
        K = \frac{E[G]}{\sigma(G)} = \frac{\theta \lambda \mu_1}{\sqrt{\lambda \mu_2}}
          = \theta \sqrt{\lambda} \frac{\mu_1}{\sqrt{\mu_2}}

    Args:
        claim_rate: Poisson claim frequency intensity lambda > 0.
        severity_mean: Expected individual claim severity mu_1 > 0.
        severity_second_moment: Second raw moment of severity mu_2 >= mu_1^2.
        safety_loading: Relative premium safety loading theta >= 0.

    Returns:
        Stability parameter K as a float64 NumPy array.

    Raises:
        TypeError: If inputs cannot be cast to real numbers.
        ValueError: If parameters are negative or mu_2 < mu_1^2.
    """
    lam = as_nonnegative_finite_array(claim_rate, "claim_rate")
    mu1 = as_nonnegative_finite_array(severity_mean, "severity_mean")
    mu2 = as_nonnegative_finite_array(
        severity_second_moment, "severity_second_moment"
    )
    theta = as_nonnegative_finite_array(safety_loading, "safety_loading")

    if np.any(lam <= 0.0) or np.any(mu1 <= 0.0):
        raise ValueError("claim_rate and severity_mean must be strictly positive.")

    check_broadcastable(
        claim_rate=lam,
        severity_mean=mu1,
        severity_second_moment=mu2,
        safety_loading=theta,
    )
    lam, mu1, mu2, theta = np.broadcast_arrays(lam, mu1, mu2, theta)

    if np.any(mu2 < mu1**2):
        raise ValueError("severity_second_moment must be at least severity_mean^2.")

    with np.errstate(over="ignore", invalid="ignore"):
        k = theta * np.sqrt(lam) * (mu1 / np.sqrt(mu2))

    return check_finite_result(k, "stability_parameter")


def stability_sensitivity(
    claim_rate: npt.ArrayLike,
    severity_mean: npt.ArrayLike,
    severity_second_moment: npt.ArrayLike,
    safety_loading: npt.ArrayLike,
) -> StabilitySensitivity:
    r"""Calculate analytical derivatives and elasticities of stability.

    Formulas:
        dK / d\theta = \sqrt{\lambda} \frac{\mu_1}{\sqrt{\mu_2}} = \frac{K}{\theta}
        dK / d\lambda = \frac{\theta \mu_1}{2 \sqrt{\lambda \mu_2}} = \frac{K}{2\lambda}
        \epsilon_\theta = 1.0,  \epsilon_\lambda = 0.5

    Args:
        claim_rate: Poisson claim frequency intensity lambda > 0.
        severity_mean: Expected individual claim severity mu_1 > 0.
        severity_second_moment: Second raw moment mu_2 >= mu_1^2.
        safety_loading: Safety loading theta > 0.

    Returns:
        StabilitySensitivity dataclass with gradients and elasticities.
    """
    k = stability_parameter(
        claim_rate, severity_mean, severity_second_moment, safety_loading
    )
    lam = as_nonnegative_finite_array(claim_rate, "claim_rate")
    theta = as_nonnegative_finite_array(
        safety_loading, "safety_loading"
    )

    if np.any(theta <= 0.0):
        raise ValueError("safety_loading must be strictly positive for sensitivity.")

    lam, theta = np.broadcast_arrays(lam, theta)

    dk_dtheta = k / theta
    dk_dlambda = k / (2.0 * lam)
    e_theta = np.ones_like(k, dtype=np.float64)
    e_lambda = np.full_like(k, 0.5, dtype=np.float64)

    return StabilitySensitivity(
        stability=k,
        d_theta=dk_dtheta,
        d_lambda=dk_dlambda,
        elasticity_theta=e_theta,
        elasticity_lambda=e_lambda,
    )


def collective_safety_loading(
    target_stability: npt.ArrayLike,
    claim_rate: npt.ArrayLike,
    severity_mean: npt.ArrayLike,
    severity_second_moment: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Determine safety loading theta required to achieve a target stability.

    Solves:
        K_{\text{target}} = \theta \sqrt{\lambda} \frac{\mu_1}{\sqrt{\mu_2}}
        \implies \theta = \frac{K_{\text{target}}}{\sqrt{\lambda}} \frac{\sqrt{\mu_2}}{\mu_1}

    Args:
        target_stability: Desired stability level K >= 0.
        claim_rate: Claim frequency intensity lambda > 0.
        severity_mean: Expected claim severity mu_1 > 0.
        severity_second_moment: Second raw moment mu_2 >= mu_1^2.

    Returns:
        Required safety loading theta as a float64 array.
    """
    k_tar = as_nonnegative_finite_array(target_stability, "target_stability")
    lam = as_nonnegative_finite_array(claim_rate, "claim_rate")
    mu1 = as_nonnegative_finite_array(severity_mean, "severity_mean")
    mu2 = as_nonnegative_finite_array(
        severity_second_moment, "severity_second_moment"
    )

    if np.any(lam <= 0.0) or np.any(mu1 <= 0.0):
        raise ValueError("claim_rate and severity_mean must be strictly positive.")

    check_broadcastable(
        target_stability=k_tar,
        claim_rate=lam,
        severity_mean=mu1,
        severity_second_moment=mu2,
    )
    k_tar, lam, mu1, mu2 = np.broadcast_arrays(k_tar, lam, mu1, mu2)

    if np.any(mu2 < mu1**2):
        raise ValueError("severity_second_moment must be at least severity_mean^2.")

    with np.errstate(over="ignore", invalid="ignore"):
        theta = (k_tar / np.sqrt(lam)) * (np.sqrt(mu2) / mu1)

    return check_finite_result(theta, "collective_safety_loading")


def loading_from_initial_reserve(
    initial_reserve: npt.ArrayLike,
    target_ruin_probability: npt.ArrayLike,
    severity_mean: npt.ArrayLike,
    severity_second_moment: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Calculate safety loading theta required for a given initial reserve.

    Under the diffusion approximation to the Cramér-Lundberg process
    (Daykin et al. 1994, Section 8.3), the asymptotic adjustment coefficient
    satisfies R \approx 2*theta*mu_1 / mu_2 for moderate loadings theta << 1.
    Equating the ruin bound psi(u) \approx exp(-R*u) to target ruin
    probability epsilon yields:
        theta(u) = \frac{\mu_2 \ln(1 / \epsilon)}{2 \mu_1 u}

    Scope and Limitations:
        - Asymptotic Regime: Best suited for moderate relative safety loadings
          and non-zero capital reserves. As u -> inf, theta -> 0.
        - Heavy Tails: For severely heavy-tailed claims where mu_2 is very large,
          exact Lundberg root solvers are recommended.

    Args:
        initial_reserve: Available initial surplus reserve u > 0.
        target_ruin_probability: Maximum acceptable ruin probability in (0, 1).
        severity_mean: Expected claim severity mu_1 > 0.
        severity_second_moment: Second raw moment mu_2 >= mu_1^2.

    Returns:
        Required relative safety loading theta as a float64 array.
    """
    u = as_nonnegative_finite_array(initial_reserve, "initial_reserve")
    eps = as_probability_array(target_ruin_probability, "target_ruin_probability")
    mu1 = as_nonnegative_finite_array(severity_mean, "severity_mean")
    mu2 = as_nonnegative_finite_array(
        severity_second_moment, "severity_second_moment"
    )

    if np.any(u <= 0.0) or np.any(mu1 <= 0.0):
        raise ValueError("initial_reserve and severity_mean must be strictly positive.")
    if np.any(eps <= 0.0) or np.any(eps >= 1.0):
        raise ValueError("target_ruin_probability must be strictly in (0, 1).")

    check_broadcastable(
        initial_reserve=u,
        target_ruin_probability=eps,
        severity_mean=mu1,
        severity_second_moment=mu2,
    )
    u, eps, mu1, mu2 = np.broadcast_arrays(u, eps, mu1, mu2)

    if np.any(mu2 < mu1**2):
        raise ValueError("severity_second_moment must be at least severity_mean^2.")

    with np.errstate(over="ignore", invalid="ignore"):
        theta = (mu2 * np.log(1.0 / eps)) / (2.0 * mu1 * u)

    return check_finite_result(theta, "loading_from_initial_reserve")
