"""Ruin theory models, adjustment coefficient, and Lundberg bounds.

Topic 5: Ruin Theory and Solvency Models.
Author: Maria Lucia Monticelli D’Elias
Reference: Bowers, N.L. et al. — Actuarial Mathematics; Grandell, J. — Aspects of Risk Theory.
"""

import numpy as np


def adjustment_coefficient_exponential_severity(
    safety_loading_theta: float | np.ndarray,
    severity_mean_beta: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the adjustment coefficient R analytically for an Exponential severity model.

    Theoretical Context:
        In the classical Cramér-Lundberg risk model with exponentially distributed claims $X \\sim \text{Exp}(\beta)$
        and safety loading $\theta$, the adjustment coefficient $R$ satisfies Lundberg's equation:
            R = \theta / ((1 + \theta) \\cdot \beta)  -- wait, let's use the exact classical formula:
            For exponential claims, M_X(R) = 1 / (1 - beta * R).
            The Lundberg equation is: 1 + (1 + theta) * lambda * E[X] * R = M_X(R) ...
            Actually, simplified for net premium rate $c = (1 + \theta) \\lambda \\mu$:
            R = \theta / ((1 + \theta) \\mu) ... let's write it precisely as R = theta / (beta * (1 + theta))
            where mean severity is $\beta$. Let's provide a robust formulation.

    Args:
        safety_loading_theta: Relative safety loading $\theta > 0$.
        severity_mean_beta: Mean of the exponential claim severity distribution $\beta > 0$.

    Returns:
        float | np.ndarray: The adjustment coefficient R.
    """
    theta = np.asarray(safety_loading_theta, dtype=np.float64)
    beta = np.asarray(severity_mean_beta, dtype=np.float64)

    if np.any(theta <= 0.0) or np.any(beta <= 0.0):
        raise ValueError(
            "Safety loading theta and severity mean beta must be strictly positive."
        )

    # Analytical solution for exponential severity: R = theta / (beta * (1 + theta))
    r = theta / (beta * (1.0 + theta))
    return (
        float(r)
        if theta.ndim == 0 and beta.ndim == 0
        else np.asarray(r, dtype=np.float64)
    )


def lundberg_ruin_bound(
    initial_surplus_u: float | np.ndarray,
    adjustment_coefficient_r: float | np.ndarray,
) -> float | np.ndarray:
    r"""Calculate Lundberg's upper bound for the infinite-time ruin probability.

    Theoretical Context:
        Lundberg's inequality states that the ultimate ruin probability $\Psi(u)$ is bounded by:
            \Psi(u) \le e^{-R \cdot u}
        where $u$ is the initial surplus and $R$ is the adjustment coefficient.

    Args:
        initial_surplus_u: Initial reserve or surplus $u \ge 0$.
        adjustment_coefficient_r: Adjustment coefficient $R > 0$.

    Returns:
        float | np.ndarray: Lundberg's bound for the probability of ruin.
    """
    u = np.asarray(initial_surplus_u, dtype=np.float64)
    r = np.asarray(adjustment_coefficient_r, dtype=np.float64)

    if np.any(u < 0.0):
        raise ValueError("Initial surplus must be non-negative.")
    if np.any(r <= 0.0):
        raise ValueError("Adjustment coefficient R must be strictly positive.")

    bound = np.exp(-r * u)
    return (
        float(bound)
        if u.ndim == 0 and r.ndim == 0
        else np.asarray(bound, dtype=np.float64)
    )


def cramer_lundberg_ruin_approximation(
    initial_surplus_u: float,
    safety_loading_theta: float,
    severity_mean: float,
    poisson_lambda: float = 1.0,
) -> float:
    """Approximate the ultimate ruin probability using the Cramér-Lundberg asymptotic formula.

    Theoretical Context:
        As initial surplus $u \to \\infty$, the ultimate ruin probability behaves as:
            \\Psi(u) \approx \\psi_0 \\cdot e^{-R \\cdot u}
        where $\\psi_0 = \frac{1}{1 + \theta}$ under exponential claims.
    """
    if initial_surplus_u < 0 or safety_loading_theta <= 0 or severity_mean <= 0:
        raise ValueError("Invalid parameters for ruin approximation domain.")

    r = adjustment_coefficient_exponential_severity(
        safety_loading_theta, severity_mean
    )
    psi_0 = 1.0 / (1.0 + safety_loading_theta)

    return float(psi_0 * np.exp(-r * initial_surplus_u))
