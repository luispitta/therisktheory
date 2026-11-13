"""Safety equation and Lundberg fundamental equation models in collective risk theory.

References:
    Beard, R., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory. Chapman and Hall.
    Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer.
    Gerber, H. (1979). An Introduction to Mathematical Risk Theory. Huebner Foundation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import optimize


def lundberg_equation(
    r: float,
    mgf_severity: Callable[[float], float],
    mean_severity: float,
    safety_loading: float,
) -> float:
    """Evaluate Lundberg's fundamental equation residual at r.

    Lundberg's equation for a compound Poisson risk model with relative
    safety loading lambda > 0 is:
        M_X(r) - 1 - (1 + lambda) * E[X] * r = 0

    Args:
        r: Evaluation point strictly inside the MGF existence domain (-inf, r_0),
            where M_X(r) < infinity.
        mgf_severity: Moment generating function M_X(r) of claim severity X.
            Must return a finite float inside its domain and inf (or raise
            ValueError) outside it.
        mean_severity: Expected claim severity E[X] (> 0).
        safety_loading: Relative safety loading lambda (> 0).

    Returns:
        Residual value of Lundberg's equation at r.

    Raises:
        ValueError: If mean_severity <= 0, safety_loading <= 0, or M_X(r) is
            not finite at r (outside the MGF domain of existence).
    """
    if mean_severity <= 0:
        raise ValueError("mean_severity must be strictly positive.")
    if safety_loading <= 0:
        raise ValueError("safety_loading must be strictly positive.")

    mgf_val = mgf_severity(r)
    if not np.isfinite(mgf_val):
        raise ValueError(
            f"mgf_severity({r}) is not finite; r lies outside the MGF existence domain."
        )

    return float(mgf_val - 1.0 - (1.0 + safety_loading) * mean_severity * r)


def adjustment_coefficient(
    mgf_severity: Callable[[float], float],
    mean_severity: float,
    safety_loading: float,
    r_min: float = 1e-9,
    r_max: float = 5.0,
) -> float:
    """Calculate the adjustment coefficient R (Lundberg's exponent).

    R > 0 is the unique positive root of:
        M_X(R) - 1 - (1 + lambda) * E[X] * R = 0
    within the MGF existence interval (0, r_0), where r_0 = sup{r >= 0 : M_X(r) < inf}.

    Root-Finding Search Interval:
        Root finding uses Brent's method over [r_min, r_max] subset (0, r_0).
        Since g(0) = 0 and g'(0) = -lambda * E[X] < 0, r_min > 0 must be small
        enough that g(r_min) < 0, and r_max must lie strictly inside the MGF
        domain of existence (r_max < r_0) with g(r_max) > 0.

    Args:
        mgf_severity: Moment generating function M_X(r) of claim severity X.
        mean_severity: Expected claim severity E[X] (> 0).
        safety_loading: Relative safety loading factor lambda (> 0).
        r_min: Strictly positive lower bound of [r_min, r_max]. Defaults to 1e-9.
        r_max: Upper bound of [r_min, r_max], strictly inside the MGF
            domain of existence (M_X(r_max) < inf). Defaults to 5.0.

    Returns:
        The unique positive adjustment coefficient R.

    Raises:
        ValueError: If parameters are non-positive, [r_min, r_max] is invalid,
            r_max is outside the MGF existence domain, or no sign change occurs.
        RuntimeError: If root finding fails to converge.
    """
    if mean_severity <= 0:
        raise ValueError("mean_severity must be strictly positive.")
    if safety_loading <= 0:
        raise ValueError("safety_loading must be strictly positive.")
    if r_min <= 0 or r_min >= r_max:
        raise ValueError("Search interval must satisfy 0 < r_min < r_max.")

    f_low = lundberg_equation(r_min, mgf_severity, mean_severity, safety_loading)
    f_high = lundberg_equation(r_max, mgf_severity, mean_severity, safety_loading)

    if f_low >= 0.0:
        raise ValueError(
            f"No sign change at lower bound r_min={r_min}; use a smaller r_min."
        )
    if f_high <= 0.0:
        raise ValueError(
            f"No sign change at upper bound r_max={r_max}; "
            "increase r_max within the MGF domain of existence."
        )

    def obj(r: float) -> float:
        return lundberg_equation(r, mgf_severity, mean_severity, safety_loading)

    sol = optimize.root_scalar(obj, bracket=[r_min, r_max], method="brentq")
    if not sol.converged:
        raise RuntimeError("Root finding algorithm did not converge.")
    return float(sol.root)


def full_retention_adjustment(
    claim_moments: tuple[float, float],
    safety_loading: float,
) -> float:
    """Calculate the two-moment approximation of R under full retention.

    Using a second-order Taylor expansion of M_X(R):
        R ≈ (2 * lambda * E[X]) / E[X^2]

    Args:
        claim_moments: Tuple (E[X], E[X^2]) of first and second raw severity moments.
        safety_loading: Relative safety loading factor lambda (> 0).

    Returns:
        Approximate adjustment coefficient R under full retention.

    Raises:
        ValueError: If moments are non-positive, E[X^2] < E[X]^2, or
            safety_loading <= 0.
    """
    mean_x, second_moment_x = claim_moments
    if mean_x <= 0 or second_moment_x <= 0:
        raise ValueError("Claim moments E[X] and E[X^2] must be strictly positive.")
    if second_moment_x < mean_x**2:
        raise ValueError("Invalid moments: E[X^2] cannot be less than E[X]^2.")
    if safety_loading <= 0:
        raise ValueError("safety_loading must be strictly positive.")
    return (2.0 * safety_loading * mean_x) / second_moment_x
