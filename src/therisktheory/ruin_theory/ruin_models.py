"""Ruin theory models for the classical Cramer-Lundberg risk process.

Topic 5: Ruin Theory.
"""

import numpy as np


def _as_finite_real_array(value, name):
    """Convert real numeric input to a finite float64 NumPy array."""
    array = np.asarray(value)

    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    return result


def _return_scalar_if_scalar(result, *inputs):
    """Return a Python float when all inputs are scalar."""
    if all(np.asarray(value).ndim == 0 for value in inputs):
        return float(np.asarray(result))
    return np.asarray(result, dtype=np.float64)


def adjustment_coefficient_exponential_severity(
    safety_loading_theta: float | np.ndarray,
    severity_mean_beta: float | np.ndarray,
) -> float | np.ndarray:
    r"""Calculate the adjustment coefficient for exponential claim severity.

    For exponential claims with mean beta and positive safety loading theta:

        R = theta / (beta * (1 + theta))
    """
    theta = _as_finite_real_array(safety_loading_theta, "safety_loading_theta")
    beta = _as_finite_real_array(severity_mean_beta, "severity_mean_beta")

    if np.any(theta <= 0):
        raise ValueError("Safety loading theta must be strictly positive.")
    if np.any(beta <= 0):
        raise ValueError("Severity mean beta must be strictly positive.")

    try:
        theta, beta = np.broadcast_arrays(theta, beta)
    except ValueError as exc:
        raise ValueError("Inputs must have broadcast-compatible shapes.") from exc

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = theta / (beta * (1.0 + theta))

    if not np.all(np.isfinite(result)):
        raise OverflowError("Adjustment coefficient is not finite.")

    return _return_scalar_if_scalar(
        result, safety_loading_theta, severity_mean_beta
    )


def lundberg_ruin_bound(
    initial_surplus_u: float | np.ndarray,
    adjustment_coefficient_r: float | np.ndarray,
) -> float | np.ndarray:
    r"""Calculate Lundberg's upper bound: exp(-R * u)."""
    surplus = _as_finite_real_array(initial_surplus_u, "initial_surplus_u")
    coefficient = _as_finite_real_array(
        adjustment_coefficient_r, "adjustment_coefficient_r"
    )

    if np.any(surplus < 0):
        raise ValueError("Initial surplus must be non-negative.")
    if np.any(coefficient <= 0):
        raise ValueError("Adjustment coefficient R must be strictly positive.")

    try:
        surplus, coefficient = np.broadcast_arrays(surplus, coefficient)
    except ValueError as exc:
        raise ValueError("Inputs must have broadcast-compatible shapes.") from exc

    result = np.exp(-coefficient * surplus)
    return _return_scalar_if_scalar(
        result, initial_surplus_u, adjustment_coefficient_r
    )


def cramer_lundberg_ruin_approximation(
    initial_surplus_u: float | np.ndarray,
    safety_loading_theta: float | np.ndarray,
    severity_mean: float | np.ndarray,
    poisson_lambda: float | np.ndarray = 1.0,
) -> float | np.ndarray:
    r"""Calculate the ultimate ruin probability for exponential claims.

    For the classical Cramer-Lundberg model with exponential claim sizes,
    the expression is exact:

        psi(u) = exp(-R*u) / (1 + theta)

    The claim arrival intensity lambda must be positive, but cancels from
    the adjustment-coefficient equation and therefore does not appear in
    the final expression.
    """
    surplus = _as_finite_real_array(initial_surplus_u, "initial_surplus_u")
    theta = _as_finite_real_array(safety_loading_theta, "safety_loading_theta")
    beta = _as_finite_real_array(severity_mean, "severity_mean")
    intensity = _as_finite_real_array(poisson_lambda, "poisson_lambda")

    if np.any(surplus < 0):
        raise ValueError("Initial surplus must be non-negative.")
    if np.any(theta <= 0):
        raise ValueError("Safety loading theta must be strictly positive.")
    if np.any(beta <= 0):
        raise ValueError("Severity mean must be strictly positive.")
    if np.any(intensity <= 0):
        raise ValueError("Poisson claim intensity must be strictly positive.")

    try:
        surplus, theta, beta, intensity = np.broadcast_arrays(
            surplus, theta, beta, intensity
        )
    except ValueError as exc:
        raise ValueError("Inputs must have broadcast-compatible shapes.") from exc

    coefficient = theta / (beta * (1.0 + theta))
    result = np.exp(-coefficient * surplus) / (1.0 + theta)

    if not np.all(np.isfinite(result)):
        raise OverflowError("Ruin probability result is not finite.")

    return _return_scalar_if_scalar(
        result, initial_surplus_u, safety_loading_theta, severity_mean,
        poisson_lambda
    )
