"""Aggregate loss models, moments, and generating functions.

Topic 3: Frequency, Severity, Aggregate Loss, Approximations, and Simulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def _as_finite_real_array(value: npt.ArrayLike, name: str) -> np.ndarray:
    """Convert a real numeric input to float64 and reject non-finite values."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a real numeric value or array"
        ) from error

    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")

    array = array.astype(np.float64, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def _as_finite_complex_array(value: npt.ArrayLike, name: str) -> np.ndarray:
    """Convert a numeric input to complex128 and reject non-finite values."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric value or array") from error

    if array.dtype.kind not in "iufc":
        raise TypeError(f"{name} must contain numeric values")

    array = array.astype(np.complex128, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def aggregate_loss_moments(
    expected_frequency: float | np.ndarray,
    variance_frequency: float | np.ndarray,
    expected_severity: float | np.ndarray,
    variance_severity: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate aggregate loss mean, variance, and standard deviation.

    E[S] = E[N] E[X]
    Var(S) = E[N] Var(X) + Var(N) E[X]**2
    """
    e_n = as_nonnegative_finite_array(expected_frequency, "expected_frequency")
    v_n = as_nonnegative_finite_array(variance_frequency, "variance_frequency")
    e_x = as_nonnegative_finite_array(expected_severity, "expected_severity")
    v_x = as_nonnegative_finite_array(variance_severity, "variance_severity")

    check_broadcastable(e_n=e_n, v_n=v_n, e_x=e_x, v_x=v_x)

    with np.errstate(over="ignore", invalid="ignore"):
        mean_s = e_n * e_x
        variance_s = e_n * v_x + v_n * (e_x**2)
        std_s = np.sqrt(variance_s)

    check_finite_result(mean_s, "aggregate mean")
    check_finite_result(variance_s, "aggregate variance")
    check_finite_result(std_s, "aggregate standard deviation")

    return (
        np.asarray(mean_s, dtype=np.float64),
        np.asarray(variance_s, dtype=np.float64),
        np.asarray(std_s, dtype=np.float64),
    )


def aggregate_mgf(
    t: float | np.ndarray,
    lambda_poisson: float | np.ndarray,
    severity_mgf_values: float | np.ndarray,
) -> np.ndarray:
    """Calculate the compound Poisson aggregate loss MGF.

    M_S(t) = exp(lambda * (M_X(t) - 1)).
    The severity MGF values must be supplied by the caller.
    """
    t_arr = _as_finite_real_array(t, "t")
    lam = as_nonnegative_finite_array(lambda_poisson, "lambda_poisson")
    mgf_x = _as_finite_real_array(severity_mgf_values, "severity_mgf_values")

    if np.any(mgf_x <= 0.0):
        raise ValueError("Severity MGF values must be strictly positive.")

    check_broadcastable(t=t_arr, lambda_poisson=lam, severity_mgf_values=mgf_x)

    with np.errstate(over="ignore", invalid="ignore"):
        mgf_s = np.exp(lam * (mgf_x - 1.0))

    check_finite_result(mgf_s, "aggregate MGF")
    return np.asarray(mgf_s, dtype=np.float64)


def aggregate_characteristic_function(
    t: float | np.ndarray,
    lambda_poisson: float | np.ndarray,
    severity_cf_values: np.ndarray,
) -> np.ndarray:
    """Calculate the compound Poisson aggregate loss characteristic function.

    phi_S(t) = exp(lambda * (phi_X(t) - 1)).
    """
    t_arr = _as_finite_real_array(t, "t")
    lam = as_nonnegative_finite_array(lambda_poisson, "lambda_poisson")
    cf_x = _as_finite_complex_array(severity_cf_values, "severity_cf_values")

    check_broadcastable(t=t_arr, lambda_poisson=lam, severity_cf_values=cf_x)

    with np.errstate(over="ignore", invalid="ignore"):
        cf_s = np.exp(lam * (cf_x - 1.0))

    check_finite_result(cf_s, "aggregate characteristic function")
    return np.asarray(cf_s, dtype=np.complex128)


def individual_model_loss(
    probabilities: np.ndarray,
    amounts: np.ndarray,
) -> tuple[float, float]:
    """Calculate the mean and variance under the individual risk model.

    E[S] = sum(q_i * b_i)
    Var(S) = sum(q_i * (1 - q_i) * b_i**2)
    """
    q = _as_finite_real_array(probabilities, "probabilities")
    b = as_nonnegative_finite_array(amounts, "amounts")

    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("Probabilities must be between 0 and 1.")

    check_broadcastable(probabilities=q, amounts=b)

    with np.errstate(over="ignore", invalid="ignore"):
        mean_ind = np.sum(q * b)
        var_ind = np.sum(q * (1.0 - q) * (b**2))

    check_finite_result(np.asarray(mean_ind), "individual model mean")
    check_finite_result(np.asarray(var_ind), "individual model variance")

    return float(mean_ind), float(var_ind)


def collective_model_loss(
    expected_count: float,
    severity_mean: float,
    severity_variance: float,
    count_variance: float | None = None,
) -> dict[str, float]:
    """Compute collective model moments.

    If count_variance is omitted, a Poisson frequency model is assumed:
    Var(N) = E[N].
    """
    en = as_nonnegative_finite_array(expected_count, "expected_count")
    ex = as_nonnegative_finite_array(severity_mean, "severity_mean")
    vx = as_nonnegative_finite_array(severity_variance, "severity_variance")

    if en.ndim != 0 or ex.ndim != 0 or vx.ndim != 0:
        raise ValueError("collective_model_loss requires scalar parameters")

    if count_variance is None:
        vn = en
    else:
        vn = as_nonnegative_finite_array(count_variance, "count_variance")
        if vn.ndim != 0:
            raise ValueError("count_variance must be a scalar")

    mean_s, var_s, std_s = aggregate_loss_moments(en, vn, ex, vx)

    return {
        "mean": float(mean_s),
        "variance": float(var_s),
        "std": float(std_s),
    }
