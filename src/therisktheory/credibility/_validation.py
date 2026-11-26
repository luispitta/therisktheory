"""Input validation helpers for credibility models."""

import numpy as np


def as_real_array(value, name):
    """Convert real numeric input to a finite float64 array."""
    array = np.asarray(value)

    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    return result


def as_probability_array(value, name):
    """Convert input to a finite probability array in [0, 1]."""
    result = as_real_array(value, name)

    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must be between 0 and 1.")

    return result


def as_nonnegative_finite_array(value, name):
    """Convert input to a finite non-negative float64 array."""
    result = as_real_array(value, name)

    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative.")

    return result


def check_broadcastable(**arrays):
    """Raise ValueError when named arrays cannot be broadcast together."""
    try:
        np.broadcast_arrays(*arrays.values())
    except ValueError as exc:
        names = ", ".join(arrays)
        raise ValueError(
            f"Inputs ({names}) must have broadcast-compatible shapes."
        ) from exc


def check_finite_result(value, name):
    """Raise OverflowError if a computed result is not finite."""
    result = np.asarray(value, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise OverflowError(f"{name} produced a non-finite result.")

    return result
