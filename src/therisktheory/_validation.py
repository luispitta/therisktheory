"""Validation utilities for inputs and results."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray

def validate_numeric_array(x: ArrayLike, name: str = "input") -> NDArray[np.float64]:
    """Validates that input is convertible to a real numeric array and finite."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{name} must be made of real numbers.") from e
    
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(f"{name} must not contain complex numbers.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite real numbers.")
    return arr

def as_real_array(x: ArrayLike, name: str = "input", *, allow_inf: bool = False) -> NDArray[np.float64]:
    arr = validate_numeric_array(x, name)
    if allow_inf:
        return arr
    return arr

def as_nonnegative_finite_array(x: ArrayLike, name: str = "input") -> NDArray[np.float64]:
    arr = validate_numeric_array(x, name)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    return arr

def as_probability_array(x: ArrayLike, name: str = "input") -> NDArray[np.float64]:
    arr = validate_numeric_array(x, name)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} probabilities must be between 0.0 and 1.0.")
    return arr

def check_broadcastable(**kwargs: NDArray[np.float64]) -> None:
    try:
        np.broadcast_arrays(*kwargs.values())
    except ValueError as e:
        raise ValueError("Shapes cannot be broadcast together.") from e

def check_finite_result(result: NDArray[np.float64], name: str = "function") -> NDArray[np.float64]:
    if not np.all(np.isfinite(result)):
        raise OverflowError(f"Result of {name} exceeds float64 range or is NaN.")
    return result
