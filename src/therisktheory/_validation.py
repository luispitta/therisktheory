"""Input and output checks shared by the package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_NUMERIC_KINDS = frozenset("iuf")


def as_real_array(
    value: npt.ArrayLike, name: str, *, allow_inf: bool = False
) -> npt.NDArray[np.float64]:
    """Returns value as a float64 array of real numbers."""
    try:
        array = np.asarray(value)
    except ValueError as error:
        raise TypeError(f"{name} must be a real number or array") from error
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"{name} must be a real number or array, got dtype {array.dtype}"
        )
    array = array.astype(np.float64, copy=False)
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not be NaN")
    if not allow_inf and np.any(np.isinf(array)):
        raise ValueError(f"{name} must be finite")
    return array


def as_nonnegative_finite_array(
    value: npt.ArrayLike, name: str
) -> npt.NDArray[np.float64]:
    """Returns value as a non-negative float64 array."""
    array = as_real_array(value, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array


def as_probability_array(
    value: npt.ArrayLike, name: str
) -> npt.NDArray[np.float64]:
    """Returns value as a probability array in [0, 1]."""
    array = as_real_array(value, name)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be between 0 and 1")
    return array


def check_broadcastable(**arrays: npt.NDArray[np.float64]) -> None:
    """Raises ValueError if the arrays cannot be broadcast together."""
    try:
        np.broadcast_shapes(*(a.shape for a in arrays.values()))
    except ValueError as error:
        shapes = ", ".join(f"{k} {a.shape}" for k, a in arrays.items())
        raise ValueError(f"incompatible shapes: {shapes}") from error


def check_finite_result(
    result: npt.NDArray[np.float64], name: str
) -> npt.NDArray[np.float64]:
    """Raises OverflowError if a computed result is not finite."""
    if not np.all(np.isfinite(result)):
        raise OverflowError(
            f"{name} overflowed: the result exceeds the float64 range"
        )
    return result
