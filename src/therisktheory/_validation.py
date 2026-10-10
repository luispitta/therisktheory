"""Input checks shared by the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_nonnegative_finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``value`` as a float64 array, raising ValueError if it is negative or not finite."""
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array
