"""Validation helpers for therisktheory."""
from __future__ import annotations

from typing import Any
import numpy as np


def validate_numeric_array(loss: Any) -> np.ndarray:
    """Validate that input is numeric array-like and does not contain invalid types or values."""
    if isinstance(loss, (str, bytes)):
        raise TypeError("Loss input cannot be a string.")
    try:
        arr = np.asarray(loss, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError("Loss input must be numeric.") from e
    
    if np.issubdtype(arr.dtype, np.object_):
        raise TypeError("Loss input contains non-numeric types.")
        
    if np.any(np.isnan(arr)):
        raise ValueError("Loss amounts cannot be NaN.")
    if np.any(np.isinf(arr)):
        raise ValueError("Loss amounts cannot be infinite.")
        
    return arr
