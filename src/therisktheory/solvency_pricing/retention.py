"""Risk retention, retention capacity, and inflation adjustment functions.
Reference:
    Beard, R. E., Pentikäinen, T., and Pesonen, E. (1984). Risk Theory.
    Panjer, H. H. and Willmot, G. E. (1992). Insurance Risk Models.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory import _validation


def retained_loss(
    loss: ArrayLike,
    retention_limit: float
) -> NDArray[np.float64]:
    """Calculate the retained loss by an insurer given a retention limit."""
    if isinstance(loss, (str, bytes)):
        raise TypeError("Loss input cannot be a string.")
    
    loss_arr = _validation.validate_numeric_array(loss)
    
    if not isinstance(retention_limit, (int, float, np.number)):
        raise TypeError("Retention limit must be numeric.")
    if retention_limit < 0:
        raise ValueError("Retention limit must be non-negative.")
    if np.any(loss_arr < 0):
        raise ValueError("Loss amounts cannot be negative.")
        
    return np.minimum(loss_arr, float(retention_limit))


def retention_capacity(
    surplus: float,
    maximum_retention_ratio: float
) -> float:
    """Determine the financial retention capacity based on insurer surplus."""
    if not isinstance(surplus, (int, float, np.number)) or not isinstance(maximum_retention_ratio, (int, float, np.number)):
        raise TypeError("Surplus and retention ratio must be numeric.")
    if surplus < 0 or not (0.0 <= maximum_retention_ratio <= 1.0):
        raise ValueError("Invalid surplus or retention ratio.")
        
    return float(surplus * maximum_retention_ratio)


def inflate_losses(
    loss: ArrayLike,
    inflation_rate: float,
    periods: int = 1
) -> NDArray[np.float64]:
    """Project the impact of inflation on risk losses over time."""
    if isinstance(loss, (str, bytes)):
        raise TypeError("Loss input cannot be a string.")
    loss_arr = _validation.validate_numeric_array(loss)
    
    if not isinstance(inflation_rate, (int, float, np.number)):
        raise TypeError("Inflation rate must be numeric.")
    if not isinstance(periods, int) and not (isinstance(periods, np.number) and np.issubdtype(type(periods), np.integer)):
        raise TypeError("Periods must be an integer.")
    if inflation_rate < -1.0 or periods < 0:
        raise ValueError("Invalid inflation rate or periods.")
        
    return loss_arr * ((1.0 + float(inflation_rate)) ** int(periods))


def deflate_losses(
    inflated_loss: ArrayLike,
    inflation_rate: float,
    periods: int = 1
) -> NDArray[np.float64]:
    """Discount the effect of inflation to bring losses back to present value."""
    if isinstance(inflated_loss, (str, bytes)):
        raise TypeError("Loss input cannot be a string.")
    loss_arr = _validation.validate_numeric_array(inflated_loss)
    
    if not isinstance(inflation_rate, (int, float, np.number)):
        raise TypeError("Inflation rate must be numeric.")
    if not isinstance(periods, int) and not (isinstance(periods, np.number) and np.issubdtype(type(periods), np.integer)):
        raise TypeError("Periods must be an integer.")
        
    if inflation_rate == -1.0 and periods > 0:
        raise ValueError("Division by zero: inflation rate cannot be -1.0 with positive periods.")
    if inflation_rate < -1.0 or periods < 0:
        raise ValueError("Invalid inflation rate or periods.")
        
    return loss_arr / ((1.0 + float(inflation_rate)) ** int(periods))
