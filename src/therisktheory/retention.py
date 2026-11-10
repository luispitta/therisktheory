git add .
git commit -m "FEEDBACK"
    
"""Risk retention, division, and inflation adjustment models in actuarial science."""

from __future__ import annotations
from typing import Union
import numpy as np
import numpy.typing as npt

NUMERIC_KINDS = {"i", "u", "f"}

def _validate_numeric_array(value: Union[float, int, npt.ArrayLike], name: str) -> npt.NDArray[np.float64]:
    """Validate that input is a finite, non-negative numeric value or array."""
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as error:
        raise TypeError(f"{name} must be a real number or array") from error

    if array.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"{name} must be a numeric type, got dtype {array.dtype}")

    array = array.astype(np.float64, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite (no NaN or infinity allowed)")

    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")

    return array

def _check_finite_result(result: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64]:
    """Raise OverflowError if a computed result exceeds float64 range."""
    if not np.all(np.isfinite(result)):
        raise OverflowError(f"{name} overflowed: the result exceeds the float64 range")
    return result

def retained_loss(loss: Union[float, int, npt.ArrayLike], retention_limit: float) -> npt.NDArray[np.float64]:
    """Calculate the retained loss by an insurer given a retention limit.

    Args:
        loss: Loss amount(s) incurred (must be non-negative and finite).
        retention_limit: Maximum retention limit per risk or event (must be non-negative and finite).

    Returns:
        The portion of the loss retained by the insurer as a float64 array.

    References:
        Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer.
    """
    loss_arr = _validate_numeric_array(loss, "loss")
    limit_arr = _validate_numeric_array(retention_limit, "retention_limit")
    
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.minimum(loss_arr, limit_arr)
        except FloatingPointError as e:
            raise OverflowError("Numerical overflow in retained_loss") from e
            
    return _check_finite_result(result, "retained_loss")

def retention_capacity(surplus: float, maximum_retention_ratio: float) -> float | npt.NDArray[np.float64]:
    """Determine the financial retention capacity based on insurer surplus.

    Args:
        surplus: Insurer's available solvency margin or surplus.
        maximum_retention_ratio: Maximum proportion of surplus allowed for retention (between 0 and 1).

    Returns:
        Maximum retention capacity amount.

    References:
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory. Chapman and Hall.
    """
    surplus_arr = _validate_numeric_array(surplus, "surplus")
    ratio_arr = _validate_numeric_array(maximum_retention_ratio, "maximum_retention_ratio")

    if np.any(ratio_arr > 1.0):
        raise ValueError("maximum_retention_ratio must be between 0.0 and 1.0")

    with np.errstate(over="raise", invalid="raise"):
        try:
            result = surplus_arr * ratio_arr
        except FloatingPointError as e:
            raise OverflowError("Numerical overflow in retention_capacity") from e

    return _check_finite_result(result, "retention_capacity")

def inflate_losses(
    loss: Union[float, int, npt.ArrayLike], 
    inflation_rate: float, 
    periods: int = 1
) -> npt.NDArray[np.float64]:
    """Project the impact of inflation on risk losses over time.

    Args:
        loss: Base loss amount(s).
        inflation_rate: Periodic inflation rate (must be >= -1.0).
        periods: Number of periods (must be a non-negative integer).

    Returns:
        Inflated loss amount(s).

    References:
        Panjer, H., & Willmot, G. (1992). Insurance Risk Models. Society of Actuaries.
    """
    loss_arr = _validate_numeric_array(loss, "loss")
    
    if not isinstance(periods, (int, np.integer)) or periods < 0:
        raise TypeError("periods must be a non-negative integer")

    if inflation_rate < -1.0:
        raise ValueError("inflation_rate cannot be less than -1.0 (-100%)")

    with np.errstate(over="raise", invalid="raise"):
        try:
            factor = (1.0 + inflation_rate) ** periods
            result = loss_arr * factor
        except FloatingPointError as e:
            raise OverflowError("Numerical overflow in inflate_losses") from e

    return _check_finite_result(result, "inflate_losses")

def deflate_losses(
    inflated_loss: Union[float, int, npt.ArrayLike], 
    inflation_rate: float, 
    periods: int = 1
) -> npt.NDArray[np.float64]:
    """Discount the effect of inflation to bring losses back to present value.

    Args:
        inflated_loss: Inflated loss amount(s).
        inflation_rate: Periodic inflation rate used for discounting (must be >= -1.0).
        periods: Number of periods (must be a non-negative integer).

    Returns:
        Deflated (present value) loss amount(s).

    References:
        Panjer, H., & Willmot, G. (1992). Insurance Risk Models. Society of Actuaries.
    """
    loss_arr = _validate_numeric_array(inflated_loss, "inflated_loss")

    if not isinstance(periods, (int, np.integer)) or periods < 0:
        raise TypeError("periods must be a non-negative integer")

    if inflation_rate < -1.0:
        raise ValueError("inflation_rate cannot be less than -1.0 (-100%)")

    if inflation_rate == -1.0 and periods > 0:
        raise ValueError("Division by zero: inflation_rate of -1.0 with positive periods is undefined for deflation.")

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            factor = (1.0 + inflation_rate) ** periods
            if factor == 0.0:
                raise ValueError("Division by zero in deflation factor.")
            result = loss_arr / factor
        except (FloatingPointError, ZeroDivisionError) as e:
            raise ValueError("Division by zero or numerical error in deflate_losses") from e

    return _check_finite_result(result, "deflate_losses")
