
"""De Finetti reserve funds and surplus trajectory models.

Topic 2: De Finetti, Games, and Ruin.
"""

import numpy as np

from .._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def de_finetti_premium(
    expected_severity: float | np.ndarray,
    safety_loading: float | np.ndarray,
) -> np.ndarray:
    """Calculate the premium using the expected value principle.

    The calculation follows P = E[X] * (1 + theta).

    Args:
        expected_severity: Expected claim amount. Must be non-negative
            and finite.
        safety_loading: Safety loading parameter. Must be non-negative
            and finite.

    Returns:
        A float64 array containing the calculated premium.

    Raises:
        TypeError: If an input is not numeric.
        ValueError: If an input is negative, non-finite, or has an
            incompatible shape.
        OverflowError: If the result exceeds the float64 range.
    """
    severity = as_nonnegative_finite_array(
        expected_severity, "expected_severity"
    )
    loading = as_nonnegative_finite_array(
        safety_loading, "safety_loading"
    )

    check_broadcastable(
        expected_severity=severity,
        safety_loading=loading,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(
            severity * (1.0 + loading),
            dtype=np.float64,
        )

    return check_finite_result(result, "De Finetti premium")


def de_finetti_profit(
    premiums: np.ndarray,
    claims: np.ndarray,
) -> np.ndarray:
    """Calculate periodic profit as premiums minus claims.

    Negative profit values are allowed.

    Args:
        premiums: Premium income amounts. Must be non-negative and finite.
        claims: Claim payment amounts. Must be non-negative and finite.

    Returns:
        A float64 array containing periodic profit.

    Raises:
        TypeError: If an input is not numeric.
        ValueError: If an input is negative, non-finite, or has an
            incompatible shape.
        OverflowError: If the result exceeds the float64 range.
    """
    premium_array = as_nonnegative_finite_array(
        premiums, "premiums"
    )
    claim_array = as_nonnegative_finite_array(
        claims, "claims"
    )

    check_broadcastable(
        premiums=premium_array,
        claims=claim_array,
    )

    result = np.asarray(
        premium_array - claim_array,
        dtype=np.float64,
    )

    return check_finite_result(result, "De Finetti profit")


def reserve_fund(
    initial_reserve: float,
    premiums: np.ndarray,
    claims: np.ndarray,
) -> np.ndarray:
    """Calculate the accumulated reserve after each period.

    The reserve follows U_t = U_0 + sum(P_k - C_k), for k = 1, ..., t.
    Negative reserve states are retained to allow ruin analysis.

    Premium and claim sequences must be one-dimensional and have
    exactly the same length. The initial reserve must be scalar.

    The returned trajectory contains the reserve after each period;
    the initial reserve U_0 is not included as a separate first element.

    Args:
        initial_reserve: Non-negative scalar initial capital U_0.
        premiums: Non-negative one-dimensional premium sequence.
        claims: Non-negative one-dimensional claim sequence.

    Returns:
        A float64 array containing the reserve after each period.

    Raises:
        TypeError: If an input is not numeric.
        ValueError: If an input is negative, non-finite, non-scalar,
            multidimensional, or has an incompatible length.
        OverflowError: If the reserve calculation exceeds float64 range.
    """
    initial = as_nonnegative_finite_array(
        initial_reserve, "initial_reserve"
    )
    premium_array = as_nonnegative_finite_array(
        premiums, "premiums"
    )
    claim_array = as_nonnegative_finite_array(
        claims, "claims"
    )

    if initial.ndim != 0:
        raise ValueError("initial_reserve must be a scalar")

    if premium_array.ndim != 1:
        raise ValueError("premiums must be one-dimensional")

    if claim_array.ndim != 1:
        raise ValueError("claims must be one-dimensional")

    if premium_array.shape != claim_array.shape:
        raise ValueError(
            "premiums and claims must have the same length"
        )

    periodic_profit = de_finetti_profit(
        premium_array,
        claim_array,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        cumulative_profit = np.cumsum(periodic_profit)
        trajectory = np.asarray(
            initial + cumulative_profit,
            dtype=np.float64,
        )

    return check_finite_result(trajectory, "Reserve fund")
