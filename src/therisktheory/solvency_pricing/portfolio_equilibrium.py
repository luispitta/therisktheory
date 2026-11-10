"""Portfolio equilibrium module for TheRiskTheory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory._validation import (
    as_nonnegative_finite_array,
    as_real_array,
    check_broadcastable,
    check_finite_result,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def portfolio_equilibrium(
    initial_capital: npt.ArrayLike,
    accumulated_profit: npt.ArrayLike,
    expected_claims: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Calculate the portfolio equilibrium surplus balance.

    Evaluates the net technical surplus incorporating initial capital,
    cumulative underwriting performance, and outstanding/expected losses:
        Surplus = initial_capital + accumulated_profit - expected_claims

    Args:
        initial_capital: Initial solvency capital or reserve (u >= 0).
        accumulated_profit: Cumulative underwriting result over horizon.
            Can be negative (technical deficits).
        expected_claims: Outstanding or expected claims to deduct.
            Defaults to 0.0 to prevent double counting if profit is net.

    Returns:
        np.ndarray: Net surplus balance. A negative value indicates ruin.

    Raises:
        TypeError: If inputs cannot be cast to floating-point numbers.
        ValueError: If initial_capital or expected_claims are negative.
    """
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")
    prof_arr = as_real_array(accumulated_profit, "accumulated_profit")
    claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")

    check_broadcastable(
        initial_capital=cap_arr,
        accumulated_profit=prof_arr,
        expected_claims=claims_arr,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = cap_arr + prof_arr - claims_arr

    return check_finite_result(result, "portfolio_equilibrium")
