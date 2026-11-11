"""Expected profit module for TheRiskTheory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def expected_profit(
    premium: npt.ArrayLike,
    expected_claims: npt.ArrayLike,
    expenses: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Calculate expected underwriting profit of an insurance portfolio.

    Under classical risk theory, expected underwriting profit is defined as
    the excess of earned premiums over expected aggregate claims and
    operational expenses:
        E[Pi] = Premium - E[Claims] - Expenses

    Args:
        premium: Total premium collected or loaded premium (P >= 0).
        expected_claims: Expected aggregate claims or losses (E[S] >= 0).
        expenses: Acquisition or administrative expenses (E >= 0).
            Defaults to 0.0.

    Returns:
        np.ndarray: Expected underwriting profit. Can be negative.

    Raises:
        TypeError: If inputs cannot be cast to floating-point numbers.
        ValueError: If inputs are negative, infinite, or shapes mismatch.
    """
    prem_arr = as_nonnegative_finite_array(premium, "premium")
    claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")
    exp_arr = as_nonnegative_finite_array(expenses, "expenses")

    check_broadcastable(
        premium=prem_arr,
        expected_claims=claims_arr,
        expenses=exp_arr,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = prem_arr - claims_arr - exp_arr

    return check_finite_result(result, "expected_profit")
