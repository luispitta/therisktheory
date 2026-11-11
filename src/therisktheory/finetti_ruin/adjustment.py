"""Adjustment coefficient for favorable unit-step risk games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def unit_step_adjustment_coefficient(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the adjustment coefficient for a unit-step game."""
    p = _validation.as_probability_array(win_probability, "win_probability")
    if np.any(p < 0.5):
        raise ValueError("win_probability must be at least 0.5")
    if np.any(p == 1.0):
        raise ValueError(
            "win_probability must be below 1 for a finite coefficient"
        )

    result = np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)
    return _validation.check_finite_result(
        result, "unit_step_adjustment_coefficient"
    )
