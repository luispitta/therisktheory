"""Round safety loading calculation for unit-step risk games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def round_safety_loading(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the expected gain per round for a unit-step game."""
    p = _validation.as_probability_array(win_probability, "win_probability")
    if np.any(p < 0.5):
        raise ValueError("win_probability must be at least 0.5")

    result = np.asarray(2.0 * p - 1.0, dtype=np.float64)
    return _validation.check_finite_result(result, "round_safety_loading")
