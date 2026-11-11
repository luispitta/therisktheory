"""Restricted safety measures for the unit-step ruin model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from therisktheory.finetti_ruin.adjustment import (
    unit_step_adjustment_coefficient,
)
from therisktheory.finetti_ruin.round_loading import round_safety_loading

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def safety_index(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the safety index in a favorable unit-step model."""
    return unit_step_adjustment_coefficient(win_probability)


def de_finetti_safety_loading(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate expected surplus gain per unit-step round."""
    return round_safety_loading(win_probability)
