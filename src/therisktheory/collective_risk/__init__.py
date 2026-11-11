"""Collective risk models, aggregate loss, and analytical approximations."""

from therisktheory.collective_risk.approximations import (
    edgeworth_approximation,
    normal_approximation,
    normal_power_approximation,
)

__all__ = [
    "edgeworth_approximation",
    "normal_approximation",
    "normal_power_approximation",
]
