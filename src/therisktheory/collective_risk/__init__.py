"""Collective risk theory, Lundberg fundamental equation, and safety equation."""

from .safety_equation import (
    adjustment_coefficient,
    full_retention_adjustment,
    lundberg_equation,
)

__all__ = [
    "adjustment_coefficient",
    "lundberg_equation",
    "full_retention_adjustment",
]
