"""Proxy module for risk retention to support top-level imports."""
from __future__ import annotations

from therisktheory.solvency_pricing.retention import (
    retained_loss,
    retention_capacity,
    inflate_losses,
    deflate_losses,
)

__all__ = [
    "retained_loss",
    "retention_capacity",
    "inflate_losses",
    "deflate_losses",
]
