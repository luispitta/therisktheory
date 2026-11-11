
"""TheRiskTheory actuarial package core."""

__version__ = "0.1.0"

from .solvency_pricing import (
    expected_aggregate_loss,
    expected_value_premium,
    pure_premium,
    guarantee_capital,
    solvency_ratio,
)
from .solvency_pricing.solvency import evaluate_portfolio_solvency

__all__ = [
    "__version__",
    "expected_aggregate_loss",
    "expected_value_premium",
    "pure_premium",
    "guarantee_capital",
    "solvency_ratio",
    "evaluate_portfolio_solvency",
]
