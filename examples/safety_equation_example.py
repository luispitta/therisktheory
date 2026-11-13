"""Example demonstrating Lundberg's fundamental equation and adjustment coefficient."""

from __future__ import annotations

import numpy as np

from therisktheory.collective_risk import (
    adjustment_coefficient,
    full_retention_adjustment,
)

try:
    from therisktheory.severity_aggregate import lognormal_severity_moments
except ImportError:
    lognormal_severity_moments = None


def _lognormal_moments(mu: float, sigma: float) -> tuple[float, float]:
    """Use Garcia's lognormal_severity_moments if available, else analytical formula."""
    if lognormal_severity_moments is not None:
        moments = lognormal_severity_moments(mu=mu, sigma=sigma)
        return float(moments["mean"].item()), float(moments["second_raw_moment"].item())
    mean_x = float(np.exp(mu + 0.5 * sigma**2))
    second_moment_x = float(np.exp(2.0 * mu + 2.0 * sigma**2))
    return mean_x, second_moment_x


def run_example() -> None:
    """Run full-retention approximation and exact Lundberg root example."""
    print("=== TheRiskTheory: Topic 5 - Collective Risk & Safety Equation ===")

    mu_log = 1.0
    sigma_log = 0.5
    mean_severity, second_moment = _lognormal_moments(mu=mu_log, sigma=sigma_log)
    safety_loading = 0.20

    print(f"Lognormal Severity: mu={mu_log}, sigma={sigma_log}")
    print(f"Mean Severity E[X]: {mean_severity:.4f}")
    print(f"Second Moment E[X^2]: {second_moment:.4f}")
    print(f"Safety Loading lambda: {safety_loading:.2%}\n")

    r_approx = full_retention_adjustment((mean_severity, second_moment), safety_loading)
    print(f"Full Retention Approximation R (moments): {r_approx:.6f}")

    # Lognormal has no finite MGF for r > 0; exact Lundberg root uses Exponential(mu)
    # whose MGF domain of existence is r < mu.
    mu_exp = 1.0 / mean_severity

    def mgf_exp(r: float) -> float:
        if r >= mu_exp:
            return float("inf")
        return mu_exp / (mu_exp - r)

    r_exact = adjustment_coefficient(
        mgf_exp,
        mean_severity,
        safety_loading,
        r_min=1e-9,
        r_max=mu_exp - 1e-3,
    )
    print(f"Exact Lundberg R (Exponential, domain r < {mu_exp:.4f}): {r_exact:.6f}")


if __name__ == "__main__":
    run_example()
