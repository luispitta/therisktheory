"""Integrated Topic 3 example: Poisson frequency and lognormal severity.

Requires the frequency and severity modules from the peer branches
to be integrated into the project.
"""

import numpy as np

from therisktheory.collective_risk.aggregate_loss import (
    aggregate_loss_moments,
)
from therisktheory.frequency import (
    poisson_count_moments,
    poisson_count_sample,
)
from therisktheory.severity_aggregate import (
    lognormal_severity_moments,
)


def main() -> None:
    """Compare theoretical and simulated aggregate loss moments."""
    rate = 2.0
    mu = np.log(100.0)
    sigma = 0.5
    n_periods = 100_000
    seed = 2026

    # 1. Theoretical frequency moments.
    frequency = poisson_count_moments(rate=rate)

    # 2. Theoretical severity moments.
    severity = lognormal_severity_moments(
        mu=mu,
        sigma=sigma,
    )

    # 3. Theoretical aggregate loss moments.
    mean_s, variance_s, std_s = aggregate_loss_moments(
        expected_frequency=frequency["mean"],
        variance_frequency=frequency["variance"],
        expected_severity=severity["mean"],
        variance_severity=severity["variance"],
    )

    # 4. Simulate claim counts independently.
    counts = poisson_count_sample(
        rate=rate,
        sample_shape=n_periods,
        rng=seed,
    )

    # 5. Simulate one lognormal severity per claim.
    severity_rng = np.random.default_rng(seed + 1)
    total_claims = int(np.sum(counts))

    claim_amounts = severity_rng.lognormal(
        mean=mu,
        sigma=sigma,
        size=total_claims,
    )

    # 6. Aggregate claim amounts by observation period.
    period_ids = np.repeat(np.arange(n_periods), counts)

    aggregate_losses = np.bincount(
        period_ids,
        weights=claim_amounts,
        minlength=n_periods,
    )

    # 7. Compare theoretical and empirical results.
    simulated_mean = np.mean(aggregate_losses)
    simulated_variance = np.var(aggregate_losses)

    print("COLLECTIVE RISK MODEL")
    print("--------------------")
    print(f"Poisson rate: {rate}")
    print(f"Lognormal mu: {mu:.6f}")
    print(f"Lognormal sigma: {sigma}")
    print(f"Simulation periods: {n_periods}")

    print("\nTHEORETICAL RESULTS")
    print(f"Aggregate mean: {float(mean_s):.4f}")
    print(f"Aggregate variance: {float(variance_s):.4f}")
    print(f"Aggregate standard deviation: {float(std_s):.4f}")

    print("\nSIMULATION RESULTS")
    print(f"Empirical mean: {simulated_mean:.4f}")
    print(f"Empirical variance: {simulated_variance:.4f}")

    print("\nABSOLUTE DIFFERENCES")
    print(f"Mean difference: {abs(simulated_mean - float(mean_s)):.4f}")
    print(
        "Variance difference: "
        f"{abs(simulated_variance - float(variance_s)):.4f}"
    )


if __name__ == "__main__":
    main()
