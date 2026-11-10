"""Demonstration of Topic 3: Monte Carlo simulation, stability, and reciprocity.

This script demonstrates:
1. Simulating compound aggregate losses S = sum_{i=1}^N X_i (Poisson + Lognormal).
2. Comparing Monte Carlo empirical moments against analytical expectations.
3. Determining optimal proportional reinsurance under variance stability constraints.
4. Evaluating technical result variance under various retention levels.
5. Demonstrating bilateral reciprocity between two independent insurer portfolios.
"""

from __future__ import annotations

import numpy as np

from therisktheory.severity_aggregate import (
    insurer_reciprocity_metric,
    optimal_reinsurance,
    simulate_aggregate_loss,
    technical_result_variance,
)


def main() -> None:
    print("=" * 75)
    print("TheRiskTheory — Topic 3: Monte Carlo Simulation and Stability Demo")
    print("=" * 75)

    # 1. Collective Risk Model Setup
    # Frequency: Poisson(lam=25)
    lam = 25.0
    # Severity: Lognormal(mean=5.0, sigma=0.6)
    mu_log = 5.0
    sig_log = 0.6
    # Theoretical individual severity moments
    mean_x = np.exp(mu_log + 0.5 * (sig_log**2))
    var_x = np.exp(2.0 * mu_log + (sig_log**2)) * (np.exp(sig_log**2) - 1.0)
    # Theoretical aggregate moments
    theor_e_s = lam * mean_x
    theor_var_s = lam * (var_x + mean_x**2)

    print("\n1. Theoretical Portfolio Moments:")
    print(f"   Expected frequency E[N]:       {lam:.1f}")
    print(f"   Expected individual severity E[X]: {mean_x:.2f}")
    print(f"   Theoretical aggregate E[S]:    {theor_e_s:.2f}")
    print(f"   Theoretical aggregate Var(S):  {theor_var_s:.2f}")

    # 2. Vectorized Monte Carlo Simulation
    n_sims = 25_000
    mc_losses = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="lognormal",
        num_simulations=n_sims,
        frequency_params={"lam": lam},
        severity_params={"mean": mu_log, "sigma": sig_log},
        random_state=42,
    )
    mc_mean = float(np.mean(mc_losses))
    mc_var = float(np.var(mc_losses, ddof=1))
    err_mean = abs(mc_mean - theor_e_s) / theor_e_s
    err_var = abs(mc_var - theor_var_s) / theor_var_s
    print(f"\n2. Monte Carlo Simulation ({n_sims:,} trajectories):")
    print(f"   Empirical Mean S_mc:          {mc_mean:.2f} (Rel Error: {err_mean:.2%})")
    print(f"   Empirical Variance Var(S_mc): {mc_var:.2f} (Rel Error: {err_var:.2%})")

    # 3. Optimal Reinsurance for Stability
    target_var = 0.40 * theor_var_s
    alpha_opt = optimal_reinsurance(
        expected_loss=theor_e_s,
        variance_loss=theor_var_s,
        reinsurer_loading=0.20,
        target_variance=target_var,
        criterion="target_variance",
    )
    print("\n3. Optimal Reinsurance (Target 60% Variance Reduction):")
    print(f"   Target Variance limit:         {target_var:.2f}")
    print(
        f"   Optimal Retention alpha*:      {float(alpha_opt):.4f} "
        f"(approx sqrt(0.40) = {np.sqrt(0.40):.4f})"
    )

    # 4. Technical Result Variance vs. Retention
    alphas = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    var_results = technical_result_variance(
        retention_rate=alphas,
        expected_frequency=lam,
        variance_frequency=lam,
        expected_severity=mean_x,
        variance_severity=var_x,
    )
    print("\n4. Retained Risk Variance across Retention Levels:")
    for a, v in zip(alphas, var_results, strict=True):
        print(
            f"   alpha = {a:.1f} -> Retained Technical Variance: {v:12.1f} ({v / theor_var_s:5.1%})"
        )

    # 5. Insurer Reciprocity Treaty (Two Independent Insurers)
    rec_results = insurer_reciprocity_metric(
        variance_1=theor_var_s,
        variance_2=1.5 * theor_var_s,
        expected_loss_1=theor_e_s,
        expected_loss_2=1.5 * theor_e_s,
        quota_1=0.40,
        correlation=0.0,
    )
    print("\n5. Bilateral Reciprocity Exchange Evaluation:")
    print("   Insurer 1 Cession Quota q1:     40.0%")
    print(f"   Insurer 2 Balanced Quota q2:    {float(rec_results['quota_2']):.1%}")
    print(f"   Insurer 1 Variance Reduction:   {float(rec_results['variance_reduction_1']):.2%}")
    print(f"   Insurer 2 Variance Reduction:   {float(rec_results['variance_reduction_2']):.2%}")
    print(f"   Total Reciprocity System Gain:  {float(rec_results['total_reciprocity_gain']):.2%}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
