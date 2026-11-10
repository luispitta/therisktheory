"""Demonstration script for reinsurance effects on the collective risk process.

This script illustrates:
1. Proportional reinsurance surplus process parameters and net drift.
2. Excess-of-Loss reinsurance moments and retained premium rates.
3. Lundberg adjustment coefficient behavior under varying retention levels.
4. Identification of optimal reinsurance retention maximizing R (both Proportional and XL).
5. Ruin probability attenuation and Lundberg upper bounds across reserve levels.
6. Solvency capital relief under target ruin probability tolerance.
7. Comparative synthesis: De Finetti safety index vs. Lundberg adjustment coefficient.

References:
    Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Chapter 4.
    Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Chapter 14.
"""

from __future__ import annotations

import numpy as np

from therisktheory.collective_risk import (
    compare_de_finetti_lundberg_retention,
    excess_of_loss_reinsurance_process,
    optimal_reinsurance_retention,
    proportional_reinsurance_process,
    reinsurance_adjustment_coefficient,
    reinsurance_capital_relief,
    reinsurance_ruin_bound,
)


def run_topic_5_demo() -> None:
    """Execute complete collective risk and reinsurance effects demonstration."""
    print("=" * 78)
    print("TOPIC 5: REINSURANCE EFFECTS ON THE COLLECTIVE RISK PROCESS & RUIN THEORY")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # 1. Proportional Reinsurance Process
    # ------------------------------------------------------------------------
    claim_rate = 100.0  # lambda = 100 claims/year
    claim_mean = 2.0  # E[X] = 2.0
    claim_var = 4.0  # Var(X) = 4.0 (Exponential(0.5))
    safety_loading = 0.25  # Insurer loading theta = 25%
    reinsurer_loading = 0.20  # Reinsurer loading xi = 20%
    retention_alpha = 0.75  # Retaining 75% of claims

    prop_res = proportional_reinsurance_process(
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        claim_variance=claim_var,
        safety_loading=safety_loading,
        reinsurer_loading=reinsurer_loading,
        retention_rate=retention_alpha,
    )

    gross_premium = claim_rate * claim_mean * (1.0 + safety_loading)
    print("\n--- 1. Proportional Reinsurance Surplus Process ---")
    print(f"Gross Premium Rate c:              {gross_premium:.2f}")
    print(f"Net Retained Premium Rate c_ret:   {prop_res['retained_premium_rate']:.2f}")
    print(f"Retained Claim Mean mu_1,ret:      {prop_res['retained_claim_mean']:.2f}")
    print(f"Retained Variance sigma_ret^2:     {prop_res['retained_claim_variance']:.2f}")
    print(f"Retained Relative Loading theta:   {prop_res['retained_safety_loading']:.4f}")
    print(f"Expected Annual Net Drift:         {prop_res['expected_annual_drift']:.2f}")
    print(f"Minimum Viable Retention alpha_min:{prop_res['minimum_viable_retention']:.4f}")

    # ------------------------------------------------------------------------
    # 2. Excess-of-Loss Reinsurance Process
    # ------------------------------------------------------------------------
    priority_d = 4.0  # Cedent retains losses up to d = 4.0

    xl_res = excess_of_loss_reinsurance_process(
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        priority=priority_d,
        safety_loading=safety_loading,
        reinsurer_loading=reinsurer_loading,
        distribution="exponential",
    )

    print("\n--- 2. Excess-of-Loss Reinsurance Surplus Process (Priority d = 4.0) ---")
    print(f"Net Retained Premium Rate c_ret:   {xl_res['retained_premium_rate']:.2f}")
    print(f"Expected Retained Severity:        {xl_res['retained_claim_mean']:.4f}")
    print(f"Expected Ceded Severity:           {xl_res['ceded_claim_mean']:.4f}")
    print(f"Retained Severity Variance:        {xl_res['retained_claim_variance']:.4f}")
    print(f"Retained Relative Loading theta:   {xl_res['retained_safety_loading']:.4f}")
    print(f"Expected Annual Net Drift:         {xl_res['expected_annual_drift']:.2f}")

    # ------------------------------------------------------------------------
    # 3. Lundberg Adjustment Coefficient Profile
    # ------------------------------------------------------------------------
    retention_grid = np.linspace(0.40, 1.0, 7)
    r_coefficients = reinsurance_adjustment_coefficient(
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        safety_loading=safety_loading,
        reinsurer_loading=reinsurer_loading,
        retention_param=retention_grid,
        treaty_type="proportional",
    )

    print("\n--- 3. Lundberg Adjustment Coefficient Curve R(alpha) ---")
    for a_val, r_val in zip(retention_grid, r_coefficients, strict=False):
        print(f"  Retention alpha = {a_val:.2f}  -->  Adjustment Coeff R = {r_val:.5f}")

    # Exact Lundberg equation residue check (Prof. Pitta's verification case)
    r_check = float(
        np.asarray(
            reinsurance_adjustment_coefficient(
                claim_rate=1.0,
                claim_mean=1.0,
                safety_loading=0.20,
                reinsurer_loading=0.30,
                retention_param=0.70,
                treaty_type="proportional",
            )
        ).item()
    )
    c_ret_check = 1.0 * (0.70 + 0.20 - 0.30 * 0.30)
    mgf_check = 1.0 / (1.0 - 0.70 * r_check)
    residue_check = 1.0 * (mgf_check - 1.0) - c_ret_check * r_check
    print(f"\n  [Residue check at alpha=0.70]: R = {r_check:.9f}, Residue = {residue_check:.2e}")

    # ------------------------------------------------------------------------
    # 4. Optimal Reinsurance Retention (Excess of Loss and Proportional)
    # ------------------------------------------------------------------------
    opt_xl = optimal_reinsurance_retention(
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        safety_loading=0.20,
        reinsurer_loading=0.25,
        treaty_type="excess_of_loss",
    )

    print("\n--- 4. Optimal Excess-of-Loss Reinsurance Retention ---")
    print(f"Optimal Priority d*:               {opt_xl['optimal_retention']:.4f}")
    print(f"Optimal Adjustment Coeff R*:       {opt_xl['optimal_adjustment_coefficient']:.5f}")
    print(f"Gross Adjustment Coeff R_gross:    {opt_xl['gross_adjustment_coefficient']:.5f}")
    print(f"Relative Solvency Gain:            {opt_xl['relative_adjustment_gain'] * 100.0:.2f}%")

    # ------------------------------------------------------------------------
    # 5. Ruin Probabilities & Lundberg Bounds
    # ------------------------------------------------------------------------
    initial_reserves = np.array([0.0, 5.0, 10.0, 20.0, 35.0, 50.0])
    ruin_eval = reinsurance_ruin_bound(
        initial_capital=initial_reserves,
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        safety_loading=safety_loading,
        reinsurer_loading=reinsurer_loading,
        retention_rate=retention_alpha,
        method="cramer_lundberg_exact",
    )

    print(f"\n--- 5. Ruin Probabilities: Gross vs Reinsured (alpha = {retention_alpha}) ---")
    print("  Initial u | Gross Ruin psi(u) | Reinsured psi_re(u) | Ruin Reduction Ratio")
    print("  " + "-" * 68)
    for u_val, psi_g, psi_re, red in zip(
        initial_reserves,
        ruin_eval["gross_ruin_probability"],
        ruin_eval["reinsured_ruin_probability"],
        ruin_eval["ruin_reduction_ratio"],
        strict=False,
    ):
        print(f"  {u_val:9.1f} | {psi_g:17.6f} | {psi_re:19.6f} | {red * 100.0:18.2f}%")

    # ------------------------------------------------------------------------
    # 6. Solvency Capital Relief
    # ------------------------------------------------------------------------
    solvency_tolerance = 0.01  # Target ruin probability 1%
    cap_relief = reinsurance_capital_relief(
        ruin_tolerance=solvency_tolerance,
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        safety_loading=safety_loading,
        reinsurer_loading=reinsurer_loading,
        retention_rate=retention_alpha,
    )

    print(f"\n--- 6. Solvency Capital Relief (Target Tolerance = {solvency_tolerance:.1%}) ---")
    print(f"Gross Required Capital:            {cap_relief['gross_required_capital']:.2f}")
    print(f"Reinsured Required Capital:        {cap_relief['reinsured_required_capital']:.2f}")
    print(f"Absolute Capital Relief:           {cap_relief['capital_relief']:.2f}")
    print(f"Percentage Capital Relief:         {cap_relief['percentage_capital_relief']:.2f}%")

    # ------------------------------------------------------------------------
    # 7. Comparison: De Finetti vs Lundberg Retention
    # ------------------------------------------------------------------------
    comparison = compare_de_finetti_lundberg_retention(
        claim_mean=claim_mean,
        claim_variance=claim_var,
        safety_loading=0.20,
        reinsurer_loading=0.30,
        claim_rate=claim_rate,
    )

    print("\n--- 7. De Finetti Safety Index vs Lundberg Adjustment Coefficient ---")
    print(f"De Finetti Optimal Retention:      {comparison['de_finetti_optimal_retention']:.4f}")
    print(f"Lundberg Optimal Retention:        {comparison['lundberg_optimal_retention']:.4f}")
    print(f"Difference (DF - Lundberg):        {comparison['retention_difference']:.4f}")
    print(f"De Finetti Safety Index:           {comparison['de_finetti_safety_index']:.5f}")
    print(f"Lundberg Adjustment Coeff R*:      {comparison['lundberg_adjustment_coefficient']:.5f}")
    print(f"Analytical Insight:                {comparison['commentary']}")
    print("=" * 78)


if __name__ == "__main__":
    run_topic_5_demo()
