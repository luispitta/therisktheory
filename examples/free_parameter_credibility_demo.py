"""Demonstration of free-parameter credibility, claim reserving, and diagnostics.

This script illustrates:
1. Non-parametric empirical Bayes estimation (Bühlmann model on benchmark data).
2. Weighted Bühlmann-Straub credibility with unequal policyholder exposures.
3. Actuarial claim reserving blending development factors with prior expectations.
4. Statistical diagnostics, portfolio balance verification, and prediction error analysis.
"""

from __future__ import annotations

import numpy as np

from therisktheory.credibility import (
    credibility_reserve_estimate,
    free_parameter_credibility,
    reserve_model_diagnostics,
)


def main() -> None:
    print("=" * 74)
    print("TheRiskTheory — Topic 4: Free-Parameter Credibility & Loss Reserving Demo")
    print("=" * 74)

    # -------------------------------------------------------------------------
    # 1. Classical Bühlmann Model (Kaas et al. 2008 Benchmark: 3 groups, 5 years)
    # -------------------------------------------------------------------------
    print("\n1. Free-Parameter Credibility (Balanced Bühlmann Model):")
    kaas_data = np.array(
        [
            [99.3, 93.7, 103.9, 92.5, 110.6],
            [112.5, 108.3, 118.0, 99.4, 111.8],
            [129.2, 140.9, 108.3, 105.0, 116.6],
        ]
    )

    res_kaas = free_parameter_credibility(kaas_data)
    print(f"   Overall Sample Mean:            {res_kaas['overall_sample_mean']:.2f}")
    print(f"   Collective Grand Mean (mu_Z):   {res_kaas['collective_mean']:.2f}")
    print(f"   Within-Risk Variance (s^2):     {res_kaas['process_variance']:.2f}")
    print(f"   Between-Risk Variance (a):      {res_kaas['variance_hypothetical_means']:.2f}")
    print(f"   Bühlmann Parameter Ratio (k):   {res_kaas['k_ratio']:.4f}")
    print(f"   Credibility Weight (Z):         {float(res_kaas['credibility_factors'][0]):.4f}")

    print("\n   Individual Risk Experience vs. Credibility Premiums:")
    for i, (m, p) in enumerate(
        zip(res_kaas["individual_means"], res_kaas["credibility_premiums"], strict=True),
        start=1,
    ):
        print(f"   Group {i}: Sample Mean = {m:6.2f} -> Credibility Premium = {p:6.2f}")

    # -------------------------------------------------------------------------
    # 2. Weighted Bühlmann-Straub Model (Unequal Policyholder Exposures)
    # -------------------------------------------------------------------------
    print("\n2. Weighted Bühlmann-Straub Credibility (Unequal Fleet Sizes):")
    loss_ratios = np.array(
        [
            [0.65, 0.70, 0.60],  # Commercial Fleet 1
            [0.85, 0.90, 0.88],  # Commercial Fleet 2
            [0.55, 0.50, 0.52],  # Commercial Fleet 3
        ]
    )
    fleet_weights = np.array(
        [
            [150.0, 180.0, 200.0],  # Fleet 1 exposure
            [500.0, 520.0, 510.0],  # Fleet 2 (large) exposure
            [80.0, 90.0, 100.0],  # Fleet 3 (small) exposure
        ]
    )

    res_bs = free_parameter_credibility(loss_ratios, weights=fleet_weights)
    print(f"   Collective Loss Ratio (mu_Z):   {res_bs['collective_mean']:.4f}")
    print(f"   Process Variance (s^2):         {res_bs['process_variance']:.6f}")
    print(f"   Hypothetical Mean Variance (a): {res_bs['variance_hypothetical_means']:.6f}")

    print("\n   Fleet Volume, Credibility Weights & Target Loss Ratios:")
    for i, (w_tot, z, x_bar, prem) in enumerate(
        zip(
            res_bs["total_weights"],
            res_bs["credibility_factors"],
            res_bs["individual_means"],
            res_bs["credibility_premiums"],
            strict=True,
        ),
        start=1,
    ):
        print(
            f"   Fleet {i}: Exposure = {w_tot:6.0f} | Z = {z:.4f} | "
            f"Obs LR = {x_bar:.4f} -> Cred LR = {prem:.4f}"
        )

    # -------------------------------------------------------------------------
    # 3. Credibility Claim Reserving (Bornhuetter-Ferguson Bridge)
    # -------------------------------------------------------------------------
    print("\n3. Credibility Claim Reserving & IBNR Estimation:")
    accident_years = np.array([2023, 2024, 2025])
    reported = np.array([850000.0, 620000.0, 250000.0])
    earned_premiums = np.array([1200000.0, 1300000.0, 1400000.0])
    expected_ultimates = earned_premiums * 0.70  # Pricing ELR = 70%
    dev_factors = np.array([1.10, 1.45, 2.50])  # Loss Development Factors

    res_reserves = credibility_reserve_estimate(
        reported_claims=reported,
        expected_claims=expected_ultimates,
        development_factor=dev_factors,
        premiums=earned_premiums,
    )

    header = (
        f"   {'Year':<6} {'Reported':>10} {'Prior Ult':>11} {'CDF':>6} "
        f"{'Z (Emerg)':>10} {'Cred Ult':>11} {'IBNR Reserve':>13}"
    )
    print(header)
    print("   " + "-" * 70)
    for yr, rep, exp, f, z, ult, rsv in zip(
        accident_years,
        reported,
        expected_ultimates,
        dev_factors,
        res_reserves["credibility_factors"],
        res_reserves["ultimate_claims"],
        res_reserves["reserve"],
        strict=True,
    ):
        print(f"   {yr:<6} {rep:10.0f} {exp:11.0f} {f:6.2f} {z:10.4f} {ult:11.0f} {rsv:13.0f}")

    # -------------------------------------------------------------------------
    # 4. Statistical Model Diagnostics
    # -------------------------------------------------------------------------
    print("\n4. Statistical Diagnostics & Shrinkage Evaluation:")
    diag = reserve_model_diagnostics(
        observed_values=res_bs["individual_means"],
        credibility_estimates=res_bs["credibility_premiums"],
        collective_mean=res_bs["collective_mean"],
        credibility_factors=res_bs["credibility_factors"],
        weights=res_bs["total_weights"],
        process_variance=res_bs["process_variance"],
        hypothetical_mean_variance=res_bs["variance_hypothetical_means"],
    )

    print(
        f"   Portfolio Imbalance:            {diag['portfolio_imbalance']:.6e} (Balance satisfied)"
    )
    print(f"   Weighted Root Mean Squared Dev: {diag['weighted_rmse']:.6f}")
    print(f"   Mean Credibility Factor:        {diag['mean_credibility']:.4f}")
    print(f"   High Credibility Proportion:    {diag['high_credibility_ratio']:.1%}")
    if "theoretical_mse" in diag:
        print(f"   Theoretical Prediction MSE:     {diag['theoretical_mse']}")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
