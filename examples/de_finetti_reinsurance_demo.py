"""Demonstration of De Finetti's proportional reinsurance framework.

This example illustrates:
1. Optimal retention calculation under De Finetti's safety index criterion.
2. Sensitivity of De Finetti's safety index and expected profit across retention levels.
3. Quota share claim allocation under optimal retention.
4. Boundary analysis and zero retention under arbitrage conditions.
5. Asymptotic ruin probability diffusion approximations as a function of reserve capital.
"""

from __future__ import annotations

import numpy as np

from therisktheory.finetti_ruin import (
    contrast_reinsurance_boundary_cases,
    de_finetti_optimal_retention,
    de_finetti_retention_ruin_approximation,
    de_finetti_retention_ruin_bound,
    interoperate_with_portfolio_reserves_and_pricing,
    proportional_reinsurance_objective,
)


def main() -> None:
    print("=" * 72)
    print("TheRiskTheory — Topic 2: De Finetti Proportional Reinsurance Demo")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 1. Baseline Portfolio Parameters & Optimal Retention Derivation
    # -------------------------------------------------------------------------
    expected_claim = 1000.0  # E[X]
    variance_claim = 250000.0  # Var(X), std = 500
    insurer_loading = 0.15  # eta (15% safety loading)
    reinsurer_loading = 0.25  # xi (25% reinsurance loading)

    alpha_star = de_finetti_optimal_retention(
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
        criterion="safety_index",
    )
    print("\n1. De Finetti Optimal Retention (Moderate Cost Regime):")
    print(f"   Insurer Loading (eta):      {insurer_loading:.1%}")
    print(f"   Reinsurer Loading (xi):    {reinsurer_loading:.1%}")
    print(f"   Optimal Retention alpha*:  {float(alpha_star):.4f}")
    print("   Analytical formula: 2 * (1 - eta / xi) = 2 * (1 - 0.15/0.25) = 0.8000")

    # -------------------------------------------------------------------------
    # 2. Objective Function Across Retention Rates
    # -------------------------------------------------------------------------
    alphas = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    safety_indices = proportional_reinsurance_objective(
        retention_rate=alphas,
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
        criterion="safety_index",
    )
    expected_profits = proportional_reinsurance_objective(
        retention_rate=alphas,
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
        criterion="expected_profit",
    )
    print("\n2. Retention Schedule Analysis:")
    for a, theta, prof in zip(alphas, safety_indices, expected_profits, strict=True):
        opt_tag = " [OPTIMUM]" if np.isclose(a, float(alpha_star)) else ""
        print(f"   alpha = {a:.1f} | E[Profit] = {prof:5.1f} | theta = {theta:.6f}{opt_tag}")

    # -------------------------------------------------------------------------
    # 3. Quota Share Loss Allocation Under Optimal Retention
    # -------------------------------------------------------------------------
    sample_claims = np.array([500.0, 1000.0, 2500.0, 5000.0])
    retained_claims = float(alpha_star) * sample_claims
    ceded_claims = (1.0 - float(alpha_star)) * sample_claims
    print("\n3. Optimal Quota Share Claim Splitting:")
    print(f"   Gross Claims:    {sample_claims}")
    print(f"   Retained Losses: {retained_claims}")
    print(f"   Ceded Losses:    {ceded_claims}")

    # -------------------------------------------------------------------------
    # 4. Zero Retention & Fronting Arbitrage (Instructor's Benchmark Case)
    # -------------------------------------------------------------------------
    arb_mu, arb_var = 1.0, 1.0
    arb_eta, arb_xi = 0.20, 0.10  # Reinsurer is cheaper than primary insurer
    arb_alpha = de_finetti_optimal_retention(arb_mu, arb_var, arb_eta, arb_xi)
    arb_theta_zero = proportional_reinsurance_objective(
        0.0, arb_mu, arb_var, arb_eta, arb_xi, criterion="safety_index"
    )
    arb_theta_01 = proportional_reinsurance_objective(
        0.10, arb_mu, arb_var, arb_eta, arb_xi, criterion="safety_index"
    )
    print("\n4. Fronting Arbitrage Regime (eta = 0.20 > xi = 0.10):")
    print(f"   Optimal Retention alpha*:  {float(arb_alpha):.4f}")
    print(f"   Safety Index at alpha=0:   {float(arb_theta_zero)} (zero variance limit)")
    print(f"   Safety Index at alpha=0.1: {float(arb_theta_01):.4f}")
    print(
        f"   Criterion check: theta(alpha*) >= theta(0.1) -> "
        f"{float(arb_theta_zero) >= float(arb_theta_01)}"
    )

    # -------------------------------------------------------------------------
    # 5. Ruin Probability Diffusion Approximation vs Initial Capital
    # -------------------------------------------------------------------------
    capitals = np.array([0.0, 5000.0, 15000.0, 30000.0])
    ruin_approx = de_finetti_retention_ruin_approximation(
        initial_capital=capitals,
        retention_rate=alpha_star,
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
    )
    # Validate backward-compatible alias
    ruin_bounds = de_finetti_retention_ruin_bound(
        initial_capital=capitals,
        retention_rate=alpha_star,
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
    )
    assert np.allclose(ruin_approx, ruin_bounds)

    print("\n5. Ruin Probability Diffusion Approximation vs. Initial Capital:")
    for u, psi in zip(capitals, ruin_approx, strict=True):
        print(f"   Capital u = {u:6.0f} -> psi_diff(u) = {psi:.6f}")

    # -------------------------------------------------------------------------
    # 6. Contrast of All Reinsurance Boundary Cases
    # -------------------------------------------------------------------------
    boundaries = contrast_reinsurance_boundary_cases(
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
        initial_capital=5000.0,
    )
    print("\n6. Reinsurance Boundary Cases Contrast (u = 5000):")
    print(f"   Economic Regime:            {boundaries['economic_regime']}")
    print(
        f"   Full Retention (alpha=1.0): theta = {boundaries['full_retention']['safety_index']:.6f}, psi = {boundaries['full_retention']['ruin_probability']:.6f}"
    )
    print(
        f"   Optimal (alpha={boundaries['optimal_retention']['retention_rate']:.2f}):      theta = {boundaries['optimal_retention']['safety_index']:.6f}, psi = {boundaries['optimal_retention']['ruin_probability']:.6f}"
    )
    print(
        f"   Critical (alpha_min={boundaries['critical_retention']['retention_rate']:.2f}):  theta = {boundaries['critical_retention']['safety_index']:.6f}, psi = {boundaries['critical_retention']['ruin_probability']:.6f}"
    )
    print(
        f"   Zero Retention (alpha=0.0): theta = {boundaries['zero_retention']['safety_index']}, psi = {boundaries['zero_retention']['ruin_probability']:.6f}"
    )
    print(f"   Relative Safety Index Gain: {boundaries['relative_safety_gain']:.2%}")
    print(f"   Ruin Probability Reduction: {boundaries['ruin_reduction_ratio']:.2%}")

    # -------------------------------------------------------------------------
    # 7. Interoperability with Reserves, Pricing, and Ruin Models
    # -------------------------------------------------------------------------
    interop = interoperate_with_portfolio_reserves_and_pricing(
        expected_loss=expected_claim,
        variance_loss=variance_claim,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
        free_surplus=3000.0,
        claim_reserve=7000.0,
        target_ruin_tolerance=0.005,
    )
    print("\n7. Interoperability with Reserves and Solvency Pricing:")
    print(f"   Topic 1 Gross Premium:      {interop['pricing']['gross_premium']:.2f}")
    print(f"   Ceded Reinsurance Premium:  {interop['pricing']['reinsurer_premium']:.2f}")
    print(f"   Net Retained Premium:       {interop['pricing']['retained_premium']:.2f}")
    print(
        f"   Initial Capital (3k+7k):    {interop['balance_sheet_reserves']['total_initial_capital']:.2f}"
    )
    print(
        f"   Cramér-Lundberg Proxy R:    Gross = {interop['ruin_solvency']['diffusion_adjustment_factor_gross']:.6f}, Optimal = {interop['ruin_solvency']['diffusion_adjustment_factor_optimal']:.6f}"
    )
    print(
        f"   Capital Relief (0.5% tol):  {interop['ruin_solvency']['capital_relief']:.2f} ({interop['ruin_solvency']['percentage_capital_relief']:.2f}%)"
    )

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
