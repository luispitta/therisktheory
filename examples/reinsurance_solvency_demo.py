"""Demonstration of reinsurance treaties and cedent solvency evaluation.

This script demonstrates the five reinsurance functions developed for Topic 1:
- Quota Share (proportional)
- Surplus Share (proportional)
- Excess of Loss (non-proportional per-risk)
- Stop-Loss (non-proportional aggregate)
- Reinsured Solvent Position (actuarial capital & solvency evaluation)
"""

from __future__ import annotations

import numpy as np

from therisktheory.solvency_pricing import (
    excess_of_loss,
    quota_share,
    reinsured_solvent_position,
    stop_loss,
    surplus_share,
)


def main() -> None:
    print("=" * 70)
    print("TheRiskTheory — Topic 1: Reinsurance and Solvency Evaluation Demo")
    print("=" * 70)

    # 1. Quota Share Reinsurance
    claims = np.array([1200.0, 4500.0, 15000.0, 30000.0])
    qs_ret, qs_ced = quota_share(claims, retention_rate=0.70)
    print("\n1. Quota Share (70% retention):")
    print(f"   Gross Claims:     {claims}")
    print(f"   Retained Losses:  {qs_ret}")
    print(f"   Ceded Losses:     {qs_ced}")

    # 2. Surplus Share Reinsurance
    sums_insured = np.array([10000.0, 25000.0, 50000.0, 80000.0])
    ss_ret, ss_ced = surplus_share(
        claims,
        sum_insured=sums_insured,
        retention_line=10000.0,
        max_lines=3.0,
    )
    print("\n2. Surplus Share (Line = 10,000, 3 Lines capacity = 30,000):")
    print(f"   Sums Insured:     {sums_insured}")
    print(f"   Gross Claims:     {claims}")
    print(f"   Retained Losses:  {ss_ret}")
    print(f"   Ceded Losses:     {ss_ced}")

    # 3. Excess of Loss (XL) Reinsurance
    xl_ret, xl_ced = excess_of_loss(claims, priority=5000.0, limit=15000.0)
    print("\n3. Excess of Loss (Priority = 5,000, Limit = 15,000):")
    print(f"   Gross Claims:     {claims}")
    print(f"   Retained Losses:  {xl_ret}")
    print(f"   Ceded Losses:     {xl_ced}")

    # 4. Stop-Loss (Aggregate Reinsurance)
    aggregate_scenarios = np.array([45000.0, 75000.0, 110000.0, 150000.0])
    sl_ret, sl_ced = stop_loss(aggregate_scenarios, deductible=60000.0, limit=50000.0)
    print("\n4. Stop-Loss (Aggregate Deductible = 60,000, Cover Limit = 50,000):")
    print(f"   Aggregate Loss:   {aggregate_scenarios}")
    print(f"   Retained Loss:    {sl_ret}")
    print(f"   Ceded Loss:       {sl_ced}")

    # 5. Reinsured Solvent Position
    solvency = reinsured_solvent_position(
        initial_capital=20000.0,
        retained_premium=50000.0,
        retained_losses=sl_ret,
    )
    print("\n5. Cedent Net Solvency Position (u = 20,000, P_ret = 50,000):")
    print(f"   Net Surplus:      {solvency['net_surplus']}")
    print(f"   Is Solvent:       {solvency['is_solvent']}")
    print(f"   Deficit:          {solvency['deficit']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
