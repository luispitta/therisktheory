"""Example: Integrating Premium Calculation with Solvency Evaluation (Topic 1).

Demonstrates how solvency evaluation depends on Pure Premium (Bracamonte)
and Loaded Premium (Garcia) to compute final available capital and solvency ratios.
"""

from therisktheory.solvency import guarantee_capital, solvency_ratio

# Fallback imports in case peer modules are in concurrent PR branches
try:
    from therisktheory.premiums import loaded_premium, pure_premium
except ImportError:
    # Fallback simulation functions based on Topic 1 definitions
    def pure_premium(
        expected_frequency: float, average_severity: float
    ) -> float:
        return expected_frequency * average_severity

    def loaded_premium(pure_prem: float, safety_loading: float) -> float:
        return pure_prem * (1.0 + safety_loading)


def main():
    # 1. Portfolio Parameters
    initial_capital = 80.0
    expected_claims = 10.0
    avg_claim_cost = 8.0
    max_loss_99 = 120.0  # Maximum expected loss at 99% confidence level
    safety_loading_rate = 0.15  # Garcia's safety loading

    # 2. Consume Peer Modules (Bracamonte & Garcia)
    p_pure = pure_premium(
        expected_claims, avg_claim_cost
    )  # 10 * 8 = 80.0 (Bracamonte)
    total_premium = loaded_premium(
        p_pure, safety_loading_rate
    )  # 80 * 1.15 = 92.0 (Garcia)

    # 3. Compute Total Available Capital for Insurer
    total_available_capital = initial_capital + total_premium  # 80 + 92 = 172.0

    # 4. Compute Solvency and Guarantee Capital Metrics (Monticelli)
    s_ratio = solvency_ratio(
        available_capital=total_available_capital, required_capital=max_loss_99
    )
    extra_cap_needed = guarantee_capital(
        initial_capital=initial_capital,
        max_loss=max_loss_99,
        safety_margin=0.10,
    )

    print("=== THE RISK THEORY - TOPIC 1 INTEGRATION ===")
    print(f"Pure Premium (Bracamonte):        {p_pure:.2f}")
    print(f"Loaded Premium (Garcia):          {total_premium:.2f}")
    print(f"Total Available Capital:          {total_available_capital:.2f}")
    print(f"Solvency Ratio (Adimensional):    {s_ratio:.2f}")
    print(f"Guarantee Capital Required:       {extra_cap_needed:.2f}")


if __name__ == "__main__":
    main()
