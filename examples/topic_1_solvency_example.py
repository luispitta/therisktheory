"""Example: Topic 1 - Solvency Evaluation & Guarantee Capital.

Demonstrates how solvency evaluation integrates pure premiums (Bracamonte)
and loaded premiums (Garcia) to compute available capital and solvency metrics.
"""

from therisktheory.solvency import evaluate_portfolio_solvency


def main():
    print("=== THE RISK THEORY - TOPIC 1 SOLVENCY INTEGRATION ===")

    # Define Portfolio Inputs
    initial_capital = 80.0
    expected_frequency = 10.0  # 10 expected claims
    average_severity = 8.0  # $8 average cost per claim
    safety_loading = 0.15  # 15% safety loading factor
    max_loss_99 = 120.0  # Peak exposure at 99% quantile
    safety_margin = 0.10  # 10% statutory margin

    # Run Integrated Evaluation
    results = evaluate_portfolio_solvency(
        initial_capital=initial_capital,
        expected_frequency=expected_frequency,
        average_severity=average_severity,
        safety_loading=safety_loading,
        max_loss=max_loss_99,
        safety_margin=safety_margin,
    )

    # Display Metrics
    print(
        f"1. Pure Premium (Bracamonte):        ${results['pure_premium']:.2f}"
    )
    print(
        f"2. Loaded Premium (Garcia):          ${results['loaded_premium']:.2f}"
    )
    print(
        f"3. Total Available Capital:          ${results['total_available_capital']:.2f}"
    )
    print(
        f"4. Solvency Ratio (Adimensional):    {results['solvency_ratio']:.2f}"
    )
    print(
        f"5. Guarantee Capital Needed:         ${results['guarantee_capital_required']:.2f}"
    )


if __name__ == "__main__":
    main()
