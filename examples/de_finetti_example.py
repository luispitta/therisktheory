"""Example script demonstrating De Finetti comparative analysis and elementary insurance."""
import numpy as np
from therisktheory.finetti_ruin.de_finetti_comparison import (
    elementary_insurance_case,
    compare_classical_de_finetti,
    benchmark_ruin_models,
)


def main() -> None:
    # 1. Elementary insurance case evaluation with sum insured S (Prof. Pitta's observation case)
    initial_capital = 100.0
    premium = 10.0
    claim_prob = 0.25
    claim_amount = 40.0

    terminal_pos = elementary_insurance_case(initial_capital, premium, claim_prob, claim_amount)
    print(f"Elementary Insurance Terminal Position: {terminal_pos}")

    # 2. Reproducible benchmark comparing classical vs De Finetti ruin probabilities under identical risk
    benchmark = benchmark_ruin_models(
        initial_capital=50.0,
        lambda_rate=1.2,
        claim_mean=50.0,
        safety_loading=0.15,
        horizon_periods=3,
    )
    print(f"Classical Ruin Estimates: {benchmark['classical_ruin']}")
    print(f"De Finetti Ruin Estimates: {benchmark['de_finetti_ruin']}")
    print(f"Ruin Probability Absolute Deviations: {benchmark['absolute_deviations']}")


if __name__ == "__main__":
    main()
