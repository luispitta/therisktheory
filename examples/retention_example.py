"""Example script demonstrating risk retention and inflation adjustments."""

import numpy as np
from therisktheory.retention import (
    retained_loss,
    retention_capacity,
    inflate_losses,
    deflate_losses,
)

def main() -> None:
    # 1. Retention capacity based on surplus
    surplus = 500000.0
    retention_ratio = 0.20
    capacity = retention_capacity(surplus, retention_ratio)
    print(f"Maximum Retention Capacity: ${capacity:,.2f}")

    # 2. Retained loss under retention limit
    losses = np.array([50000.0, 120000.0, 80000.0])
    retained = retained_loss(losses, capacity)
    print(f"Original Losses: {losses}")
    print(f"Retained Losses: {retained}")

    # 3. Inflation projection and reversal check
    inflation_rate = 0.05  # 5% annual inflation
    periods = 3
    inflated = inflate_losses(losses, inflation_rate, periods)
    deflated = deflate_losses(inflated, inflation_rate, periods)

    print(f"\nInflated Losses (t={periods}): {inflated}")
    print(f"Deflated Losses (PV): {deflated}")
    print(f"Reversibility Check Passed: {np.allclose(deflated, losses)}")

if __name__ == "__main__":
    main()
