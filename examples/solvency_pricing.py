"""Example: Solvency, pricing, and finite-horizon ruin estimation."""

from therisktheory.solvency_pricing.expected_profit import expected_profit
from therisktheory.solvency_pricing.finite_horizon_ruin_estimate import (
    finite_horizon_ruin_estimate,
)
from therisktheory.solvency_pricing.portfolio_equilibrium import (
    portfolio_equilibrium,
)

# 1. Expected underwriting profit
claims = 100_000.0
loading = 0.20
premium = (1.0 + loading) * claims
expenses = 5_000.0

profit = expected_profit(premium, claims, expenses)
print(f"Expected Profit: {profit.item():,.2f}")

# 2. Portfolio equilibrium surplus
initial_capital = 50_000.0
surplus = portfolio_equilibrium(
    initial_capital=initial_capital,
    accumulated_profit=profit,
)
print(f"Surplus Balance: {surplus.item():,.2f}")

# 3. Finite-horizon ruin probability estimation
t_horizon = 5.0
c_rate = premium / t_horizon
lam = 50.0
mean_severity = claims / (lam * t_horizon)

psi = finite_horizon_ruin_estimate(
    initial_capital=initial_capital,
    premium_rate=c_rate,
    claim_intensity=lam,
    mean_claim_size=mean_severity,
    time_horizon=t_horizon,
)
print(f"Ruin Probability psi(u): {psi.item():.6f}")

# 4. Deterministic case (Prof. Pitta review scenario: lambda=0, c=0)
psi_det = finite_horizon_ruin_estimate(
    initial_capital=1e-7,
    premium_rate=0.0,
    claim_intensity=0.0,
    mean_claim_size=100.0,
    time_horizon=1.0,
)
print(f"Deterministic ruin: {psi_det.item():.1f}")
