"""Example demonstrating stability parameters and safety loading optimization."""

import numpy as np

from therisktheory.collective_risk.stability import (
    collective_safety_loading,
    loading_from_initial_reserve,
    stability_parameter,
    stability_sensitivity,
)

# Portfolio characteristics
lam = 150.0
mu_1 = 500.0
mu_2 = 500_000.0  # Var(X) = 250,000
theta = 0.15

# 1. Stability parameter and sensitivity
k = stability_parameter(lam, mu_1, mu_2, theta)
sens = stability_sensitivity(lam, mu_1, mu_2, theta)

print("Collective Stability Analysis:")
print(f"Stability parameter K: {k.item():.4f}")
print(f"Derivative dK/d(theta): {sens.d_theta.item():.4f}")
print(f"Elasticity w.r.t frequency: {sens.elasticity_lambda.item():.2f}")

# 2. Required loading for target stability K = 2.0
target_k = 2.0
theta_req = collective_safety_loading(target_k, lam, mu_1, mu_2)
print(f"Safety loading required for K={target_k}: {theta_req.item():.4%}")

# 3. Capital vs Loading trade-off (Target Ruin Probability = 1%)
reserves = np.array([5_000.0, 15_000.0, 50_000.0])
thetas_u = loading_from_initial_reserve(
    reserves,
    target_ruin_probability=0.01,
    severity_mean=mu_1,
    severity_second_moment=mu_2,
)

print("\nTrade-off: Initial Capital vs Required Loading (Ruin Target = 1%):")
for u_val, th in zip(reserves, thetas_u, strict=True):
    print(f"Reserve u = {u_val:8,f} -> Required Loading theta = {th:6.2%}")
