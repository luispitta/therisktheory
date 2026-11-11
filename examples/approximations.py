"""Example: Aggregate loss approximations vs Monte Carlo simulation."""

import numpy as np

from therisktheory.collective_risk.approximations import (
    edgeworth_approximation,
    normal_approximation,
    normal_power_approximation,
)

# Compound Poisson parameters (lambda=50, exponential claims mean=200)
lam = 50.0
mean_x = 200.0

# Wald aggregate moments
mu_s = lam * mean_x
var_s = lam * 2.0 * (mean_x**2)
skew_s = (lam * 6.0 * (mean_x**3)) / (var_s**1.5)

levels = np.array([8_000.0, 10_000.0, 12_000.0, 14_000.0])

p_norm = normal_approximation(levels, mu_s, var_s)
p_edge = edgeworth_approximation(levels, mu_s, var_s, skew_s)
p_np = normal_power_approximation(levels, mu_s, var_s, skew_s)

# Monte Carlo simulation (15,000 paths)
rng = np.random.default_rng(42)
n_claims = rng.poisson(lam, size=15_000)
claims = rng.exponential(mean_x, size=n_claims.sum())
owners = np.repeat(np.arange(15_000), n_claims)
totals = np.bincount(owners, weights=claims, minlength=15_000)
p_sim = np.array([np.mean(totals <= x) for x in levels])

print("Aggregate Loss Approximations vs Monte Carlo Simulation:")
print(f"E[S]={mu_s:,.0f}, Var(S)={var_s:,.0f}, Skewness={skew_s:.4f}")
print("-" * 65)
print("   Level       Normal    Edgeworth       NP      Simulation")
print("-" * 65)
for x, pn, pe, pnp, ps in zip(levels, p_norm, p_edge, p_np, p_sim, strict=True):
    print(f"{x:9.0f}  {pn:10.4f}  {pe:10.4f}  {pnp:10.4f}  {ps:10.4f}")
