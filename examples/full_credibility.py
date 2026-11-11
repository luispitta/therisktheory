"""Example demonstrating full credibility standards and partial transition."""

import numpy as np
from scipy.stats import norm

from therisktheory.credibility import (
    full_credibility_required_exposure,
    full_credibility_standard,
)

# 1. Independent calculation verification using scipy.stats.norm
p = 0.95
k = 0.05
z_indep = norm.ppf((1.0 + p) / 2.0)
n_indep = (z_indep / k) ** 2

n_std = full_credibility_standard(target_accuracy=k, confidence_level=p)
print(f"Calculated standard n_0 (95%, 5%): {n_std.item():,.2f}")
print(f"Independent scipy verification:     {n_indep:,.2f}")

# 2. Compound aggregate loss standard (severity variance CV^2 = 1.0)
mean_x = 2_500.0
var_x = 6_250_000.0
n_comp = full_credibility_standard(
    target_accuracy=k,
    confidence_level=p,
    severity_mean=mean_x,
    severity_variance=var_x,
)
print(f"Compound Loss Standard (CV=1.0):   {n_comp.item():,.2f} claims")

# 3. Compatibility with Monticelli's partial credibility: Z = min(1, sqrt(n/n_0))
obs_claims = np.array([500.0, 1_000.0, 1_537.0, 2_000.0])
z_factors = np.minimum(1.0, np.sqrt(obs_claims / n_std.item()))

print("\nCredibility factor Z progression (Transition to Full Credibility):")
for n_val, z_val in zip(obs_claims, z_factors, strict=True):
    status = "FULL CREDIBILITY (Z=1.0)" if z_val == 1.0 else f"Partial (Z={z_val:.4f})"
    print(f"Observed claims: {n_val:6,.0f} -> {status}")

# 4. Required policyholders/exposures
annual_freq = 0.08
req_vehicles = full_credibility_required_exposure(
    claim_frequency_rate=annual_freq,
    target_accuracy=k,
    confidence_level=p,
)
print(f"\nRequired Vehicle-Years for Full Credibility: {req_vehicles.item():,.0f}")
