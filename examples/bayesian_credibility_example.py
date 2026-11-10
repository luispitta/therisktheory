"""
Example script demonstrating Bayesian credibility premium calculation 
using the therisktheory library.
"""

import numpy as np
from therisktheory.credibility import (
    bayesian_credibility_premium,
    posterior_predictive_mean,
)


def run_example():
    print("=== TheRiskTheory: Topic 4 - Bayesian Credibility Example ===")
    
    alpha_prior = 3.0
    beta_prior = 1.5
    print(f"Prior Distribution: Gamma(alpha = {alpha_prior}, beta = {beta_prior})")
    print(f"Prior Mean (Expected Portfolio Risk Rate): {alpha_prior / beta_prior:.4f}\n")

    risk_ids = ["Risk_A", "Risk_B", "Risk_C"]
    sample_claims = np.array([2.0, 8.0, 15.0])
    exposures = np.array([1.0, 3.0, 5.0])

    print("Observed Data:")
    for r_id, claims, exp in zip(risk_ids, sample_claims, exposures):
        print(f" - {r_id}: Claims = {claims}, Exposure units = {exp}, Sample rate = {claims/exp:.4f}")

    premiums = bayesian_credibility_premium(alpha_prior, beta_prior, sample_claims, exposures)
    predictive_means = posterior_predictive_mean(alpha_prior, beta_prior, sample_claims, exposures)

    print("\nResults:")
    for r_id, prem, p_mean in zip(risk_ids, premiums, predictive_means):
        print(f" - {r_id}: Bayesian Credibility Premium = {prem:.4f} | Predictive Mean = {p_mean:.4f}")


if __name__ == "__main__":
    run_example()
