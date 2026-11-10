"""Example script demonstrating Gamma approximation, transformations, and benchmark vs simulation."""
import numpy as np
from therisktheory.severity_aggregate import (
    esscher_transform,
    gamma_approximation,
    haldane_transform,
    wilson_hilferty_transform,
)

def main() -> None:
    print("=== Topic 3: Approximations and Transformations ===")
    
    # 1. Moment-matched Gamma Approximation
    mean_loss = 100000.0
    var_loss = 250000000.0
    quantiles = np.array([0.90, 0.95, 0.99])
    gamma_res = gamma_approximation(mean_loss, var_loss, quantiles=quantiles)
    print(f"Gamma Alpha (shape): {gamma_res['shape_alpha']:.4f}")
    print(f"Gamma Theta (scale): {gamma_res['scale_theta']:.4f}")
    print(f"Gamma Approximated Quantiles (90%, 95%, 99%): {gamma_res['quantiles']}")
    
    # 2. Benchmarking Analytical Gamma vs Monte Carlo Simulation
    np.random.seed(42)
    simulated_losses = np.random.gamma(
        shape=gamma_res["shape_alpha"],
        scale=gamma_res["scale_theta"],
        size=100000,
    )
    sim_quantiles = np.quantile(simulated_losses, quantiles)
    abs_errors = np.abs(gamma_res["quantiles"] - sim_quantiles)
    print(f"Simulated Quantiles (Monte Carlo): {sim_quantiles}")
    print(f"Absolute Precision Difference: {abs_errors}")
    
    # 3. Wilson-Hilferty Transform
    sample_values = np.array([12.0, 18.5, 24.0])
    df = 15.0
    wh_z = wilson_hilferty_transform(sample_values, df)
    print(f"Wilson-Hilferty Standardized Z-scores: {wh_z}")
    
    # 4. Haldane Transform
    props = np.array([0.05, 0.15, 0.30])
    n_base = 1000.0
    haldane_res = haldane_transform(props, n_base)
    print(f"Haldane Transformed Values: {haldane_res}")
    
    # 5. Esscher Transform (Exponential Tilting)
    losses = np.array([10.0, 50.0, 100.0, 200.0])
    probs = np.array([0.50, 0.30, 0.15, 0.05])
    h_param = 0.01
    skewed_probs = esscher_transform(losses, probs, h_param)
    print(f"Original Probabilities: {probs}")
    print(f"Esscher-transformed Probabilities (h={h_param}): {skewed_probs}")

if __name__ == "__main__":
    main()
