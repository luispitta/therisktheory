"""Example demonstrating De Finetti ruin probability and theorem terms."""

import numpy as np

from therisktheory.finetti_ruin import (
    de_finetti_ruin_probability,
    de_finetti_theorem_terms,
    validate_de_finetti_identity,
)

u = 2
p = 0.60

psi = de_finetti_ruin_probability(u, p)
terms = de_finetti_theorem_terms(u, p)
is_valid = validate_de_finetti_identity(u, p)

print(f"Initial reserve u: {u}")
print(f"Win probability p: {p}")
print(f"Ruin probability psi(u): {psi.item():.6f}")
print(f"Lundberg bound exp(-R*u): {terms.upper_bound.item():.6f}")
print(f"Safety index (R): {terms.safety_index.item():.6f}")
print(f"Surplus drift E[G]: {terms.safety_loading.item():.2f}")
print(f"De Finetti identity valid: {np.all(is_valid)}")
