"""De Finetti ruin probability, theorem terms, and identity validation.

Topic 2: De Finetti, Games, and Ruin.
Author: Adrianna Passanisi

References:
    - Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
      Modern Actuarial Risk Theory: Using R. 2nd ed. Springer.
      Section 4.5, pp. 98-99.
    - Moriconi, F. (1982). A Pricing Model in a Sensitive Insurance Market.
      ASTIN Bulletin, 13(2), 135-149.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation
from therisktheory.finetti_ruin.de_finetti_safety import (
    de_finetti_safety_loading,
    safety_index,
)
from therisktheory.finetti_ruin.games import gambler_ruin_probability

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class DeFinettiTheoremTerms:
    """Analytical components of De Finetti's ruin theorem.

    Attributes:
        ruin_probability: Exact probability of eventual ruin psi(u).
        upper_bound: Exponential Lundberg-De Finetti bound exp(-R * u).
        safety_index: Adjustment coefficient R = ln(p / q).
        safety_loading: Expected surplus drift E[G] = 2p - 1.
        decay_factor: Exact ratio psi(u) / exp(-R * u) = exp(-R) = q / p.
    """

    ruin_probability: npt.NDArray[np.float64]
    upper_bound: npt.NDArray[np.float64]
    safety_index: npt.NDArray[np.float64]
    safety_loading: npt.NDArray[np.float64]
    decay_factor: npt.NDArray[np.float64]


def de_finetti_ruin_probability(
    initial_reserve: npt.ArrayLike,
    win_probability: npt.ArrayLike,
    target_reserve: npt.ArrayLike = np.inf,
) -> npt.NDArray[np.float64]:
    r"""Calculate ruin probability under De Finetti's discrete reserve model.

    In De Finetti's unit-step model, the reserve evolves as:
        U(n) = u + \sum_{i=1}^n G_i
    where G_i = +1 with probability p and G_i = -1 with probability q = 1 - p.
    Ruin occurs when U(n) < 0 (reaching -1). Starting from reserve u >= 0,
    this corresponds to a gambler's ruin process starting at z = u + 1.

    Args:
        initial_reserve: Starting reserve u >= 0 (integer units of stake).
        win_probability: Probability p of positive unit increment.
        target_reserve: Optional upper barrier b >= u, or np.inf for
            unlimited horizon. Defaults to np.inf.

    Returns:
        The ruin probability as a float64 NumPy array in [0.0, 1.0].

    Raises:
        TypeError: If inputs cannot be converted to real numbers.
        ValueError: If u is negative, u or b are non-integers, or shapes
            cannot be broadcast together.
    """
    u = _validation.as_nonnegative_finite_array(
        initial_reserve, "initial_reserve"
    )
    p = _validation.as_probability_array(win_probability, "win_probability")
    b = _validation.as_real_array(
        target_reserve, "target_reserve", allow_inf=True
    )

    _validation.check_broadcastable(
        initial_reserve=u,
        win_probability=p,
        target_reserve=b,
    )

    if np.any(u != np.floor(u)):
        raise ValueError("initial_reserve must be a whole number of units")

    finite_b = np.isfinite(b)
    if np.any(b[finite_b] != np.floor(b[finite_b])):
        raise ValueError("target_reserve must be a whole number or inf")

    if np.any(u > b):
        raise ValueError("initial_reserve cannot exceed target_reserve")

    z = u + 1.0
    a = np.where(finite_b, b + 1.0, np.inf)

    return gambler_ruin_probability(
        initial_capital=z,
        target_capital=a,
        win_probability=p,
    )


def de_finetti_theorem_terms(
    initial_reserve: npt.ArrayLike,
    win_probability: npt.ArrayLike,
) -> DeFinettiTheoremTerms:
    r"""Compute analytical components of the Lundberg-De Finetti theorem.

    For an infinite horizon and favorable game (0.5 < p < 1):
        psi(u) = (q / p)^(u + 1) = exp(-R * (u + 1))
        psi(u) <= exp(-R * u)

    Args:
        initial_reserve: Starting reserve u >= 0 (integer units).
        win_probability: Winning probability p in (0.5, 1.0).

    Returns:
        DeFinettiTheoremTerms containing exact ruin probability,
        upper bound, safety index, safety loading, and decay factor.

    Raises:
        TypeError: If inputs cannot be cast to real numbers.
        ValueError: If u is negative or non-integer, or if p is not in (0.5, 1).
    """
    u = _validation.as_nonnegative_finite_array(
        initial_reserve, "initial_reserve"
    )
    p = _validation.as_probability_array(win_probability, "win_probability")

    _validation.check_broadcastable(initial_reserve=u, win_probability=p)

    if np.any(u != np.floor(u)):
        raise ValueError("initial_reserve must be a whole number of units")
    if np.any(p <= 0.5) or np.any(p >= 1.0):
        raise ValueError("win_probability must be strictly between 0.5 and 1.0")

    r = safety_index(p)
    drift = de_finetti_safety_loading(p)
    psi = de_finetti_ruin_probability(u, p, target_reserve=np.inf)
    bound = np.exp(-r * u)
    decay = np.exp(-r)

    return DeFinettiTheoremTerms(
        ruin_probability=psi,
        upper_bound=bound,
        safety_index=r,
        safety_loading=drift,
        decay_factor=decay,
    )


def validate_de_finetti_identity(
    initial_reserve: npt.ArrayLike,
    win_probability: npt.ArrayLike,
    *,
    atol: float = 1e-12,
) -> npt.NDArray[np.bool_]:
    """Validate mathematical consistency of De Finetti's ruin identity.

    Normalizes inputs first to support sequences (lists/tuples) properly.
    Uses strict absolute tolerance (|a - b| <= atol) without relative tolerance.

    Args:
        initial_reserve: Starting reserve u >= 0 (supports scalar or sequence).
        win_probability: Winning probability p in (0.5, 1.0).
        atol: Absolute numerical tolerance. Defaults to 1e-12.

    Returns:
        NumPy boolean array indicating whether theoretical identity holds.
    """
    u = _validation.as_nonnegative_finite_array(
        initial_reserve, "initial_reserve"
    )
    p = _validation.as_probability_array(win_probability, "win_probability")

    terms = de_finetti_theorem_terms(u, p)

    expected_psi = np.exp(-terms.safety_index * (u + 1.0))
    exact_match = np.abs(terms.ruin_probability - expected_psi) <= atol

    bound_holds = terms.ruin_probability <= (terms.upper_bound + atol)

    q = 1.0 - p
    r = terms.safety_index
    equation_root = p * np.exp(-r) + q * np.exp(r)
    root_match = np.abs(equation_root - 1.0) <= atol

    return exact_match & bound_holds & root_match
