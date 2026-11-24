"""Unit tests and mathematical validation for De Finetti ruin functions."""

import numpy as np
import pytest

from therisktheory.finetti_ruin import (
    de_finetti_ruin_probability,
    de_finetti_theorem_terms,
    validate_de_finetti_identity,
)


def test_validate_de_finetti_identity_list_input() -> None:
    """Verify sequence (list) inputs are normalized without TypeError.

    Explicit test case from Prof. Pitta's review: [0, 1] with p = 0.75.
    """
    res = validate_de_finetti_identity([0, 1], 0.75)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    assert np.all(res)


def test_de_finetti_ruin_probability_analytical() -> None:
    """Verify analytical example from Kaas et al. (2008) Section 4.5."""
    res = de_finetti_ruin_probability(initial_reserve=2, win_probability=0.6)
    expected = (0.4 / 0.6) ** 3
    assert np.isclose(res.item(), expected, atol=1e-12)


def test_de_finetti_ruin_boundary_fair_game() -> None:
    """For a fair game (p = 0.5), ruin is certain (psi = 1.0)."""
    res = de_finetti_ruin_probability(initial_reserve=5, win_probability=0.5)
    assert res.item() == 1.0


def test_de_finetti_ruin_boundary_zero_reserve() -> None:
    """At u = 0 with p = 0.75: psi(0) = q / p = 1/3."""
    res = de_finetti_ruin_probability(initial_reserve=0, win_probability=0.75)
    assert np.isclose(res.item(), 1.0 / 3.0, atol=1e-12)


def test_de_finetti_ruin_finite_target() -> None:
    """Verify finite barrier (target reserve)."""
    res = de_finetti_ruin_probability(
        initial_reserve=1, win_probability=0.5, target_reserve=3
    )
    assert np.isclose(res.item(), 0.5)


def test_de_finetti_theorem_terms_and_bound() -> None:
    """Verify the Lundberg-De Finetti inequality psi(u) <= exp(-R * u)."""
    terms = de_finetti_theorem_terms(initial_reserve=3, win_probability=0.7)
    assert terms.ruin_probability <= terms.upper_bound
    expected_ratio = (1.0 - 0.7) / 0.7
    assert np.isclose(terms.decay_factor.item(), expected_ratio, atol=1e-12)


def test_validate_de_finetti_identity_vectorized() -> None:
    """Verify identity holds across array inputs."""
    reserves = np.array([0, 1, 2, 5, 10])
    probs = np.array([0.55, 0.60, 0.70, 0.80, 0.90])
    valid = validate_de_finetti_identity(reserves, probs)
    assert np.all(valid)


def test_de_finetti_ruin_invalid_inputs() -> None:
    """Test validation and error handling."""
    with pytest.raises(ValueError, match="whole number of units"):
        de_finetti_ruin_probability(initial_reserve=2.5, win_probability=0.6)

    with pytest.raises(ValueError, match="strictly between 0.5 and 1.0"):
        de_finetti_theorem_terms(initial_reserve=2, win_probability=0.5)
