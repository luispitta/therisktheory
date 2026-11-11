"""Tests for solvency module."""

import numpy as np
import pytest

from therisktheory.solvency_pricing import guarantee_capital, solvency_ratio


def test_solvency_ratio_scalar():
    """Test solvency ratio calculation with valid scalar values."""
    assert solvency_ratio(150000.0, 100000.0) == 1.5


def test_solvency_ratio_vectorized():
    """Test solvency ratio calculation with NumPy arrays."""
    avail = np.array([100.0, 200.0, 300.0])
    req = np.array([100.0, 100.0, 150.0])
    expected = np.array([1.0, 2.0, 2.0])
    np.testing.assert_allclose(solvency_ratio(avail, req), expected)


def test_solvency_ratio_invalid_inputs():
    """Test boundary and invalid input handling for solvency ratio."""
    with pytest.raises(
        ValueError, match="Premium income must be strictly positive."
    ):
        solvency_ratio(100.0, 0.0)


def test_guarantee_capital_basic():
    """Test basic guarantee capital calculation."""
    cap = guarantee_capital(3.0, 200.0, 0.10, 0.2)
    assert cap > 0.0
