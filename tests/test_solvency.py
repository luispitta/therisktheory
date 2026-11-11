"""Tests for solvency and guarantee capital models."""

import numpy as np

from therisktheory.solvency_pricing import guarantee_capital, solvency_margin


def test_guarantee_capital():
    """Test guarantee capital calculation."""
    u = 100.0
    max_claim = 250.0
    margin = 0.1
    expected = (250.0 - 100.0) * 1.1
    res = guarantee_capital(u, max_claim, margin)
    np.testing.assert_allclose(res, expected)


def test_solvency_margin():
    """Test solvency margin calculation."""
    premiums = np.array([1000.0, 500.0])
    claims = np.array([800.0, 1200.0])
    coefficient = 0.2
    expected = np.array([1000.0, 1200.0]) * 0.2
    res = solvency_margin(premiums, claims, coefficient)
    np.testing.assert_allclose(res, expected)
    