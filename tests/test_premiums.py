"""Tests for pure premiums and premium calculation principles."""

import numpy as np

from therisktheory.solvency_pricing import expected_value_premium, pure_premium


def test_pure_premium():
    """Test pure premium calculation."""
    freq = np.array([2.0, 5.0])
    sev = np.array([100.0, 200.0])
    expected = np.array([200.0, 1000.0])
    res = pure_premium(freq, sev)
    np.testing.assert_allclose(res, expected)


def test_expected_value_premium():
    """Test expected value premium principle with safety loading."""
    net = np.array([200.0, 1000.0])
    loading = 0.15
    expected = net * 1.15
    res = expected_value_premium(net, loading)
    np.testing.assert_allclose(res, expected)