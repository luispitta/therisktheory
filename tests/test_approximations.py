"""Unit tests for Gamma approximation and actuarial transformations."""
from __future__ import annotations
import numpy as np
import pytest
from therisktheory.severity_aggregate import (
    esscher_transform,
    gamma_approximation,
    haldane_transform,
    wilson_hilferty_transform,
)

def test_gamma_approximation() -> None:
    res = gamma_approximation(1000.0, 50000.0, quantiles=[0.95])
    assert "shape_alpha" in res
    assert "scale_theta" in res
    assert res["shape_alpha"] > 0
    assert res["scale_theta"] > 0
    assert len(res["quantiles"]) == 1

def test_wilson_hilferty_transform() -> None:
    vals = np.array([10.0, 15.0, 20.0])
    z = wilson_hilferty_transform(vals, degrees_freedom=10.0)
    assert isinstance(z, np.ndarray)
    assert z.shape == vals.shape

def test_haldane_transform() -> None:
    props = np.array([0.1, 0.2, 0.5])
    res = haldane_transform(props, total_count=100.0)
    assert isinstance(res, np.ndarray)
    assert res.shape == props.shape

def test_esscher_transform() -> None:
    losses = np.array([10.0, 20.0])
    probs = np.array([0.6, 0.4])
    tilted = esscher_transform(losses, probs, esscher_parameter=0.05)
    np.testing.assert_allclose(np.sum(tilted), 1.0, atol=1e-6)

def test_invalid_approximations_raise_errors() -> None:
    with pytest.raises(ValueError):
        gamma_approximation(-100.0, 500.0)
    with pytest.raises(ValueError):
        wilson_hilferty_transform(10.0, degrees_freedom=-1.0)
    with pytest.raises(ValueError):
        haldane_transform(1.5, total_count=100.0)
    with pytest.raises(ValueError):
        esscher_transform(np.array([10.0]), np.array([0.5]), 0.1)
