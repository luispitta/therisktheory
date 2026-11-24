"""Test module for De Finetti comparison and elementary insurance case."""
from __future__ import annotations
import numpy as np
import pytest
from therisktheory.finetti_ruin.de_finetti_comparison import (
    compare_classical_de_finetti,
    elementary_insurance_case,
    benchmark_ruin_models,
)

def test_elementary_insurance_case() -> None:
    """Test standard and boundary values for elementary insurance."""
    res_scalar = elementary_insurance_case(100.0, 10.0, 0.25, 40.0)
    assert res_scalar == pytest.approx(100.0)
    cap = np.array([100.0, 200.0])
    prem = np.array([50.0, 50.0])
    prob = np.array([0.1, 0.5])
    claim = np.array([100.0, 200.0])
    res_vector = elementary_insurance_case(cap, prem, prob, claim)
    expected = cap + prem - (prob * claim)
    np.testing.assert_allclose(res_vector, expected)

def test_compare_classical_de_finetti() -> None:
    """Test absolute deviations between classical and De Finetti models."""
    classical = np.array([0.05, 0.10])
    de_finetti = np.array([0.07, 0.08])
    deviations = compare_classical_de_finetti(classical, de_finetti)
    np.testing.assert_allclose(deviations, [0.02, 0.02])

def test_benchmark_ruin_models() -> None:
    """Test reproducible benchmark generation for model comparison."""
    res = benchmark_ruin_models(50.0, 1.0, 40.0, 0.2, 3)
    assert "classical_ruin" in res
    assert "de_finetti_ruin" in res
    assert "absolute_deviations" in res
    assert len(res["absolute_deviations"]) == 3
    # Verify sensitivity to safety loading theta
    res_low = benchmark_ruin_models(100.0, 1.0, 100.0, 0.1, 3)
    res_high = benchmark_ruin_models(100.0, 1.0, 100.0, 1.0, 3)
    assert not np.allclose(res_low["de_finetti_ruin"], res_high["de_finetti_ruin"])

def test_elementary_insurance_boundary_probabilities() -> None:
    """Test boundary probabilities (0.0 and 1.0)."""
    assert elementary_insurance_case(100.0, 50.0, 0.0, 200.0) == 150.0
    assert elementary_insurance_case(100.0, 50.0, 1.0, 200.0) == 100.0 + 50.0 - 200.0

def test_invalid_inputs_raise_errors() -> None:
    """Ensure invalid parameters raise ValueError or TypeError."""
    with pytest.raises(ValueError):
        elementary_insurance_case(-10.0, 50.0, 0.2, 200.0)
    with pytest.raises(ValueError):
        elementary_insurance_case(100.0, 50.0, 1.5, 200.0)
    with pytest.raises(TypeError):
        elementary_insurance_case("invalid", 50.0, 0.2, 200.0)
    with pytest.raises(ValueError):
        benchmark_ruin_models(-10.0, 1.0, 40.0, 0.2, 3)
