"""Comprehensive and robust test suite for partial credibility models and premiums.

Topic 4: Credibility Theory and Reserves.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np
import pytest
from scipy.special import erfinv
from therisktheory.credibility.partial_credibility import (
    partial_credibility_standard,
    partial_credibility_premium,
    integrated_partial_credibility_evaluation,
)


# ==========================================
# 1. ANALYTICAL TESTS (Pruebas Analíticas)
# ==========================================
def test_analytical_partial_credibility_premium():
    """Analytical test: Verify linear credibility formula against hand-calculated results."""
    mu = 1000.0
    x_bar = 1200.0
    z = 0.4
    
    result = partial_credibility_premium(mu, x_bar, z)
    # P_C = 0.4 * 1200 + 0.6 * 1000 = 480 + 600 = 1080.0
    assert result == pytest.approx(1080.0)


def test_analytical_full_credibility_standard():
    """Analytical test: Verify standard formula for 95% confidence and 5% error."""
    standard = partial_credibility_standard(100.0, target_accuracy=0.05, confidence_level=0.95)
    z_approx = np.sqrt(2.0) * erfinv(0.95)
    expected = (z_approx / 0.05) ** 2
    assert standard == pytest.approx(expected, rel=1e-3)


def test_analytical_custom_confidence_standard():
    """Analytical test: Verify full credibility standard under 99% confidence level."""
    standard = partial_credibility_standard(50.0, target_accuracy=0.01, confidence_level=0.99)
    z_approx = np.sqrt(2.0) * erfinv(0.99)
    expected = (z_approx / 0.01) ** 2
    assert standard == pytest.approx(expected, rel=1e-3)


# ==========================================
# 2. VECTORIZATION TESTS (Pruebas Vectorizadas)
# ==========================================
def test_vectorization_credibility_premium():
    """Vectorization test: Check array inputs for weights and experience means simultaneously."""
    mu = np.array([1000.0, 2000.0, 3000.0])
    x_bar = np.array([1100.0, 1900.0, 3100.0])
    z = np.array([0.5, 0.8, 0.2])
    
    result = partial_credibility_premium(mu, x_bar, z)
    expected = z * x_bar + (1.0 - z) * mu
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)


def test_vectorization_full_credibility_standard():
    """Vectorization test: Verify array support for exposure counts in standard calculations."""
    exposures = np.array([10.0, 50.0, 100.0])
    result = partial_credibility_standard(exposures, target_accuracy=0.05, confidence_level=0.95)
    
    z_score = np.sqrt(2.0) * erfinv(0.95)
    expected_val = (z_score / 0.05) ** 2
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.full_like(exposures, expected_val))


# ==========================================
# 3. BOUNDARY & LIMIT TESTS (Pruebas de Límites)
# ==========================================
def test_boundary_full_credibility_weight():
    """Boundary test: When Z = 1.0, premium equals observed experience."""
    assert partial_credibility_premium(1000.0, 1400.0, 1.0) == pytest.approx(1400.0)


def test_boundary_zero_credibility_weight():
    """Boundary test: When Z = 0.0, premium equals collective prior."""
    assert partial_credibility_premium(1000.0, 1400.0, 0.0) == pytest.approx(1000.0)


def test_boundary_target_accuracy_limits():
    """Boundary test: Stricter target accuracy (smaller k) requires exponentially larger standards."""
    std_loose = partial_credibility_standard(10.0, target_accuracy=0.10)
    std_strict = partial_credibility_standard(10.0, target_accuracy=0.01)
    
    assert std_strict > (std_loose * 10.0)


# ==========================================
# 4. MATHEMATICAL PROPERTIES TESTS (Propiedades Matemáticas)
# ==========================================
def test_property_convex_combination_bounds():
    """Property test: Credibility premium must always lie between collective prior and experience mean."""
    mu = 1500.0
    x_bar = 1200.0
    
    for z in [0.1, 0.3, 0.5, 0.7, 0.9]:
        res = partial_credibility_premium(mu, x_bar, z)
        # Since x_bar < mu, premium must be bounded between x_bar and mu
        assert x_bar <= res <= mu


def test_property_monotonicity_with_experience():
    """Property test: As observed experience increases, credibility premium increases monotonically."""
    mu = 1000.0
    z = 0.5
    
    res_low = partial_credibility_premium(mu, 1100.0, z)
    res_high = partial_credibility_premium(mu, 1500.0, z)
    
    assert res_high > res_low


# ==========================================
# 5. INTEGRATION & WRAPPER TESTS (Pruebas de Integración)
# ==========================================
def test_integrated_evaluation_wrapper():
    """Integration test: Verify wrapper dictionary outputs for teammate interoperability."""
    res = integrated_partial_credibility_evaluation(
        exposure_count=500.0,
        observed_mean=1150.0,
        collective_base=1000.0
    )
    assert isinstance(res, dict)
    assert "full_credibility_standard" in res
    assert "credibility_weight_z" in res
    assert "adjusted_premium" in res
    assert res["adjusted_premium"] > 1000.0
    assert 0.0 <= res["credibility_weight_z"] <= 1.0


# ==========================================
# 6. DOMAIN ERROR & EXCEPTION TESTS (Pruebas de Errores)
# ==========================================
def test_invalid_domain_errors():
    """Invalid-input test: Out-of-bound parameters must trigger ValueError."""
    # Negative collective premium or experience mean
    with pytest.raises(ValueError):
        partial_credibility_premium(-100.0, 500.0, 0.5)

    with pytest.raises(ValueError):
        partial_credibility_premium(1000.0, -50.0, 0.5)

    # Credibility weight outside [0, 1]
    with pytest.raises(ValueError):
        partial_credibility_premium(1000.0, 500.0, -0.1)

    with pytest.raises(ValueError):
        partial_credibility_premium(1000.0, 500.0, 1.2)

    # Invalid accuracy or confidence bounds
    with pytest.raises(ValueError):
        partial_credibility_standard(100.0, target_accuracy=1.5)

    with pytest.raises(ValueError):
        partial_credibility_standard(100.0, confidence_level=0.0)

    # Negative expected claims
    with pytest.raises(ValueError):
        partial_credibility_standard(-10.0)