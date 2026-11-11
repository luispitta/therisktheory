"""Comprehensive test suite for ruin theory and solvency models.

Topic 5: Ruin Theory and Solvency Models.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np
import pytest
from therisktheory.ruin_theory.ruin_models import (
    adjustment_coefficient_exponential_severity,
    cramer_lundberg_ruin_approximation,
    lundberg_ruin_bound,
)


# ==========================================
# 1. ANALYTICAL TESTS (Pruebas Analíticas)
# ==========================================
def test_analytical_adjustment_coefficient():
    """Analytical test: Verify adjustment coefficient against hand-calculated values."""
    theta = 0.2  # 20% safety loading
    beta = 50.0  # Mean severity

    r = adjustment_coefficient_exponential_severity(theta, beta)
    # Expected: R = 0.2 / (50.0 * 1.2) = 0.2 / 60.0 = 1/300 ≈ 0.0033333
    expected = 0.2 / (50.0 * 1.2)
    assert r == pytest.approx(expected)


def test_analytical_lundberg_bound():
    """Analytical test: Verify Lundberg bound calculation for given surplus and R."""
    u = 100.0
    r = 0.01
    bound = lundberg_ruin_bound(u, r)
    assert bound == pytest.approx(np.exp(-1.0))


# ==========================================
# 2. VECTORIZATION TESTS (Pruebas Vectorizadas)
# ==========================================
def test_vectorization_ruin_bound():
    """Vectorization test: Verify array inputs for initial surplus."""
    u_array = np.array([0.0, 50.0, 100.0, 200.0])
    r = 0.02

    bounds = lundberg_ruin_bound(u_array, r)
    expected = np.exp(-r * u_array)

    assert isinstance(bounds, np.ndarray)
    np.testing.assert_allclose(bounds, expected)


# ==========================================
# 3. BOUNDARY & LIMIT TESTS (Pruebas de Límites)
# ==========================================
def test_boundary_zero_surplus():
    """Boundary test: When initial surplus u = 0, Lundberg bound must equal 1.0."""
    bound = lundberg_ruin_bound(0.0, 0.05)
    assert bound == pytest.approx(1.0)


def test_boundary_infinite_surplus_tendency():
    """Boundary test: As initial surplus approaches infinity, ruin probability bound approaches zero."""
    bound_high = lundberg_ruin_bound(10000.0, 0.05)
    assert bound_high < 1e-10


# ==========================================
# 4. MATHEMATICAL PROPERTIES TESTS (Propiedades Matemáticas)
# ==========================================
def test_property_ruin_bound_monotonicity():
    """Property test: Ruin bound must decrease monotonically as initial surplus u increases."""
    r = 0.01
    u1, u2, u3 = 10.0, 50.0, 100.0

    b1 = lundberg_ruin_bound(u1, r)
    b2 = lundberg_ruin_bound(u2, r)
    b3 = lundberg_ruin_bound(u3, r)

    assert b1 > b2 > b3


# ==========================================
# 5. DOMAIN ERROR TESTS (Pruebas de Errores)
# ==========================================
def test_invalid_domain_errors():
    """Invalid-input test: Negative or zero parameters must raise ValueError."""
    with pytest.raises(ValueError):
        adjustment_coefficient_exponential_severity(-0.1, 50.0)

    with pytest.raises(ValueError):
        lundberg_ruin_bound(-10.0, 0.01)

    with pytest.raises(ValueError):
        cramer_lundberg_ruin_approximation(50.0, 0.2, -5.0)
