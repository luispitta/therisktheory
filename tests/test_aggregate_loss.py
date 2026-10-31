"""Comprehensive tests for aggregate loss models covering analytical, boundary, 
vectorization, and mathematical properties.

Topic 3: Frequency, Severity, Aggregate Loss, Approximations, and Simulation.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np
import pytest
from therisktheory.collective_risk.aggregate_loss import (
    aggregate_loss_moments,
    aggregate_mgf,
    aggregate_characteristic_function,
    individual_model_loss,
    collective_model_loss,
)


# ==========================================
# 1. ANALYTICAL TESTS (Tests analíticos)
# ==========================================
def test_analytical_aggregate_loss_moments():
    """Analytical test: Verify aggregate loss mean and variance against hand-calculated results.
    
    Given:
        E[N] = 10.0, Var(N) = 10.0 (Poisson)
        E[X] = 50.0, Var(X) = 2500.0
    Expected:
        Mean(S) = E[N] * E[X] = 10 * 50 = 500.0
        Var(S) = E[N]*Var(X) + Var(N)*(E[X])^2 = 10*2500 + 10*(50^2) = 50000.0
    """
    e_n, v_n, e_x, v_x = 10.0, 10.0, 50.0, 2500.0
    mean_s, var_s, std_s = aggregate_loss_moments(e_n, v_n, e_x, v_x)
    
    assert mean_s == pytest.approx(500.0)
    assert var_s == pytest.approx(50000.0)
    assert std_s == pytest.approx(np.sqrt(50000.0))


# ==========================================
# 2. VECTORIZATION TESTS (Pruebas vectorizadas)
# ==========================================
def test_vectorization_aggregate_loss_moments():
    """Vectorization test: Ensure scalar and array inputs return properly aligned vector results."""
    e_n = np.array([5.0, 10.0, 20.0])
    v_n = np.array([5.0, 10.0, 20.0])
    e_x = np.array([10.0, 50.0, 100.0])
    v_x = np.array([50.0, 2500.0, 10000.0])

    mean_s, var_s, std_s = aggregate_loss_moments(e_n, v_n, e_x, v_x)
    
    expected_means = e_n * e_x
    expected_vars = e_n * v_x + v_n * (e_x ** 2)

    assert isinstance(mean_s, np.ndarray)
    np.testing.assert_allclose(mean_s, expected_means)
    np.testing.assert_allclose(var_s, expected_vars)
    np.testing.assert_allclose(std_s, np.sqrt(expected_vars))


# ==========================================
# 3. BOUNDARY / LIMIT TESTS (Pruebas de límites y fronteras)
# ==========================================
def test_boundary_zero_claims():
    """Boundary test: Zero expected claim frequency should yield zero aggregate loss and variance."""
    mean_s, var_s, std_s = aggregate_loss_moments(0.0, 0.0, 50.0, 2500.0)
    assert mean_s == pytest.approx(0.0)
    assert var_s == pytest.approx(0.0)
    assert std_s == pytest.approx(0.0)


def test_invalid_input_domain_errors():
    """Invalid-input test: Negative parameters or invalid probabilities must trigger ValueError."""
    # Negative frequency or severity parameters
    with pytest.raises(ValueError):
        aggregate_loss_moments(-1.0, 5.0, 50.0, 100.0)

    with pytest.raises(ValueError):
        aggregate_mgf(0.1, -2.0, np.array([1.0]))

    # Individual model probability outside [0, 1]
    with pytest.raises(ValueError):
        individual_model_loss(np.array([1.2, 0.5]), np.array([100.0, 200.0]))

    # Negative claim amounts in individual model
    with pytest.raises(ValueError):
        individual_model_loss(np.array([0.1, 0.5]), np.array([-100.0, 200.0]))


# ==========================================
# 4. MATHEMATICAL PROPERTIES TESTS (Propiedades matemáticas)
# ==========================================
def test_property_monotonicity_with_severity():
    """Property test: Aggregate loss mean and variance must increase monotonically with severity mean."""
    en, vn = 5.0, 5.0
    vx = 100.0
    
    mean_low, var_low, _ = aggregate_loss_moments(en, vn, 10.0, vx)
    mean_high, var_high, _ = aggregate_loss_moments(en, vn, 50.0, vx)
    
    assert mean_high > mean_low
    assert var_high > var_low


def test_property_compound_mgf_neutral_element():
    """Property test: MGF of aggregate loss evaluated at severity MGF = 1.0."""
    lam = 3.0
    t = 0.0
    # If severity MGF equals 1.0 (at t=0), MGF of aggregate loss must be exp(lam * (1 - 1)) = 1.0
    mgf_x = np.array([1.0])
    res = aggregate_mgf(t, lam, mgf_x)
    np.testing.assert_allclose(res, 1.0)


def test_property_individual_model_additivity():
    """Property test: Individual risk model expectations follow linear additivity."""
    probs = np.array([0.2, 0.3])
    amounts = np.array([100.0, 200.0])
    
    mean_ind, _ = individual_model_loss(probs, amounts)
    expected_sum = (0.2 * 100.0) + (0.3 * 200.0)
    
    assert mean_ind == pytest.approx(expected_sum)