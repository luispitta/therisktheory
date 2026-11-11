
"""Comprehensive tests for aggregate loss models.

Cover analytical, boundary, vectorization, and mathematical properties.

Topic 3: Frequency, Severity, Aggregate Loss, Approximations, and Simulation.
Author: Maria Lucia Monticelli D'Elias
"""

import numpy as np
import pytest

from therisktheory.collective_risk.aggregate_loss import (
    aggregate_characteristic_function,
    aggregate_loss_moments,
    aggregate_mgf,
    collective_model_loss,
    individual_model_loss,
)

# ==========================================
# 1. ANALYTICAL TESTS
# ==========================================


def test_analytical_aggregate_loss_moments():
    """Verify aggregate loss moments against hand-calculated results.

    Given:
        E[N] = 10.0, Var(N) = 10.0 (Poisson)
        E[X] = 50.0, Var(X) = 2500.0

    Expected:
        Mean(S) = E[N] * E[X] = 500.0
        Var(S) = E[N]*Var(X) + Var(N)*(E[X])^2 = 50000.0
    """
    e_n, v_n, e_x, v_x = 10.0, 10.0, 50.0, 2500.0
    mean_s, var_s, std_s = aggregate_loss_moments(e_n, v_n, e_x, v_x)

    assert mean_s == pytest.approx(500.0)
    assert var_s == pytest.approx(50000.0)
    assert std_s == pytest.approx(np.sqrt(50000.0))


# ==========================================
# 2. VECTORIZATION TESTS
# ==========================================


def test_vectorization_aggregate_loss_moments():
    """Ensure scalar and array inputs return aligned vector results."""
    e_n = np.array([5.0, 10.0, 20.0])
    v_n = np.array([5.0, 10.0, 20.0])
    e_x = np.array([10.0, 50.0, 100.0])
    v_x = np.array([50.0, 2500.0, 10000.0])

    mean_s, var_s, std_s = aggregate_loss_moments(e_n, v_n, e_x, v_x)

    expected_means = e_n * e_x
    expected_vars = e_n * v_x + v_n * (e_x**2)

    assert isinstance(mean_s, np.ndarray)
    np.testing.assert_allclose(mean_s, expected_means)
    np.testing.assert_allclose(var_s, expected_vars)
    np.testing.assert_allclose(std_s, np.sqrt(expected_vars))


# ==========================================
# 3. BOUNDARY / LIMIT TESTS
# ==========================================


def test_boundary_zero_claims():
    """Verify zero frequency produces zero aggregate loss and variance."""
    mean_s, var_s, std_s = aggregate_loss_moments(0.0, 0.0, 50.0, 2500.0)

    assert mean_s == pytest.approx(0.0)
    assert var_s == pytest.approx(0.0)
    assert std_s == pytest.approx(0.0)


def test_invalid_input_domain_errors():
    """Verify invalid parameters and probabilities trigger ValueError."""
    with pytest.raises(ValueError):
        aggregate_loss_moments(-1.0, 5.0, 50.0, 100.0)

    with pytest.raises(ValueError):
        aggregate_mgf(0.1, -2.0, np.array([1.0]))

    with pytest.raises(ValueError):
        individual_model_loss(np.array([1.2, 0.5]), np.array([100.0, 200.0]))

    with pytest.raises(ValueError):
        individual_model_loss(np.array([0.1, 0.5]), np.array([-100.0, 200.0]))


# ==========================================
# 4. MATHEMATICAL PROPERTIES TESTS
# ==========================================


def test_property_monotonicity_with_severity():
    """Verify aggregate loss moments increase with severity mean."""
    en, vn = 5.0, 5.0
    vx = 100.0

    mean_low, var_low, _ = aggregate_loss_moments(en, vn, 10.0, vx)
    mean_high, var_high, _ = aggregate_loss_moments(en, vn, 50.0, vx)

    assert mean_high > mean_low
    assert var_high > var_low


def test_property_compound_mgf_neutral_element():
    """Verify the aggregate MGF equals one at the neutral element."""
    lam = 3.0
    t = 0.0

    # At t=0, the severity MGF equals 1, so the aggregate MGF is also 1.
    mgf_x = np.array([1.0])
    res = aggregate_mgf(t, lam, mgf_x)

    np.testing.assert_allclose(res, 1.0)


def test_property_individual_model_additivity():
    """Verify individual risk model expectations follow linear additivity."""
    probs = np.array([0.2, 0.3])
    amounts = np.array([100.0, 200.0])

    mean_ind, _ = individual_model_loss(probs, amounts)
    expected_sum = (0.2 * 100.0) + (0.3 * 200.0)

    assert mean_ind == pytest.approx(expected_sum)


# ==========================================
# 5. CHARACTERISTIC FUNCTION TESTS
# ==========================================


def test_characteristic_function_neutral_element():
    """The aggregate characteristic function equals one at t=0."""
    result = aggregate_characteristic_function(
        t=0.0,
        lambda_poisson=3.0,
        severity_cf_values=np.array([1.0 + 0.0j]),
    )

    np.testing.assert_allclose(result, 1.0 + 0.0j)


def test_characteristic_function_vectorization():
    """The compound Poisson characteristic function supports arrays."""
    lam = np.array([1.0, 2.0, 3.0])
    cf_x = np.array([0.9 + 0.1j, 0.8 + 0.2j, 0.7 + 0.3j])

    result = aggregate_characteristic_function(
        t=np.array([0.1, 0.2, 0.3]),
        lambda_poisson=lam,
        severity_cf_values=cf_x,
    )

    expected = np.exp(lam * (cf_x - 1.0))
    np.testing.assert_allclose(result, expected)


# ==========================================
# 6. COLLECTIVE MODEL TESTS
# ==========================================


def test_collective_model_loss_poisson_default():
    """The default count variance must equal the expected count."""
    result = collective_model_loss(
        expected_count=2.0,
        severity_mean=10.0,
        severity_variance=25.0,
    )

    assert result["mean"] == pytest.approx(20.0)
    assert result["variance"] == pytest.approx(250.0)
    assert result["std"] == pytest.approx(np.sqrt(250.0))


def test_collective_model_loss_explicit_count_variance():
    """An explicitly supplied count variance must be respected."""
    result = collective_model_loss(
        expected_count=2.0,
        severity_mean=10.0,
        severity_variance=25.0,
        count_variance=4.0,
    )

    assert result["mean"] == pytest.approx(20.0)
    assert result["variance"] == pytest.approx(450.0)
    assert result["std"] == pytest.approx(np.sqrt(450.0))


# ==========================================
# 7. NUMERICAL VALIDATION TESTS
# ==========================================


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_aggregate_moments_reject_non_finite_inputs(invalid_value):
    """Aggregate moments must reject NaN and infinite inputs."""
    with pytest.raises(ValueError):
        aggregate_loss_moments(invalid_value, 1.0, 10.0, 5.0)


def test_aggregate_moments_reject_incompatible_shapes():
    """Moment inputs must have broadcast-compatible shapes."""
    with pytest.raises(ValueError):
        aggregate_loss_moments(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            10.0,
            5.0,
        )


def test_aggregate_moments_detect_overflow():
    """Aggregate moment calculations must detect floating-point overflow."""
    with pytest.raises(OverflowError):
        aggregate_loss_moments(1e308, 1.0, 1e308, 1.0)


def test_aggregate_mgf_reject_non_finite_inputs():
    """The MGF must reject non-finite evaluation points and severity values."""
    with pytest.raises(ValueError):
        aggregate_mgf(np.nan, 1.0, 1.0)

    with pytest.raises(ValueError):
        aggregate_mgf(0.1, 1.0, np.inf)


def test_aggregate_mgf_detect_overflow():
    """The MGF must report overflow instead of returning infinity."""
    with pytest.raises(OverflowError):
        aggregate_mgf(1.0, 1000.0, 2.0)


def test_characteristic_function_reject_non_finite_inputs():
    """The characteristic function must reject non-finite values."""
    with pytest.raises(ValueError):
        aggregate_characteristic_function(
            0.1, 1.0, np.array([np.nan + 0.0j])
        )

    with pytest.raises(ValueError):
        aggregate_characteristic_function(
            np.inf, 1.0, np.array([1.0 + 0.0j])
        )


def test_characteristic_function_reject_incompatible_shapes():
    """Characteristic function inputs must be broadcast-compatible."""
    with pytest.raises(ValueError):
        aggregate_characteristic_function(
            np.array([0.1, 0.2]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0 + 0.0j]),
        )


def test_individual_model_reject_non_finite_inputs():
    """The individual model must reject NaN and infinite parameters."""
    with pytest.raises(ValueError):
        individual_model_loss(np.array([np.nan]), np.array([100.0]))

    with pytest.raises(ValueError):
        individual_model_loss(np.array([0.5]), np.array([np.inf]))


def test_individual_model_reject_incompatible_shapes():
    """Individual probabilities and amounts must be broadcast-compatible."""
    with pytest.raises(ValueError):
        individual_model_loss(
            np.array([0.2, 0.3]),
            np.array([100.0, 200.0, 300.0]),
        )


def test_individual_model_detect_overflow():
    """The individual model must detect overflow in its moments."""
    with pytest.raises(OverflowError):
        individual_model_loss(
            np.array([1.0]),
            np.array([1e308]),
        )


def test_collective_model_reject_non_finite_inputs():
    """Collective model parameters must be finite."""
    with pytest.raises(ValueError):
        collective_model_loss(np.nan, 10.0, 25.0)

    with pytest.raises(ValueError):
        collective_model_loss(2.0, np.inf, 25.0)


def test_collective_model_reject_array_parameters():
    """The collective model summary function requires scalar parameters."""
    with pytest.raises(ValueError):
        collective_model_loss(
            np.array([1.0, 2.0]),
            10.0,
            25.0,
        )
