import numpy as np
import pytest

from therisktheory.solvency_pricing.loadings import (
    loaded_premium,
    premium_principle_checks,
    safety_loading,
)


def test_safety_loading_basic():
    assert safety_loading(600.0, 660.0) == pytest.approx(60.0)


def test_safety_loading_zero():
    assert safety_loading(600.0, 600.0) == pytest.approx(0.0)


def test_safety_loading_vectorization():
    result = safety_loading(
        np.array([100.0, 200.0]),
        np.array([110.0, 230.0]),
    )
    np.testing.assert_allclose(result, [10.0, 30.0])


def test_safety_loading_rejects_premium_below_expected_loss():
    with pytest.raises(ValueError):
        safety_loading(600.0, 590.0)


def test_safety_loading_rejects_negative_values():
    with pytest.raises(ValueError):
        safety_loading(-10.0, 20.0)


def test_loaded_premium_expected_value():
    assert loaded_premium(600.0, 0.10) == pytest.approx(660.0)


def test_loaded_premium_variance():
    result = loaded_premium(
        600.0,
        0.0005,
        principle="variance",
        variance=240000.0,
    )
    assert result == pytest.approx(720.0)


def test_loaded_premium_standard_deviation():
    result = loaded_premium(
        600.0,
        0.10,
        principle="standard_deviation",
        variance=10000.0,
    )
    assert result == pytest.approx(610.0)


def test_loaded_premium_requires_variance():
    with pytest.raises(ValueError):
        loaded_premium(600.0, 0.10, principle="variance")


def test_loaded_premium_rejects_unknown_principle():
    with pytest.raises(ValueError):
        loaded_premium(600.0, 0.10, principle="unknown")


def test_loaded_premium_rejects_unused_variance():
    with pytest.raises(ValueError):
        loaded_premium(600.0, 0.10, variance=100.0)


def test_premium_checks_expected_value():
    result = premium_principle_checks(
        600.0,
        0.10,
        maximum_loss=700.0,
    )

    assert bool(result["non_negative_loading"])
    assert bool(result["no_rip_off"])
    assert not bool(result["translation_consistency"])
    assert bool(result["independent_additivity"])


def test_premium_checks_variance_principle():
    result = premium_principle_checks(
        600.0,
        0.0005,
        principle="variance",
        variance=240000.0,
        maximum_loss=800.0,
    )

    assert bool(result["translation_consistency"])
    assert bool(result["independent_additivity"])


def test_premium_checks_standard_deviation_additivity():
    result = premium_principle_checks(
        600.0,
        0.10,
        principle="standard_deviation",
        variance=10000.0,
    )

    assert not bool(result["independent_additivity"])


def test_premium_checks_zero_loading_is_additive():
    result = premium_principle_checks(
        600.0,
        0.0,
        principle="standard_deviation",
        variance=10000.0,
    )

    assert bool(result["independent_additivity"])


def test_premium_checks_requires_positive_cash_shift():
    with pytest.raises(ValueError):
        premium_principle_checks(600.0, 0.10, cash_shift=0.0)
