import numpy as np
import pytest

from therisktheory.solvency import (
    evaluate_portfolio_solvency,
    guarantee_capital,
    solvency_ratio,
)


# 1. PRUEBAS ANALÍTICAS EXACTAS (Calculadas manualmente)
def test_guarantee_capital_analytical_case():
    """Tests exact analytical case: U_0 = 80, max_loss = 100, margin = 0.10 -> Target = 110, Extra = 30."""
    result = guarantee_capital(
        initial_capital=80.0, max_loss=100.0, safety_margin=0.10
    )
    assert result == pytest.approx(30.0)


def test_solvency_ratio_analytical_case():
    """Tests adimensional quotient: Available = 150, Required = 100 -> Ratio = 1.5."""
    ratio = solvency_ratio(available_capital=150.0, required_capital=100.0)
    assert ratio == pytest.approx(1.5)


# 2. PRUEBAS DE FRONTERA Y PROPIEDADES LÓGICAS
def test_guarantee_capital_sufficient_reserves():
    """Tests that no additional capital is required if initial capital covers max loss + margin."""
    result = guarantee_capital(
        initial_capital=150.0, max_loss=100.0, safety_margin=0.10
    )
    assert result == pytest.approx(0.0)


def test_vectorized_inputs_support():
    """Tests that functions properly handle NumPy array inputs."""
    capitals = np.array([50.0, 80.0, 150.0])
    losses = np.array([100.0, 100.0, 100.0])

    ratios = solvency_ratio(capitals, losses)
    np.testing.assert_allclose(ratios, np.array([0.5, 0.8, 1.5]))

    extra_cap = guarantee_capital(capitals, losses, safety_margin=0.10)
    np.testing.assert_allclose(extra_cap, np.array([60.0, 30.0, 0.0]))


# 3. PRUEBAS DE EXCEPCIONES Y MANEJO DE ERRORES DE DOMINIO
def test_invalid_negative_capital_exception():
    """Tests that negative inputs raise a clear ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        guarantee_capital(initial_capital=-10.0, max_loss=100.0)


def test_solvency_ratio_zero_division_exception():
    """Tests that required_capital <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        solvency_ratio(available_capital=100.0, required_capital=0.0)


# 4. PRUEBA DE INTEGRACIÓN DE PORTAFOLIO
def test_evaluate_portfolio_solvency_integration():
    """Tests complete portfolio solvency evaluation flow."""
    res = evaluate_portfolio_solvency(
        initial_capital=80.0,
        expected_frequency=10.0,
        average_severity=8.0,
        safety_loading=0.15,
        max_loss=120.0,
        safety_margin=0.10,
    )

    assert res["pure_premium"] == pytest.approx(80.0)
    assert res["loaded_premium"] == pytest.approx(92.0)
    assert res["total_available_capital"] == pytest.approx(172.0)
    assert res["solvency_ratio"] == pytest.approx(1.4333333333333333)
    assert res["guarantee_capital_required"] == pytest.approx(52.0)
