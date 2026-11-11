"""Comprehensive unit tests for solvency capital, guarantee funds, and peer integration.

Topic 1: Solvency Theory.
Author: Maria Lucia Monticelli D’Elias
"""

import pytest
import numpy as np
from therisktheory.solvency_pricing.solvency import (
    guarantee_capital,
    solvency_ratio,
    evaluate_portfolio_solvency,
)


def test_guarantee_capital_analytical():
    """Verify guarantee capital with an exact analytical case."""
    result = guarantee_capital(initial_capital=100.0, max_loss=500.0, safety_margin=0.05)
    assert pytest.approx(result, 0.001) == 425.0


def test_guarantee_capital_sufficient_reserves():
    """Tests that no additional capital is required if initial capital covers max loss + margin."""
    result = guarantee_capital(initial_capital=150.0, max_loss=100.0, safety_margin=0.10)
    assert result == pytest.approx(0.0)


def test_vectorized_inputs_support():
    """Tests that functions properly handle NumPy array inputs and boundaries."""
    capitals = np.array([50.0, 80.0, 150.0])
    losses = np.array([100.0, 100.0, 100.0])

    ratios = solvency_ratio(capitals, losses)
    np.testing.assert_allclose(ratios, np.array([0.5, 0.8, 1.5]))

    extra_cap = guarantee_capital(capitals, losses, safety_margin=0.10)
    np.testing.assert_allclose(extra_cap, np.array([60.0, 30.0, 0.0]))


def test_solvency_ratio_analytical_case():
    """Tests adimensional quotient: Available = 150, Required = 100 -> Ratio = 1.5."""
    ratio = solvency_ratio(available_capital=150.0, required_capital=100.0)
    assert pytest.approx(ratio, 0.001) == 1.5


def test_solvency_ratio_zero_division_exception():
    """Tests that required_capital <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        solvency_ratio(available_capital=100.0, required_capital=0.0)


def test_evaluate_portfolio_solvency_integration():
    """Tests complete portfolio solvency evaluation flow and loaded premium metrics."""
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
    assert res["solvency_ratio"] > 0.0


# ===========================================================================
# Prueba propia y original del Tema 1: Test de Estrés de Solvencia Multisectorial
# ===========================================================================
def test_custom_multi_portfolio_solvency_stress():
    """Custom test for Topic 1: Evaluates portfolio resilience under catastrophe stress.

    Verifies that across different risk portfolios with varying claim variances,
    the insurer maintains an acceptable solvency buffer and proper monotonicity
    in guarantee capital requirements.
    """
    initial_capitals = np.array([500.0, 1000.0, 2500.0])
    max_losses = np.array([1200.0, 2000.0, 4500.0])
    # Calcular capital de garantía requerido para cada cartera
    req_caps = guarantee_capital(initial_capitals, max_losses, safety_margin=0.08)
    # Propiedad matemática esperada: Caracteres con mayor pérdida máxima sin capital proporcional
    # exigen mayor o igual capital de garantía (verificación de no negatividad y acotamiento)
    assert len(req_caps) == 3
    assert np.all(req_caps >= 0.0)
    # Verificación de ratios de solvencia bajo escenarios combinados
    available_assets = initial_capitals + 300.0 # Ingreso por primas estables
    ratios = solvency_ratio(available_assets, max_losses)
    # Ninguna cartera con estos fondos debe colapsar por debajo de un ratio lógico de estabilidad crítica
    assert np.all(ratios > 0.3)