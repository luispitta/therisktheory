"""Unit tests for risk retention, division, and inflation adjustment models."""
from __future__ import annotations

import numpy as np
import pytest

from therisktheory.retention import (
    retained_loss,
    retention_capacity,
    inflate_losses,
    deflate_losses,
)


def test_retained_loss_boundaries() -> None:
    """Pruebas de frontera y valores estándar para la retención de siniestros."""
    assert retained_loss(0.0, 1000.0) == 0.0
    assert retained_loss(1500.0, 1000.0) == 1000.0
    assert retained_loss(500.0, 1000.0) == 500.0
    
    # Verificación vectorizada con NumPy
    losses = np.array([200.0, 1500.0])
    res = retained_loss(losses, 1000.0)
    np.testing.assert_allclose(res, [200.0, 1000.0])


def test_retention_capacity() -> None:
    """Prueba del cálculo de capacidad máxima de retención financiera."""
    assert retention_capacity(1000.0, 0.25) == 250.0


def test_inflation_deflation_inversion() -> None:
    """Demostración formal de inversión y reversibilidad entre inflación y deflación."""
    base = 1000.0
    rate = 0.05
    periods = 3
    inflated = inflate_losses(base, rate, periods)
    deflated = deflate_losses(inflated, rate, periods)
    np.testing.assert_allclose(deflated, base)


def test_inflation_edge_cases() -> None:
    """Prueba de casos límite y división por cero en deflate_losses (tasa -1 con períodos positivos)."""
    with pytest.raises(ValueError, match="Division by zero"):
        deflate_losses(100.0, -1.0, periods=2)


def test_invalid_types_and_values() -> None:
    """Pruebas de manejo de excepciones para NaN, infinitos, tipos no numéricos y valores negativos."""
    # Tipos incorrectos y NaN / Inf
    with pytest.raises(TypeError):
        retained_loss("100.0", 500.0)
    with pytest.raises(ValueError):
        retained_loss(np.nan, 500.0)
    with pytest.raises(ValueError):
        retained_loss(float('inf'), 500.0)
        
    # Pérdidas o límites negativos
    with pytest.raises(ValueError):
        retained_loss(-50.0, 500.0)
    with pytest.raises(ValueError):
        retention_capacity(1000.0, 1.5)  # Ratio mayor a 1.0 (100%)
        
    # Tasas de inflación inválidas o períodos no enteros
    with pytest.raises(ValueError):
        inflate_losses(100.0, -1.5)
    with pytest.raises(TypeError):
        inflate_losses(100.0, 0.05, periods=1.5)  # Período fraccionario no entero
