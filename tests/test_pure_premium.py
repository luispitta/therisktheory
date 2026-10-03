import numpy as np
import pytest

from therisktheory.solvency_pricing import pure_premium


def test_pure_premium_scalar() -> None:
    result = pure_premium(2.0, 500.0)
    assert np.asarray(result).item() == pytest.approx(1000.0)


def test_pure_premium_vectorized() -> None:
    frequency = np.array([0.5, 1.0, 2.0])
    severity = np.array([1000.0, 750.0, 500.0])
    expected = np.array([500.0, 750.0, 1000.0])

    np.testing.assert_allclose(pure_premium(frequency, severity), expected)


def test_pure_premium_broadcasting() -> None:
    frequency = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        pure_premium(frequency, 100.0),
        np.array([100.0, 200.0, 300.0]),
    )


@pytest.mark.parametrize(
    ("frequency", "severity"),
    [(-1.0, 100.0), (1.0, -100.0)],
)
def test_pure_premium_rejects_negative_inputs(frequency: float, severity: float) -> None:
    with pytest.raises(ValueError):
        pure_premium(frequency, severity)
