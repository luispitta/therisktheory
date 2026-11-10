"""Unit tests for Monte Carlo collective aggregate loss simulation."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.severity_aggregate import (
    optimal_reinsurance,
    simulate_aggregate_loss,
    technical_result_variance,
)


def test_simulate_aggregate_loss_poisson_exponential() -> None:
    # E[N] = 10, E[X] = 100 -> E[S] = 1000
    # Var(N) = 10, Var(X) = 10000 -> Var(S) = 10*10000 + 10*10000 = 200000
    sims = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="exponential",
        num_simulations=20_000,
        frequency_params={"lam": 10.0},
        severity_params={"scale": 100.0},
        random_state=42,
    )
    assert sims.shape == (20_000,)
    assert np.all(sims >= 0.0)
    assert np.mean(sims) == pytest.approx(1000.0, rel=0.05)


def test_simulate_aggregate_loss_negative_binomial_lognormal() -> None:
    sims = simulate_aggregate_loss(
        frequency_dist="negative_binomial",
        severity_dist="lognormal",
        num_simulations=10_000,
        frequency_params={"n": 5.0, "p": 0.5},
        severity_params={"mean": 4.0, "sigma": 0.5},
        random_state=123,
    )
    assert sims.shape == (10_000,)
    assert np.all(sims >= 0.0)


def test_simulate_aggregate_loss_geometric_gamma() -> None:
    sims = simulate_aggregate_loss(
        frequency_dist="geometric",
        severity_dist="gamma",
        num_simulations=5_000,
        frequency_params={"p": 0.25},
        severity_params={"shape": 2.0, "scale": 50.0},
        random_state=99,
    )
    assert sims.shape == (5_000,)
    assert np.all(sims >= 0.0)


def test_simulate_aggregate_loss_pareto_and_weibull() -> None:
    sims_pareto = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="pareto",
        num_simulations=5_000,
        frequency_params={"lam": 2.0},
        severity_params={"a": 3.0, "scale": 50.0},
        random_state=42,
    )
    assert sims_pareto.shape == (5_000,)

    sims_weibull = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="weibull",
        num_simulations=5_000,
        frequency_params={"lam": 2.0},
        severity_params={"a": 1.5, "scale": 100.0},
        random_state=42,
    )
    assert sims_weibull.shape == (5_000,)


def test_simulate_aggregate_loss_custom_callables() -> None:
    def custom_freq(size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.binomial(n=10, p=0.3, size=size)

    def custom_sev(size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=50.0, high=150.0, size=size)

    sims = simulate_aggregate_loss(
        frequency_dist=custom_freq,
        severity_dist=custom_sev,
        num_simulations=5_000,
        random_state=7,
    )
    assert sims.shape == (5_000,)
    # E[N] = 3.0, E[X] = 100.0 -> E[S] = 300.0
    assert np.mean(sims) == pytest.approx(300.0, rel=0.08)


def test_simulate_aggregate_loss_reproducibility() -> None:
    sims_1 = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="exponential",
        num_simulations=1000,
        frequency_params={"lam": 5.0},
        severity_params={"scale": 20.0},
        random_state=12345,
    )
    sims_2 = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="exponential",
        num_simulations=1000,
        frequency_params={"lam": 5.0},
        severity_params={"scale": 20.0},
        random_state=12345,
    )
    np.testing.assert_array_equal(sims_1, sims_2)


def test_simulate_aggregate_loss_zero_claims() -> None:
    # Custom frequency returning zero claims
    sims = simulate_aggregate_loss(
        frequency_dist=lambda size, rng: np.zeros(size, dtype=np.int64),
        severity_dist="exponential",
        num_simulations=500,
    )
    assert np.all(sims == 0.0)


def test_joint_peer_frequency_and_severity_interfaces() -> None:
    """Joint integration test combining peer frequency and severity interfaces."""

    # 1. Custom frequency sampler implementing peer counting process interface
    def peer_frequency_sampler(size: int, rng: np.random.Generator) -> np.ndarray:
        # Negative binomial count interface (r = 4.0, p = 0.5)
        # E[N] = r*(1-p)/p = 4.0 * 0.5 / 0.5 = 4.0
        # Var(N) = r*(1-p)/p^2 = 4.0 * 0.5 / 0.25 = 8.0
        return rng.negative_binomial(n=4.0, p=0.5, size=size)

    # 2. Custom severity sampler implementing peer severity interface
    def peer_severity_sampler(size: int, rng: np.random.Generator) -> np.ndarray:
        # Lognormal severity interface: mu = 4.0, sigma = 0.5
        # E[X] = exp(4.0 + 0.5^2 / 2) = exp(4.125) ~ 61.8678
        # Var(X) = exp(2*4.0 + 0.25) * (exp(0.25) - 1) ~ 3827.63 * 0.284025 ~ 1087.16
        return rng.lognormal(mean=4.0, sigma=0.5, size=size)

    num_sims = 15_000
    sims = simulate_aggregate_loss(
        frequency_dist=peer_frequency_sampler,
        severity_dist=peer_severity_sampler,
        num_simulations=num_sims,
        random_state=42,
    )

    assert sims.shape == (num_sims,)
    assert np.all(sims >= 0.0)

    # Theoretical moments under Wald's identity
    e_n = 4.0
    v_n = 8.0
    e_x = np.exp(4.0 + 0.5**2 / 2.0)
    v_x = np.exp(2.0 * 4.0 + 0.5**2) * (np.exp(0.5**2) - 1.0)
    expected_mean = e_n * e_x
    expected_var = e_n * v_x + v_n * (e_x**2)

    assert np.mean(sims) == pytest.approx(expected_mean, rel=0.08)
    assert np.var(sims) == pytest.approx(expected_var, rel=0.15)

    # Joint evaluation with reinsurance stability
    ret_var = technical_result_variance(
        retention_rate=0.7,
        expected_frequency=e_n,
        variance_frequency=v_n,
        expected_severity=e_x,
        variance_severity=v_x,
    )
    assert float(ret_var) == pytest.approx((0.7**2) * expected_var)

    opt_alpha = optimal_reinsurance(
        expected_loss=expected_mean,
        variance_loss=expected_var,
        reinsurer_loading=0.15,
        target_variance=0.5 * expected_var,
        criterion="target_variance",
    )
    assert float(opt_alpha) == pytest.approx(np.sqrt(0.5))


def test_custom_generator_dimension_checks() -> None:
    """Verify that custom generators are validated for correct 1D array shapes."""

    # Frequency generator returning 2D array
    def bad_freq_2d(size: int, rng: np.random.Generator) -> np.ndarray:
        return np.ones((size, 2), dtype=np.int64)

    with pytest.raises(ValueError, match="1D array of shape"):
        simulate_aggregate_loss(
            frequency_dist=bad_freq_2d,
            severity_dist="exponential",
            num_simulations=10,
        )

    # Frequency generator returning wrong length
    def bad_freq_len(size: int, rng: np.random.Generator) -> np.ndarray:
        return np.ones(size + 5, dtype=np.int64)

    with pytest.raises(ValueError, match="1D array of shape"):
        simulate_aggregate_loss(
            frequency_dist=bad_freq_len,
            severity_dist="exponential",
            num_simulations=10,
        )

    # Severity generator returning 2D array
    def bad_sev_2d(size: int, rng: np.random.Generator) -> np.ndarray:
        return np.ones((size, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="1D array of shape"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.full(size, 2, dtype=np.int64),
            severity_dist=bad_sev_2d,
            num_simulations=5,
        )

    # Severity generator returning wrong length
    def bad_sev_len(size: int, rng: np.random.Generator) -> np.ndarray:
        return np.ones(size + 2, dtype=np.float64)

    with pytest.raises(ValueError, match="1D array of shape"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.full(size, 2, dtype=np.int64),
            severity_dist=bad_sev_len,
            num_simulations=5,
        )


def test_custom_generator_value_checks() -> None:
    """Verify that custom generators reject non-finite, negative, or non-integer values."""

    # Non-finite frequency claims
    with pytest.raises(ValueError, match="non-finite claim counts"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([1, np.nan, 3]),
            severity_dist="exponential",
            num_simulations=3,
        )

    # Non-integer frequency claims (e.g. floats with decimals)
    with pytest.raises(ValueError, match="discrete integer claim counts"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([1.5, 2.0, 3.0]),
            severity_dist="exponential",
            num_simulations=3,
        )

    # Non-finite severity values
    with pytest.raises(ValueError, match="non-finite claim amounts"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([2]),
            severity_dist=lambda size, rng: np.array([10.0, np.inf]),
            num_simulations=1,
        )

    # Negative severity values
    with pytest.raises(ValueError, match="negative claim amounts"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([2]),
            severity_dist=lambda size, rng: np.array([10.0, -5.0]),
            num_simulations=1,
        )


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_simulate_aggregate_loss_non_finite_params(bad_val: float) -> None:
    """Verify rejection of non-finite parameter inputs for built-in models."""
    with pytest.raises(ValueError):
        simulate_aggregate_loss("poisson", "exponential", frequency_params={"lam": bad_val})

    with pytest.raises(ValueError):
        simulate_aggregate_loss(
            "negative_binomial", "exponential", frequency_params={"n": bad_val, "p": 0.5}
        )

    with pytest.raises(ValueError):
        simulate_aggregate_loss(
            "negative_binomial", "exponential", frequency_params={"n": 5.0, "p": bad_val}
        )

    with pytest.raises(ValueError):
        simulate_aggregate_loss("geometric", "exponential", frequency_params={"p": bad_val})

    with pytest.raises(ValueError):
        simulate_aggregate_loss("poisson", "exponential", severity_params={"scale": bad_val})

    with pytest.raises(ValueError):
        simulate_aggregate_loss(
            "poisson", "lognormal", severity_params={"mean": bad_val, "sigma": 1.0}
        )

    with pytest.raises(ValueError):
        simulate_aggregate_loss(
            "poisson", "lognormal", severity_params={"mean": 0.0, "sigma": bad_val}
        )

    with pytest.raises(ValueError):
        simulate_aggregate_loss("poisson", "pareto", severity_params={"a": bad_val, "scale": 1.0})

    with pytest.raises(ValueError):
        simulate_aggregate_loss(
            "poisson", "gamma", severity_params={"shape": bad_val, "scale": 1.0}
        )

    with pytest.raises(ValueError):
        simulate_aggregate_loss("poisson", "weibull", severity_params={"a": bad_val, "scale": 1.0})


def test_simulate_aggregate_loss_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_simulations"):
        simulate_aggregate_loss("poisson", "exponential", num_simulations=0)

    with pytest.raises(ValueError, match="num_simulations"):
        simulate_aggregate_loss("poisson", "exponential", num_simulations=-5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported frequency"):
        simulate_aggregate_loss("unknown_freq", "exponential", num_simulations=10)

    with pytest.raises(ValueError, match="Unsupported severity"):
        simulate_aggregate_loss("poisson", "unknown_sev", num_simulations=10)

    with pytest.raises(ValueError, match="lam"):
        simulate_aggregate_loss("poisson", "exponential", frequency_params={"lam": -1.0})

    with pytest.raises(ValueError, match="Negative Binomial"):
        simulate_aggregate_loss(
            "negative_binomial", "exponential", frequency_params={"n": -1.0, "p": 0.5}
        )

    with pytest.raises(ValueError, match="Geometric"):
        simulate_aggregate_loss("geometric", "exponential", frequency_params={"p": -0.5})

    with pytest.raises(ValueError, match="scale"):
        simulate_aggregate_loss("poisson", "exponential", severity_params={"scale": -10.0})

    with pytest.raises(ValueError, match="sigma"):
        simulate_aggregate_loss(
            "poisson", "lognormal", severity_params={"mean": 0.0, "sigma": -1.0}
        )

    with pytest.raises(ValueError, match="Pareto"):
        simulate_aggregate_loss("poisson", "pareto", severity_params={"a": -2.0, "scale": 1.0})

    with pytest.raises(ValueError, match="Gamma"):
        simulate_aggregate_loss("poisson", "gamma", severity_params={"shape": -1.0, "scale": 1.0})

    with pytest.raises(ValueError, match="Weibull"):
        simulate_aggregate_loss("poisson", "weibull", severity_params={"a": -1.0, "scale": 1.0})

    with pytest.raises(ValueError, match="negative claim counts"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([-1, 2]),
            severity_dist="exponential",
            num_simulations=2,
        )
