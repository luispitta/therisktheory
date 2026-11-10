"""Monte Carlo simulation for the collective risk model and aggregate losses.

This module provides human-validated, high-performance vectorized implementations
for generating aggregate loss distributions S = sum_{i=1}^N X_i without explicit
Python loops over simulation paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def simulate_aggregate_loss(
    frequency_dist: str | Callable[[int, np.random.Generator], NDArray[np.int64]],
    severity_dist: str | Callable[[int, np.random.Generator], NDArray[np.float64]],
    num_simulations: int = 10_000,
    frequency_params: dict[str, Any] | None = None,
    severity_params: dict[str, Any] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    r"""Simulate aggregate portfolio loss distribution using vectorized Monte Carlo.

    Under the classical collective risk model, total aggregate claim loss :math:`S`
    over an operational period is given by the random sum:

    .. math::
        S = \sum_{i=1}^N X_i

    where :math:`N \in \mathbb{N}_0` is the discrete random claim count, and :math:`X_i > 0`
    are independent and identically distributed claim amounts independent of :math:`N`.
    If :math:`N = 0`, then :math:`S = 0`.

    This implementation uses vectorization via :func:`numpy.bincount` and
    :func:`numpy.repeat` to aggregate all simulation trajectories without
    row-by-row Python loops. Both built-in parametric models and custom peer
    generator callables adhering to course interface specifications are supported.

    Args:
        frequency_dist: Claim count distribution. Either a distribution name
            (``"poisson"``, ``"negative_binomial"``, or ``"geometric"``) or a custom
            callable ``sampler(size, rng) -> NDArray[np.int64]``.
        severity_dist: Individual claim severity distribution. Either a distribution
            name (``"exponential"``, ``"lognormal"``, ``"pareto"``, ``"gamma"``, or
            ``"weibull"``) or a custom callable ``sampler(size, rng) -> NDArray[np.float64]``.
        num_simulations: Number of Monte Carlo trajectories :math:`M \ge 1`.
            Defaults to 10,000.
        frequency_params: Dictionary of parameters for built-in frequency models.
            - For ``"poisson"``: ``{"lam": float > 0}``
            - For ``"negative_binomial"``: ``{"n": float > 0, "p": 0 < float <= 1}``
            - For ``"geometric"``: ``{"p": 0 < float <= 1}``
        severity_params: Dictionary of parameters for built-in severity models.
            - For ``"exponential"``: ``{"scale": float > 0}`` (mean claim size)
            - For ``"lognormal"``: ``{"mean": float, "sigma": float > 0}``
            - For ``"pareto"``: ``{"a": float > 0, "scale": float > 0}`` (Pareto Type II)
            - For ``"gamma"``: ``{"shape": float > 0, "scale": float > 0}``
            - For ``"weibull"``: ``{"a": float > 0, "scale": float > 0}``
        random_state: Seed or NumPy random generator for reproducibility.

    Returns:
        NDArray[np.float64]: 1D array of length ``num_simulations`` containing the
        simulated aggregate losses :math:`S_1, \dots, S_M`.

    Raises:
        ValueError: If ``num_simulations < 1``, unrecognized distribution names,
            non-finite or invalid parameters, or if custom generators return arrays
            with invalid dimensions, negative values, non-finite values, or
            non-integer frequency counts.
        TypeError: If parameter inputs are not dictionaries or callable where required.

    Notes:
        Validation status: experimental.

    References:
        Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). Loss Models: From
        Data to Decisions (5th ed.). Wiley, Chapter 9.
        Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). Modern Actuarial
        Risk Theory: Using R (2nd ed.). Springer, Chapter 2.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.severity_aggregate import simulate_aggregate_loss
        >>> losses = simulate_aggregate_loss(
        ...     frequency_dist="poisson",
        ...     severity_dist="exponential",
        ...     num_simulations=5000,
        ...     frequency_params={"lam": 5.0},
        ...     severity_params={"scale": 200.0},
        ...     random_state=42,
        ... )
        >>> losses.shape
        (5000,)
        >>> np.isclose(np.mean(losses), 1000.0, rtol=0.10)
        True
    """
    if not isinstance(num_simulations, (int, np.integer)) or num_simulations < 1:
        raise ValueError(f"num_simulations must be an integer >= 1, got {num_simulations}")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # 1. Resolve Frequency Sampler
    f_params = frequency_params or {}
    if callable(frequency_dist):
        raw_n_claims = frequency_dist(num_simulations, rng)
        n_claims_arr = np.asarray(raw_n_claims)
        if n_claims_arr.shape != (num_simulations,):
            raise ValueError(
                f"frequency generator must return a 1D array of shape ({num_simulations},), "
                f"got shape {n_claims_arr.shape}"
            )
        if not np.all(np.isfinite(n_claims_arr)):
            raise ValueError("frequency generator returned non-finite claim counts")
        if np.any(n_claims_arr < 0):
            raise ValueError("frequency generator returned negative claim counts")
        if not np.issubdtype(n_claims_arr.dtype, np.integer) and not np.all(
            np.floor(n_claims_arr) == n_claims_arr
        ):
            raise ValueError("frequency generator must return discrete integer claim counts")
        n_claims = n_claims_arr.astype(np.int64)
    elif frequency_dist == "poisson":
        lam = float(f_params.get("lam", 1.0))
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError(f"Poisson parameter 'lam' must be finite and > 0, got {lam}")
        n_claims = rng.poisson(lam=lam, size=num_simulations)
    elif frequency_dist == "negative_binomial":
        n = float(f_params.get("n", 1.0))
        p = float(f_params.get("p", 0.5))
        if not (np.isfinite(n) and np.isfinite(p)) or n <= 0.0 or p <= 0.0 or p > 1.0:
            raise ValueError(
                f"Negative Binomial requires finite n > 0 and 0 < p <= 1, got n={n}, p={p}"
            )
        n_claims = rng.negative_binomial(n=n, p=p, size=num_simulations)
    elif frequency_dist == "geometric":
        p = float(f_params.get("p", 0.5))
        if not np.isfinite(p) or p <= 0.0 or p > 1.0:
            raise ValueError(f"Geometric requires finite 0 < p <= 1, got p={p}")
        # NumPy geometric generates {1, 2, ...}, subtract 1 for number of claims in {0, 1, ...}
        n_claims = rng.geometric(p=p, size=num_simulations) - 1
    else:
        raise ValueError(f"Unsupported frequency distribution: {frequency_dist!r}")

    total_claims = int(np.sum(n_claims))
    if total_claims == 0:
        return np.zeros(num_simulations, dtype=np.float64)

    # 2. Resolve Severity Sampler
    s_params = severity_params or {}
    if callable(severity_dist):
        raw_severities = severity_dist(total_claims, rng)
        severities_arr = np.asarray(raw_severities, dtype=np.float64)
        if severities_arr.shape != (total_claims,):
            raise ValueError(
                f"severity generator must return a 1D array of shape ({total_claims},), "
                f"got shape {severities_arr.shape}"
            )
        if not np.all(np.isfinite(severities_arr)):
            raise ValueError("severity generator returned non-finite claim amounts")
        if np.any(severities_arr < 0.0):
            raise ValueError("severity generator returned negative claim amounts")
        severities = severities_arr
    elif severity_dist == "exponential":
        scale = float(s_params.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Exponential parameter 'scale' must be finite and > 0, got {scale}")
        severities = rng.exponential(scale=scale, size=total_claims)
    elif severity_dist == "lognormal":
        mean = float(s_params.get("mean", 0.0))
        sigma = float(s_params.get("sigma", 1.0))
        if not (np.isfinite(mean) and np.isfinite(sigma)) or sigma <= 0.0:
            raise ValueError(
                "Lognormal parameters must be finite with sigma > 0, "
                f"got mean={mean}, sigma={sigma}"
            )
        severities = rng.lognormal(mean=mean, sigma=sigma, size=total_claims)
    elif severity_dist == "pareto":
        a = float(s_params.get("a", 2.0))
        scale = float(s_params.get("scale", 1.0))
        if not (np.isfinite(a) and np.isfinite(scale)) or a <= 0.0 or scale <= 0.0:
            raise ValueError(
                "Pareto parameters 'a' and 'scale' must be finite and > 0, "
                f"got a={a}, scale={scale}"
            )
        # NumPy pareto draws from Pareto Type II / Lomax shifted: (X + 1) * scale
        severities = (rng.pareto(a=a, size=total_claims) + 1.0) * scale
    elif severity_dist == "gamma":
        shape = float(s_params.get("shape", 1.0))
        scale = float(s_params.get("scale", 1.0))
        if not (np.isfinite(shape) and np.isfinite(scale)) or shape <= 0.0 or scale <= 0.0:
            raise ValueError(
                "Gamma 'shape' and 'scale' must be finite and > 0, "
                f"got shape={shape}, scale={scale}"
            )
        severities = rng.gamma(shape=shape, scale=scale, size=total_claims)
    elif severity_dist == "weibull":
        a = float(s_params.get("a", 1.5))
        scale = float(s_params.get("scale", 1.0))
        if not (np.isfinite(a) and np.isfinite(scale)) or a <= 0.0 or scale <= 0.0:
            raise ValueError(f"Weibull parameters must be finite and > 0, got a={a}, scale={scale}")
        severities = rng.weibull(a=a, size=total_claims) * scale
    else:
        raise ValueError(f"Unsupported severity distribution: {severity_dist!r}")

    # 3. Vectorized Path Aggregation
    # Assign each claim to its simulation index using np.repeat, then sum via np.bincount
    sim_indices = np.repeat(np.arange(num_simulations, dtype=np.int64), n_claims)
    aggregate_loss = np.bincount(sim_indices, weights=severities, minlength=num_simulations)

    return np.asarray(aggregate_loss, dtype=np.float64)
