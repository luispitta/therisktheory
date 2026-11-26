"""Comparative analysis between classical risk theory and De Finetti's framework.

Reference:
    De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
    Beard, R. E., Pentikäinen, T., and Pesonen, E. (1984). Risk Theory.
"""
from __future__ import annotations

import operator

import numpy as np
import numpy.typing as npt
from scipy.stats import binom

from therisktheory import _validation

def elementary_insurance_case(
    initial_capital: float | int | npt.ArrayLike,
    premium: float | int | npt.ArrayLike,
    claim_prob: float | int | npt.ArrayLike,
    claim_amount: float | int | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates an elementary insurance case under De Finetti's framework.

    Args:
        initial_capital: Starting reserve or capital amount(s).
        premium: Periodic premium collected.
        claim_prob: Probability of claim occurrence in [0, 1].
        claim_amount: Contractual sum insured or claim payout.

    Returns:
        Expected terminal position as a float64 NumPy array:
        E[U_1] = U_0 + P - (p * S)
    """
    try:
        cap_arr = np.asarray(initial_capital, dtype=np.float64)
        prem_arr = np.asarray(premium, dtype=np.float64)
        prob_arr = np.asarray(claim_prob, dtype=np.float64)
        claim_arr = np.asarray(claim_amount, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise TypeError("Inputs must be real numbers or numeric arrays.") from err

    if (
        not np.all(np.isfinite(cap_arr))
        or not np.all(np.isfinite(prem_arr))
        or not np.all(np.isfinite(prob_arr))
        or not np.all(np.isfinite(claim_arr))
    ):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(cap_arr < 0) or np.any(prem_arr < 0) or np.any(claim_arr < 0):
        raise ValueError(
            "Initial capital, premium, and claim amount must be non-negative."
        )

    if np.any(prob_arr < 0.0) or np.any(prob_arr > 1.0):
        raise ValueError("Claim probability must be bounded between 0.0 and 1.0.")

    return np.asarray(cap_arr + prem_arr - (prob_arr * claim_arr), dtype=np.float64)


def compare_classical_de_finetti(
    classical_ruin_prob: float | int | npt.ArrayLike,
    de_finetti_ruin_prob: float | int | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes absolute deviations between classical and De Finetti ruin estimates.

    Args:
        classical_ruin_prob: Ruin probabilities from classical Cramér-Lundberg models.
        de_finetti_ruin_prob: Ruin probabilities from De Finetti's discrete game models.

    Returns:
        Array of absolute differences for model comparison and criticism.
    """
    try:
        c_arr = np.asarray(classical_ruin_prob, dtype=np.float64)
        df_arr = np.asarray(de_finetti_ruin_prob, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise TypeError("Probabilities must be real numbers or numeric arrays.") from err

    if not np.all(np.isfinite(c_arr)) or not np.all(np.isfinite(df_arr)):
        raise ValueError("Probabilities must contain only finite values.")

    if np.any(c_arr < 0.0) or np.any(c_arr > 1.0) or np.any(df_arr < 0.0) or np.any(df_arr > 1.0):
        raise ValueError("Probabilities must be bounded between 0.0 and 1.0.")

    return np.abs(c_arr - df_arr)


def benchmark_ruin_models(
    initial_capital: float,
    lambda_rate: float,
    claim_mean: float,
    safety_loading: float,
    horizon_periods: int = 1,
    *,
    n_paths: int = 20_000,
    seed: int = 20261126,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compare finite-horizon ruin in two explicitly distinct models.

    The classical process has Poisson arrivals, exponential claim sizes,
    continuous premium income and strict ruin (surplus below zero).
    It is simulated at claim epochs, not just at observation dates.

    The discrete comparison uses unit gains/losses of size claim_mean,
    floor(lambda_rate * t) rounds, and odds q/p = 1/(1 + safety_loading).
    Its first-passage probability is exact by the ballot theorem. This
    is a pedagogical comparison, not a moment-matched equivalence.

    Args:
        initial_capital: Nonnegative starting surplus.
        lambda_rate: Positive Poisson arrival intensity.
        claim_mean: Positive exponential mean and discrete stake.
        safety_loading: Positive relative loading.
        horizon_periods: Number of integer observation times.
        n_paths: Number of independent classical simulation paths.
        seed: Reproducible NumPy random seed.

    Returns:
        Classical estimates, their binomial standard errors, exact discrete
        probabilities and absolute deviations, at times 1 through T.

    Raises:
        ValueError: If inputs are invalid or the teaching workload is too large.
        TypeError: If integer arguments are not integers or inputs are nonnumeric.
    """
    vals = []
    for name, value in (
        ("initial_capital", initial_capital),
        ("lambda_rate", lambda_rate),
        ("claim_mean", claim_mean),
        ("safety_loading", safety_loading),
    ):
        arr = _validation.as_nonnegative_finite_array(value, name)
        if arr.ndim != 0:
            raise ValueError(f"{name} must be scalar")
        vals.append(float(arr))
    u, lam, mu, theta = vals
    if min(lam, mu, theta) <= 0:
        raise ValueError("rate, claim mean and loading must be positive")
    if isinstance(horizon_periods, (bool, np.bool_)) or isinstance(
        n_paths, (bool, np.bool_)
    ):
        raise TypeError("horizon_periods and n_paths must be integers")
    periods = operator.index(horizon_periods)
    count = operator.index(n_paths)
    if not 1 <= periods <= 10_000 or not 1 <= count <= 1_000_000:
        raise ValueError("invalid horizon_periods or n_paths")
    end = lam * periods
    if not np.isfinite(end) or end > 1000 or count * end > 20_000_000:
        raise ValueError("benchmark workload exceeds the teaching limit")

    # Normalize time by lambda and money by the exponential mean.
    # Ruin is possible only immediately after a claim.
    times = lam * np.arange(1, periods + 1, dtype=np.float64)
    rng = np.random.default_rng(seed)
    capital = u / mu
    first_ruin = np.full(count, np.inf)
    elapsed = np.zeros(count)
    aggregate = np.zeros(count)
    active = np.arange(count)
    while active.size:
        elapsed[active] += rng.exponential(size=active.size)
        active = active[elapsed[active] <= end]
        if not active.size:
            break
        aggregate[active] += rng.exponential(size=active.size)
        ruined = aggregate[active] > capital + (1 + theta) * elapsed[active]
        first_ruin[active[ruined]] = elapsed[active[ruined]]
        active = active[~ruined]
    classical = np.searchsorted(
        np.sort(first_ruin), times, side="right"
    ) / count
    standard_error = np.sqrt(classical * (1 - classical) / count)

    # For starting fortune z, the first hit of zero at k has probability
    # z/k * P(Binomial(k, p) = (k-z)/2).
    rounds = np.floor(times).astype(np.int64)
    maximum = int(rounds[-1])
    discrete = np.zeros(periods, dtype=np.float64)
    if capital < maximum:
        z = int(np.floor(capital)) + 1
        k = np.arange(z, maximum + 1, 2, dtype=np.int64)
        q = 1.0 / (2.0 + theta)
        first_hit = z / k * binom.pmf((k + z) // 2, k, q)
        cumulative = np.concatenate(([0.0], np.cumsum(first_hit)))
        discrete = cumulative[np.searchsorted(k, rounds, side="right")]
    return {
        "classical_ruin": classical,
        "classical_standard_error": standard_error,
        "de_finetti_ruin": discrete,
        "absolute_deviations": np.abs(classical - discrete),
    }
