"""Fair and unfair games and the gambler's ruin problem.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). Modern
    Actuarial Risk Theory: Using R, 2nd ed. Springer. Section 4.5.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt

_PROBABILITY_SUM_TOLERANCE = 1e-9
_SIMULATION_CHUNK = 256


def fair_game_expectation(
    probabilities: npt.ArrayLike,
    payoffs: npt.ArrayLike,
    *,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Computes fair stake of a game E[Y] = sum_i p_i y_i."""
    probs, values = _outcome_arrays(probabilities, payoffs, axis)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.sum(probs * values, axis=axis))
    return _validation.check_finite_result(result, "fair_game_expectation")


def unfair_game_expectation(
    probabilities: npt.ArrayLike,
    payoffs: npt.ArrayLike,
    stake: npt.ArrayLike,
    *,
    rounds: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Computes expected net gain n * (E[Y] - c)."""
    n = _as_count(rounds, "rounds", minimum=0)
    price = _validation.as_real_array(stake, "stake")
    mean = fair_game_expectation(probabilities, payoffs, axis=axis)
    _validation.check_broadcastable(expectation=mean, stake=price)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(n * (mean - price))
    return _validation.check_finite_result(result, "unfair_game_expectation")


def gambler_ruin_probability(
    initial_capital: npt.ArrayLike,
    target_capital: npt.ArrayLike,
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes probability that gambler is ruined."""
    z = _validation.as_nonnegative_finite_array(
        initial_capital, "initial_capital"
    )
    a = _validation.as_real_array(
        target_capital, "target_capital", allow_inf=True
    )
    p = _validation.as_probability_array(win_probability, "win_probability")
    _validation.check_broadcastable(
        initial_capital=z, target_capital=a, win_probability=p
    )
    _check_integer(z, "initial_capital")
    _check_integer(a[np.isfinite(a)], "target_capital")
    if np.any(z > a):
        raise ValueError("initial_capital must not exceed target_capital")

    z, a, p = np.broadcast_arrays(z, a, p)
    finite = np.isfinite(a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_r = np.log1p(-p) - np.log(p)
        a_fin = np.where(finite, a, 0.0)
        fair = (a_fin - z) / np.where(a_fin > 0, a_fin, 1.0)
        q_wins = np.expm1(-(a_fin - z) * log_r) / np.expm1(-a_fin * log_r)
        p_wins = (
            np.exp(z * log_r)
            * -np.expm1((a_fin - z) * log_r)
            / -np.expm1(a_fin * log_r)
        )
        no_target = np.where(log_r < 0, np.exp(z * log_r), 1.0)

    result = np.select(
        [
            z == 0,
            finite & (z == a),
            p == 0,
            p == 1,
            ~finite,
            log_r == 0,
            log_r > 0,
        ],
        [1.0, 0.0, 1.0, 0.0, no_target, fair, q_wins],
        default=p_wins,
    )
    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class GamblerRuinSimulation:
    """Result of simulate_gambler_ruin."""

    ruin_probability: float
    standard_error: float
    mean_duration: float
    n_paths: int
    unfinished_paths: int


def simulate_gambler_ruin(
    initial_capital: int,
    target_capital: float,
    win_probability: float,
    *,
    n_paths: int = 10_000,
    max_rounds: int = 100_000,
    rng: np.random.Generator | int | None = None,
) -> GamblerRuinSimulation:
    """Estimates gambler ruin probability via Monte Carlo."""
    n = _as_count(n_paths, "n_paths", minimum=1)
    max_n = _as_count(max_rounds, "max_rounds", minimum=1)
    gambler_ruin_probability(initial_capital, target_capital, win_probability)
    z = float(initial_capital)
    a = float(target_capital)
    p = float(win_probability)
    generator = np.random.default_rng(rng)
    fortune = np.full(n, z)
    ruined = np.zeros(n, dtype=bool)
    duration = np.zeros(n, dtype=np.int64)
    if z <= 0:
        ruined[:] = True
        active = np.empty(0, dtype=np.int64)
    elif z >= a:
        active = np.empty(0, dtype=np.int64)
    else:
        active = np.arange(n)
    played = 0
    while active.size and played < max_n:
        block = min(_SIMULATION_CHUNK, max_n - played)
        steps = np.where(generator.random((active.size, block)) < p, 1, -1)
        paths = fortune[active, None] + np.cumsum(steps, axis=1)
        down = paths <= 0
        stop = down | (paths >= a)
        stopped = stop.any(axis=1)
        first = stop.argmax(axis=1)
        done = active[stopped]
        ruined[done] = down[stopped, first[stopped]]
        duration[done] = played + first[stopped] + 1
        fortune[active] = paths[:, -1]
        active = active[~stopped]
        played += block
    finished = np.ones(n, dtype=bool)
    finished[active] = False
    estimate = float(ruined.mean())
    return GamblerRuinSimulation(
        ruin_probability=estimate,
        standard_error=float(np.sqrt(estimate * (1.0 - estimate) / n)),
        mean_duration=(
            float(duration[finished].mean()) if finished.any() else np.nan
        ),
        n_paths=n,
        unfinished_paths=int(active.size),
    )


def _outcome_arrays(
    probabilities: npt.ArrayLike, payoffs: npt.ArrayLike, axis: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    probs = _validation.as_probability_array(probabilities, "probabilities")
    values = _validation.as_real_array(payoffs, "payoffs")
    _validation.check_broadcastable(probabilities=probs, payoffs=values)
    probs, values = np.broadcast_arrays(probs, values)
    if probs.ndim == 0:
        raise ValueError("probabilities must list at least one outcome")
    totals = np.sum(probs, axis=axis)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=_PROBABILITY_SUM_TOLERANCE):
        raise ValueError("probabilities must add up to 1")
    return probs, values


def _as_count(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def _check_integer(values: npt.NDArray[np.float64], name: str) -> None:
    if np.any(values != np.floor(values)):
        raise ValueError(f"{name} must be a whole number of stakes")
