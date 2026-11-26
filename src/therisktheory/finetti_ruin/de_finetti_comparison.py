"""Comparative analysis between classical risk theory and De Finetti's framework.

Reference:
    De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
    Beard, R. E., Pentikäinen, T., and Pesonen, E. (1984). Risk Theory.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt

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
) -> dict[str, npt.NDArray[np.float64]]:
    """Performs a reproducible benchmark deriving classical (Cramér-Lundberg asymptotic)
    and De Finetti discrete game ruin probabilities under identical risk metrics and horizons.

    Args:
        initial_capital: Starting surplus U_0 (fixed across horizon).
        lambda_rate: Poisson claim arrival intensity.
        claim_mean: Expected single claim severity.
        safety_loading: Relative safety loading coefficient (theta > 0).
        horizon_periods: Discrete time horizon steps T for evaluation (t = 1, ..., T).

    Returns:
        Dictionary containing classical estimates, De Finetti discrete estimates, and absolute deviations.
    """
    if initial_capital < 0 or lambda_rate <= 0 or claim_mean <= 0 or safety_loading <= 0 or horizon_periods < 1:
        raise ValueError("Invalid parameters for benchmark generation.")

    # Premium income rate c = (1 + theta) * lambda * expected_claim
    premium_rate = (1.0 + safety_loading) * lambda_rate * claim_mean

    # Classical Cramér-Lundberg adjustment coefficient approximation (Lundberg's exponent R)
    r_adj = (2.0 * safety_loading) / (claim_mean * (1.0 + safety_loading))

    # Time horizon steps array (t = 1 to horizon_periods)
    time_steps = np.arange(1, horizon_periods + 1, dtype=np.float64)

    # Classical ruin probability estimate over horizon t
    classical_est = np.array([
        min(1.0, np.exp(-r_adj * initial_capital) * (1.0 + 0.1 / t))
        for t in time_steps
    ], dtype=np.float64)

    # De Finetti discrete game ruin probability estimate using explicit p_equivalent
    p_equivalent = 1.0 / (1.0 + safety_loading)
    
    de_finetti_est = np.array([
        min(1.0, (p_equivalent ** (initial_capital / claim_mean)) * (1.0 + (t / (horizon_periods + 1))))
        for t in time_steps
    ], dtype=np.float64)
    de_finetti_est = np.clip(de_finetti_est, 0.0, 1.0)

    deviations = compare_classical_de_finetti(classical_est, de_finetti_est)

    return {
        "classical_ruin": classical_est,
        "de_finetti_ruin": de_finetti_est,
        "absolute_deviations": deviations,
    }
