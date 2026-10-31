"""Aggregate loss models, moments, and generating functions.

Topic 3: Frequency, Severity, Aggregate Loss, Approximations, and Simulation.
Author: Maria Lucia Monticelli D’Elias
Reference: Panjer, H.; Willmot, G. — Insurance Risk Models; Hogg, R.; Klugman, S. — Loss Distributions.
"""

import numpy as np


def aggregate_loss_moments(
    expected_frequency: float | np.ndarray,
    variance_frequency: float | np.ndarray,
    expected_severity: float | np.ndarray,
    variance_severity: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the mean, variance, and standard deviation of aggregate losses.

    Theoretical Context:
        Under the collective risk model, the aggregate loss S is given by the sum 
        of a random number of claims N, $S = \sum_{i=1}^{N} X_i$.
        Using conditional expectation and variance (Wald's identities / Bienaymé formula):
            E[S] = E[N] * E[X]
            Var(S) = E[N] * Var(X) + Var(N) * (E[X])^2
            SD(S) = sqrt(Var(S))

    Args:
        expected_frequency: Expected number of claims E[N]. Must be non-negative.
        variance_frequency: Variance of claim count Var(N). Must be non-negative.
        expected_severity: Expected claim severity E[X]. Must be non-negative.
        variance_severity: Variance of claim severity Var(X). Must be non-negative.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing (mean_s, variance_s, std_s).
    """
    e_n = np.asarray(expected_frequency, dtype=np.float64)
    v_n = np.asarray(variance_frequency, dtype=np.float64)
    e_x = np.asarray(expected_severity, dtype=np.float64)
    v_x = np.asarray(variance_severity, dtype=np.float64)

    if np.any(e_n < 0.0) or np.any(v_n < 0.0) or np.any(e_x < 0.0) or np.any(v_x < 0.0):
        raise ValueError("All frequency and severity parameters must be non-negative.")

    mean_s = e_n * e_x
    variance_s = e_n * v_x + v_n * (e_x ** 2)
    std_s = np.sqrt(variance_s)

    return np.asarray(mean_s, dtype=np.float64), np.asarray(variance_s, dtype=np.float64), np.asarray(std_s, dtype=np.float64)


def aggregate_mgf(
    t: float | np.ndarray,
    lambda_poisson: float | np.ndarray,
    severity_mgf_values: float | np.ndarray,
) -> np.ndarray:
    """Calculate the Moment Generating Function (MGF) of the compound Poisson aggregate loss.

    Theoretical Context:
        For a compound Poisson process with parameter lambda and claim severity MGF M_X(t),
        the MGF of the aggregate loss S is given by:
            M_S(t) = exp( lambda * (M_X(t) - 1) )

    Args:
        t: Evaluation point or array for the MGF parameter t.
        lambda_poisson: Poisson process intensity parameter (lambda). Must be non-negative.
        severity_mgf_values: Pre-calculated values of the severity MGF M_X(t).

    Returns:
        np.ndarray: Evaluated MGF values of the aggregate loss model.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    lam = np.asarray(lambda_poisson, dtype=np.float64)
    mgf_x = np.asarray(severity_mgf_values, dtype=np.float64)

    if np.any(lam < 0.0):
        raise ValueError("Poisson intensity lambda must be non-negative.")

    mgf_s = np.exp(lam * (mgf_x - 1.0))
    return np.asarray(mgf_s, dtype=np.float64)


def aggregate_characteristic_function(
    t: float | np.ndarray,
    lambda_poisson: float | np.ndarray,
    severity_cf_values: np.ndarray,
) -> np.ndarray:
    """Calculate the Characteristic Function of the compound Poisson aggregate loss.

    Theoretical Context:
        The characteristic function phi_S(t) for a compound Poisson model is:
            phi_S(t) = exp( lambda * (phi_X(t) - 1) )
        where phi_X(t) is the characteristic function of the individual claim severity.

    Args:
        t: Evaluation point or array for parameter t.
        lambda_poisson: Poisson frequency parameter (lambda). Must be non-negative.
        severity_cf_values: Complex or real array of severity characteristic function values.

    Returns:
        np.ndarray: Characteristic function values of aggregate losses.
    """
    lam = np.asarray(lambda_poisson, dtype=np.float64)
    cf_x = np.asarray(severity_cf_values)

    if np.any(lam < 0.0):
        raise ValueError("Poisson intensity lambda must be non-negative.")

    cf_s = np.exp(lam * (cf_x - 1.0))
    return np.asarray(cf_s)


def individual_model_loss(
    probabilities: np.ndarray,
    amounts: np.ndarray,
) -> tuple[float, float]:
    """Calculate the exact mean and variance of losses under the individual risk model.

    Theoretical Context:
        In the individual risk model, there is a fixed number of independent risks,
        where each risk i has a probability of claim q_i and a fixed amount b_i.
            E[S] = sum(q_i * b_i)
            Var(S) = sum(q_i * (1 - q_i) * b_i^2)

    Args:
        probabilities: Array of claim occurrence probabilities (q_i) in [0, 1].
        amounts: Array of corresponding fixed claim amounts (b_i). Must be non-negative.

    Returns:
        tuple[float, float]: Total expected aggregate loss and its variance.
    """
    q = np.asarray(probabilities, dtype=np.float64)
    b = np.asarray(amounts, dtype=np.float64)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Probabilities must be between 0 and 1.")
    if np.any(b < 0.0):
        raise ValueError("Claim amounts must be non-negative.")

    mean_ind = np.sum(q * b)
    var_ind = np.sum(q * (1.0 - q) * (b ** 2))

    return float(mean_ind), float(var_ind)


def collective_model_loss(
    expected_count: float,
    severity_mean: float,
    severity_variance: float,
    count_variance: float | None = None,
) -> dict[str, float]:
    """Compute summary statistics for a collective risk model.

    Args:
        expected_count: Expected claim count E[N].
        severity_mean: Expected severity E[X].
        severity_variance: Severity variance Var(X).
        count_variance: Variance of claim count Var(N). Defaults to E[N] (Poisson assumption).

    Returns:
        dict[str, float]: Dictionary containing mean, variance, and standard deviation.
    """
    en = float(expected_count)
    vn = float(count_variance) if count_variance is not None else en
    ex = float(severity_mean)
    vx = float(severity_variance)

    mean_s, var_s, std_s = aggregate_loss_moments(en, vn, ex, vx)

    return {
        "mean": float(mean_s),
        "variance": float(var_s),
        "std": float(std_s),
    }