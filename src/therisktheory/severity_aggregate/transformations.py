"""Gamma approximations and actuarial transformations (Wilson-Hilferty, Haldane, Esscher).

Reference:
    Panjer, H. H. and Willmot, G. E. (1992). Insurance Risk Models. Society of Actuaries.
    Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer-Verlag.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from therisktheory import _validation

def gamma_approximation(
    mean: float | int | ArrayLike,
    variance: float | int | ArrayLike,
    quantiles: ArrayLike | None = None,
) -> dict[str, Any]:
    """Approximates an aggregate loss distribution using a moment-matched Gamma distribution."""
    mean_arr = _validation.validate_numeric_array(mean)
    var_arr = _validation.validate_numeric_array(variance)
    if np.any(mean_arr <= 0) or np.any(var_arr <= 0):
        raise ValueError("Mean and variance must be strictly positive.")
    alpha = (mean_arr ** 2) / var_arr
    scale = var_arr / mean_arr
    result = {
        "shape_alpha": alpha,
        "scale_theta": scale,
        "mean": mean_arr,
        "variance": var_arr,
    }
    if quantiles is not None:
        q_arr = _validation.validate_numeric_array(quantiles)
        if np.any(q_arr < 0.0) or np.any(q_arr > 1.0):
            raise ValueError("Quantile probabilities must be between 0.0 and 1.0.")
        from scipy.stats import gamma
        approx_quantiles = gamma.ppf(q_arr, a=alpha, scale=scale)
        result["quantiles"] = approx_quantiles
    return result

def wilson_hilferty_transform(
    values: ArrayLike,
    degrees_freedom: float | int,
) -> NDArray[np.float64]:
    """Applies the Wilson-Hilferty cube-root transformation to approximate a Chi-squared or Gamma variable as normal."""
    arr = _validation.validate_numeric_array(values)
    if not isinstance(degrees_freedom, (int, float, np.number)) or degrees_freedom <= 0:
        raise ValueError("Degrees of freedom must be a strictly positive number.")
    if np.any(arr < 0):
        raise ValueError("Input values for Wilson-Hilferty transform must be non-negative.")
    nu = float(degrees_freedom)
    term = (arr / nu) ** (1.0 / 3.0)
    mean_adjustment = 1.0 - (2.0 / (9.0 * nu))
    std_adjustment = np.sqrt(2.0 / (9.0 * nu))
    return np.asarray((term - mean_adjustment) / std_adjustment, dtype=np.float64)

def haldane_transform(
    proportions: ArrayLike,
    total_count: float | int,
) -> NDArray[np.float64]:
    """Applies Haldane's variance-stabilizing transformation for proportions or rates."""
    arr = _validation.validate_numeric_array(proportions)
    if not isinstance(total_count, (int, float, np.number)) or total_count <= 0:
        raise ValueError("Total count or exposure base must be a strictly positive number.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("Proportions must be bounded between 0.0 and 1.0.")
    n = float(total_count)
    adjusted = (arr * n + 0.5) / (n + 1.0)
    return np.asarray(np.log(adjusted / (1.0 - adjusted)), dtype=np.float64)

def esscher_transform(
    losses: ArrayLike,
    probabilities: ArrayLike,
    esscher_parameter: float | int,
) -> NDArray[np.float64]:
    """Computes the Esscher transform (exponential tilt) of a discrete loss distribution."""
    loss_arr = _validation.validate_numeric_array(losses)
    prob_arr = _validation.validate_numeric_array(probabilities)
    if loss_arr.shape != prob_arr.shape:
        raise ValueError("Losses and probabilities arrays must have the same shape.")
    if np.any(prob_arr < 0.0) or np.any(prob_arr > 1.0):
        raise ValueError("Probabilities must be between 0.0 and 1.0.")
    if not np.isclose(np.sum(prob_arr), 1.0, atol=1e-5):
        raise ValueError("Probability mass function must sum to 1.0.")
    if not isinstance(esscher_parameter, (int, float, np.number)):
        raise TypeError("Esscher parameter must be a numeric value.")
    h = float(esscher_parameter)
    numerator = np.exp(h * loss_arr) * prob_arr
    mgf_h = np.sum(numerator)
    if mgf_h <= 0 or not np.isfinite(mgf_h):
        raise ValueError("Esscher transform generated an invalid moment-generating function value.")
    return np.asarray(numerator / mgf_h, dtype=np.float64)
