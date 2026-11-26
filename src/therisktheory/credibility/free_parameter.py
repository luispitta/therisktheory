"""Free-parameter credibility, claim-reserving models, and statistical diagnostics.

This module implements empirical Bayes (free-parameter) credibility models following
Bühlmann (1967) and Bühlmann-Straub (1970), credibility-based claim reserving (including
Bornhuetter-Ferguson credibility bridges), and statistical loss reserve diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def free_parameter_credibility(
    loss_matrix: ArrayLike,
    weights: ArrayLike | None = None,
) -> dict[str, Any]:
    r"""Estimate structural parameters and credibility premiums without parametric priors.

    Implements the non-parametric (free-parameter / empirical Bayes) credibility framework
    of Bühlmann (1967) and Bühlmann-Straub (1970). For :math:`I \ge 2` independent risks
    observed over :math:`T \ge 2` periods, loss ratios or claim amounts :math:`X_{it}`
    are observed alongside operational volume or natural weights :math:`P_{it} > 0`.

    The model assumes:

    .. math::
        \mathbb{E}[X_{it} \mid \Theta_i] = m(\Theta_i), \quad
        \text{Var}(X_{it} \mid \Theta_i) = \frac{s^2(\Theta_i)}{P_{it}}

    The structural parameters are estimated without assuming any parametric family for
    either :math:`\Theta_i` or :math:`X_{it} \mid \Theta_i`:

    - **Expected process variance (within-risk variance):**

      .. math::
          \widehat{s}^2 = \frac{1}{\sum_{i=1}^I (T_i - 1)} \sum_{i=1}^I \sum_{t=1}^T
          P_{it} (X_{it} - \bar{X}_i)^2

    - **Variance of hypothetical means (between-risk variance):**

      .. math::
          c^* = P_{\bullet\bullet} - \frac{\sum_{i=1}^I P_{i\bullet}^2}{P_{\bullet\bullet}},
          \quad \widehat{a} = \max\left(0, \frac{1}{c^*} \left[ \sum_{i=1}^I P_{i\bullet}
          (\bar{X}_i - \bar{X}_{\bullet\bullet})^2 - (I - 1) \widehat{s}^2 \right]\right)

    - **Credibility factor for risk :math:`i`:**

      .. math::
          Z_i = \frac{P_{i\bullet}}{P_{i\bullet} + k} = \frac{\widehat{a} P_{i\bullet}}
          {\widehat{a} P_{i\bullet} + \widehat{s}^2} \in [0.0, 1.0], \quad
          \text{where } k = \frac{\widehat{s}^2}{\widehat{a}}

    - **Credibility premium:**

      .. math::
          \widehat{P}_i = Z_i \bar{X}_i + (1 - Z_i) \widehat{\mu}_Z, \quad
          \text{where } \widehat{\mu}_Z = \frac{\sum_{i=1}^I Z_i \bar{X}_i}{\sum_{i=1}^I Z_i}

    Args:
        loss_matrix: 2D array-like of shape :math:`(I, T)` containing historical loss ratios
            or claim amounts for :math:`I \ge 2` risks over :math:`T \ge 2` periods.
        weights: Optional 2D array-like of shape :math:`(I, T)` containing positive exposures
            or natural weights :math:`P_{it} > 0`. If ``None``, defaults to unweighted
            Bühlmann model with :math:`P_{it} = 1.0`.

    Returns:
        dict[str, Any]: A dictionary containing:
            - ``"credibility_premiums"``: NDArray[np.float64] of shape :math:`(I,)`.
            - ``"credibility_factors"``: NDArray[np.float64] of shape :math:`(I,)`.
            - ``"individual_means"``: NDArray[np.float64] of shape :math:`(I,)`.
            - ``"collective_mean"``: float, credibility-weighted grand mean :math:`\widehat{\mu}_Z`.
            - ``"overall_sample_mean"``: float, sample mean :math:`\bar{X}_{\bullet\bullet}`.
            - ``"process_variance"``: float, unbiased estimate :math:`\widehat{s}^2`.
            - ``"variance_hypothetical_means"``: float, estimate :math:`\widehat{a}`.
            - ``"raw_variance_hypothetical_means"``: float, raw :math:`\widehat{a}_{\text{raw}}`.
            - ``"k_ratio"``: float, Bühlmann parameter ratio :math:`k`.
            - ``"total_weights"``: NDArray[np.float64], total exposures :math:`P_{i\bullet}`.

    Raises:
        ValueError: If ``loss_matrix`` is not 2D, has :math:`I < 2` or :math:`T < 2`,
            contains non-finite numbers, or if ``weights`` has mismatched dimensions
            or negative entries.

    Notes:
        Validation status: experimental.

    References:
        Bühlmann, H. (1967). Experience Rating and Credibility. ASTIN Bulletin, 4(3), 199–207.
        Bühlmann, H., & Straub, E. (1970). Glaubwürdigkeit für Schandensätze.
        Bulletin of the Association of Swiss Actuaries, 70, 111–133.
        Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). Modern Actuarial Risk Theory
        (2nd ed.). Springer, Chapter 8.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.credibility import free_parameter_credibility
        >>> # Kaas et al. (2008), Example 8.2.1 data: 5 states, 4 years
        >>> data = np.array([
        ...     [40.0, 50.0, 60.0, 70.0],
        ...     [20.0, 30.0, 40.0, 50.0],
        ...     [60.0, 70.0, 80.0, 90.0],
        ... ])
        >>> res = free_parameter_credibility(data)
        >>> float(res["overall_sample_mean"])
        53.333333333333336
        >>> float(res["process_variance"])
        166.66666666666666
    """
    x = np.asarray(loss_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("loss_matrix must be a 2D array of shape (I, T)")
    num_risks, num_periods = x.shape
    if num_risks < 2:
        raise ValueError(f"loss_matrix must contain at least 2 risks, got {num_risks}")
    if num_periods < 2:
        raise ValueError(f"loss_matrix must contain at least 2 periods, got {num_periods}")
    if not np.all(np.isfinite(x)):
        raise ValueError("loss_matrix contains non-finite values (NaN or Inf)")

    if weights is None:
        p = np.ones_like(x, dtype=np.float64)
    else:
        p = np.asarray(weights, dtype=np.float64)
        if p.shape != x.shape:
            raise ValueError(f"weights shape {p.shape} does not match loss_matrix shape {x.shape}")
        if np.any(p < 0.0) or not np.all(np.isfinite(p)):
            raise ValueError("weights must be non-negative and finite")

    # Row totals: P_{i.}
    row_weights = np.sum(p, axis=1)
    if np.any(row_weights <= 0.0):
        raise ValueError("Each risk must have strictly positive total weight P_{i.} > 0")

    total_exposure = np.sum(row_weights)

    # Individual weighted sample means: \bar{X}_i
    individual_means = np.sum(p * x, axis=1) / row_weights

    # Overall weighted sample mean: \bar{X}_{..}
    overall_mean = float(np.sum(p * x) / total_exposure)

    # 1. Expected process variance: \hat{s}^2
    # Degrees of freedom: sum_{i=1}^I (T_i - 1) where T_i is number of periods with P_{it} > 0
    active_periods_per_risk = np.sum(p > 0.0, axis=1)
    dof_s2 = int(np.sum(np.maximum(0, active_periods_per_risk - 1)))
    if dof_s2 < 1:
        raise ValueError("Insufficient degrees of freedom to estimate process variance")

    # Deviation from individual mean: (X_{it} - \bar{X}_i)
    sq_deviations = (x - individual_means[:, np.newaxis]) ** 2
    s2_hat = float(np.sum(p * sq_deviations) / dof_s2)

    # 2. Variance of hypothetical means: \hat{a}
    # c* = P_{..} - sum(P_{i.}^2) / P_{..}
    sum_sq_row_weights = np.sum(row_weights**2)
    c_star = float(total_exposure - (sum_sq_row_weights / total_exposure))
    if c_star <= 0.0:
        raise ValueError("Degenerate exposure structure: c* <= 0")

    # Between-group sum of squares: sum P_{i.} (\bar{X}_i - \bar{X}_{..})^2
    ssb = float(np.sum(row_weights * ((individual_means - overall_mean) ** 2)))

    # Raw unbiased estimator: (SSB - (I - 1)*s^2) / c*
    dof_between = num_risks - 1
    a_raw = float((ssb - dof_between * s2_hat) / c_star)

    # Truncation at 0 for parameter space admissibility: a = Var(m(Theta)) >= 0
    a_hat = max(0.0, a_raw)

    # 3. Credibility factors: Z_i = a * P_{i.} / (a * P_{i.} + s^2) = P_{i.} / (P_{i.} + s^2/a)
    if a_hat > 0.0:
        k_ratio = s2_hat / a_hat
        z_factors = row_weights / (row_weights + k_ratio)
    else:
        k_ratio = float("inf")
        z_factors = np.zeros(num_risks, dtype=np.float64)

    # 4. Credibility-weighted collective mean: \hat{mu}_Z = sum(Z_i * \bar{X}_i) / sum(Z_i)
    total_z = float(np.sum(z_factors))
    if total_z > 0.0:
        collective_mean = float(np.sum(z_factors * individual_means) / total_z)
    else:
        collective_mean = overall_mean

    # 5. Credibility premiums: \hat{P}_i = Z_i * \bar{X}_i + (1 - Z_i) * \hat{mu}_Z
    cred_premiums = z_factors * individual_means + (1.0 - z_factors) * collective_mean

    return {
        "credibility_premiums": np.asarray(cred_premiums, dtype=np.float64),
        "credibility_factors": np.asarray(z_factors, dtype=np.float64),
        "individual_means": np.asarray(individual_means, dtype=np.float64),
        "collective_mean": collective_mean,
        "overall_sample_mean": overall_mean,
        "process_variance": s2_hat,
        "variance_hypothetical_means": a_hat,
        "raw_variance_hypothetical_means": a_raw,
        "k_ratio": k_ratio,
        "total_weights": np.asarray(row_weights, dtype=np.float64),
    }


def credibility_reserve_estimate(
    reported_claims: float | ArrayLike,
    expected_claims: float | ArrayLike,
    development_factor: float | ArrayLike,
    credibility_factor: float | ArrayLike | None = None,
    *,
    premiums: float | ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    r"""Estimate actuarial claim reserves blending historical development and prior expectations.

    Connects credibility theory to claim reserving and IBNR (Incurred But Not Reported)
    estimation. For an accident year or portfolio cohort, let :math:`C \ge 0` denote cumulative
    reported (or paid) claims to date, :math:`M > 0` denote apriori expected ultimate claims
    (e.g., :math:`M = \text{Premium} \times \text{Expected Loss Ratio}`), and :math:`f \ge 1.0`
    denote the cumulative development factor (:math:`CDF`), with emergence proportion
    :math:`p = 1 / f \in (0, 1]`.

    The credibility ultimate loss estimate balances the empirical Chain-Ladder projection
    :math:`\widehat{U}_{\text{CL}} = C \cdot f` against the prior estimate :math:`M`:

    .. math::
        \widehat{U}_{\text{cred}} = Z \widehat{U}_{\text{CL}} + (1 - Z) M
        = Z (C \cdot f) + (1 - Z) M

    When ``credibility_factor`` is not provided (``None``), the function defaults to the
    canonical **Bornhuetter-Ferguson credibility bridge**, setting :math:`Z = p = 1/f`:

    .. math::
        \widehat{U}_{\text{BF}} = \frac{1}{f} (C \cdot f) + \left(1 - \frac{1}{f}\right) M
        = C + (1 - p) M

    The outstanding claim reserve is given by:

    .. math::
        \widehat{R}_{\text{cred}} = \max\left(0, \widehat{U}_{\text{cred}} - C\right)

    Args:
        reported_claims: Cumulative reported or paid claims :math:`C \ge 0`.
            Scalars and array-like inputs are accepted.
        expected_claims: Apriori expected ultimate claims :math:`M > 0`.
            Scalars and array-like inputs are accepted.
        development_factor: Cumulative development factor :math:`f \ge 1.0`.
            Scalars and array-like inputs are accepted.
        credibility_factor: Credibility weight :math:`Z \in [0.0, 1.0]`. If ``None``,
            defaults to :math:`Z = 1 / f` (Bornhuetter-Ferguson method).
        premiums: Optional earned premium :math:`P > 0` to compute implied loss ratios.

    Returns:
        dict[str, NDArray[np.float64]]: A dictionary containing:
            - ``"ultimate_claims"``: Estimated ultimate claims :math:`\widehat{U}_{\text{cred}}`.
            - ``"reserve"``: Estimated outstanding reserve :math:`\widehat{R}_{\text{cred}}`.
            - ``"chain_ladder_ultimate"``: Empirical development projection :math:`C \cdot f`.
            - ``"prior_expected_ultimate"``: Prior baseline :math:`M`.
            - ``"credibility_factors"``: Applied credibility weight :math:`Z`.
            - ``"emergence_percentage"``: Proportion emerged :math:`p = 1 / f`.
            - ``"implied_loss_ratio"``: Ultimate claims divided by premium (if provided).

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), or if :math:`C < 0`,
            :math:`M \le 0`, :math:`f < 1.0`, or :math:`Z \notin [0, 1]`.

    Notes:
        Validation status: experimental.

    References:
        Bornhuetter, R. L., & Ferguson, R. E. (1972). The Actuary and IBNR.
        Proceedings of the Casualty Actuarial Society, 59(112), 181–195.
        Mack, T. (2000). Credibility Theory for Loss Reserving. ASTIN Bulletin, 30(2), 333–347.
        Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). Loss Models (5th ed.). Wiley.

    Examples:
        >>> from therisktheory.credibility import credibility_reserve_estimate
        >>> res = credibility_reserve_estimate(
        ...     reported_claims=60000.0,
        ...     expected_claims=100000.0,
        ...     development_factor=1.25,
        ... )
        >>> float(res["ultimate_claims"])
        80000.0
        >>> float(res["reserve"])
        20000.0
    """
    c = np.asarray(reported_claims, dtype=np.float64)
    m = np.asarray(expected_claims, dtype=np.float64)
    f = np.asarray(development_factor, dtype=np.float64)

    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(m)) and np.all(np.isfinite(f))):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(c < 0.0):
        raise ValueError("reported_claims must be non-negative")
    if np.any(m <= 0.0):
        raise ValueError("expected_claims must be strictly positive")
    if np.any(f < 1.0):
        raise ValueError("development_factor must be greater than or equal to 1.0")

    emergence = 1.0 / f

    if credibility_factor is None:
        z = emergence
    else:
        z = np.asarray(credibility_factor, dtype=np.float64)
        if not np.all(np.isfinite(z)):
            raise ValueError("credibility_factor must contain only finite numbers")
        if np.any((z < 0.0) | (z > 1.0)):
            raise ValueError("credibility_factor must be between 0.0 and 1.0 inclusive")

    cl_ultimate = c * f
    cred_ultimate = z * cl_ultimate + (1.0 - z) * m
    reserve = np.maximum(0.0, cred_ultimate - c)

    result: dict[str, NDArray[np.float64]] = {
        "ultimate_claims": np.asarray(cred_ultimate, dtype=np.float64),
        "reserve": np.asarray(reserve, dtype=np.float64),
        "chain_ladder_ultimate": np.asarray(cl_ultimate, dtype=np.float64),
        "prior_expected_ultimate": np.asarray(m, dtype=np.float64),
        "credibility_factors": np.asarray(z, dtype=np.float64),
        "emergence_percentage": np.asarray(emergence, dtype=np.float64),
    }

    if premiums is not None:
        prem = np.asarray(premiums, dtype=np.float64)
        if not np.all(np.isfinite(prem)):
            raise ValueError("premiums must contain only finite numbers")
        if np.any(prem <= 0.0):
            raise ValueError("premiums must be strictly positive")
        result["implied_loss_ratio"] = np.asarray(cred_ultimate / prem, dtype=np.float64)

    return result


def reserve_model_diagnostics(
    observed_values: ArrayLike,
    credibility_estimates: ArrayLike,
    collective_mean: float | ArrayLike,
    credibility_factors: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    process_variance: float | None = None,
    hypothetical_mean_variance: float | None = None,
) -> dict[str, Any]:
    r"""Compute statistical diagnostics, shrinkage metrics, and prediction errors for credibility.

    Evaluates portfolio balance, mean squared prediction error reduction, shrinkage
    properties, and residual dispersion for credibility-weighted estimates:

    1. **Portfolio Imbalance:**
       Measures replication of total observed loss volume:

       .. math::
           \text{Imbalance} = \sum_{i=1}^I P_i (\widehat{Y}_i - X_i)

       Under an optimal credibility weighting scheme, :math:`\text{Imbalance} \approx 0.0`.

    2. **Theoretical Mean Squared Prediction Error (MSE):**
       In the Bühlmann model, the mean squared prediction error is:

       .. math::
           \text{MSE}(Z_i) = \mathbb{E}[(\widehat{\mu}_i - m(\Theta_i))^2] = (1 - Z_i) a

       Yielding a relative variance reduction of :math:`Z_i` over the collective baseline.

    3. **Empirical Fit Metrics:**
       Weighted Mean Absolute Error (:math:`\text{MAE}`) and Mean Squared Error (:math:`\text{MSE}`)
       against observed portfolio experience.

    Args:
        observed_values: 1D array-like of observed experience :math:`X_i`.
        credibility_estimates: 1D array-like of fitted credibility values :math:`\widehat{Y}_i`.
        collective_mean: Overall collective benchmark :math:`\mu`.
        credibility_factors: 1D array-like of credibility weights :math:`Z_i \in [0, 1]`.
        weights: Optional operational weights :math:`P_i > 0`. If ``None``, defaults to 1.0.
        process_variance: Optional estimated within-risk variance :math:`s^2`.
        hypothetical_mean_variance: Optional between-risk variance :math:`a`.

    Returns:
        dict[str, Any]: Statistical diagnostic metrics, including:
            - ``"portfolio_imbalance"``: float, total exposure-weighted deviation.
            - ``"relative_imbalance"``: float, imbalance divided by total observed volume.
            - ``"weighted_mae"``: float, weighted mean absolute deviation.
            - ``"weighted_rmse"``: float, weighted root mean squared error.
            - ``"mean_credibility"``: float, average credibility weight.
            - ``"high_credibility_ratio"``: float, fraction of risks with :math:`Z_i \ge 0.5`.
            - ``"theoretical_mse"``: NDArray[np.float64], :math:`(1 - Z_i) a` (if :math:`a` given).
            - ``"variance_reduction_percentage"``: NDArray[np.float64], :math:`Z_i \times 100\%`.
            - ``"residuals"``: NDArray[np.float64], :math:`X_i - \widehat{Y}_i`.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), or if input arrays
            have mismatched dimensions or invalid bounds.

    Notes:
        Validation status: experimental.

    References:
        Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). Loss Models (5th ed.). Wiley.
        Bühlmann, H. (1967). Experience Rating and Credibility. ASTIN Bulletin.
    """
    y_obs = np.asarray(observed_values, dtype=np.float64)
    y_fit = np.asarray(credibility_estimates, dtype=np.float64)
    z = np.asarray(credibility_factors, dtype=np.float64)

    if not (np.all(np.isfinite(y_obs)) and np.all(np.isfinite(y_fit)) and np.all(np.isfinite(z))):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    coll_m = np.asarray(collective_mean, dtype=np.float64)
    if not np.all(np.isfinite(coll_m)):
        raise ValueError("collective_mean must contain only finite numbers")
    m = float(coll_m.item())

    if y_obs.ndim != 1 or y_fit.ndim != 1 or z.ndim != 1:
        raise ValueError("observed_values, credibility_estimates, and factors must be 1D arrays")
    if not (len(y_obs) == len(y_fit) == len(z)):
        raise ValueError("Input arrays must have identical length")
    if np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("credibility_factors must be between 0.0 and 1.0 inclusive")

    n = len(y_obs)
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must contain only finite numbers")
        if w.shape != y_obs.shape or np.any(w < 0.0):
            raise ValueError("weights must be non-negative with matching shape")

    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        raise ValueError("total weight must be strictly positive")

    residuals = y_obs - y_fit
    imbalance = float(np.sum(w * (y_fit - y_obs)))
    total_obs_vol = float(np.sum(w * np.abs(y_obs)))
    rel_imbalance = imbalance / total_obs_vol if total_obs_vol > 0.0 else 0.0

    weighted_mae = float(np.sum(w * np.abs(residuals)) / total_weight)
    weighted_rmse = float(np.sqrt(np.sum(w * (residuals**2)) / total_weight))

    mean_z = float(np.mean(z))
    high_cred_ratio = float(np.mean(z >= 0.5))

    diag: dict[str, Any] = {
        "collective_mean": m,
        "portfolio_imbalance": imbalance,
        "relative_imbalance": rel_imbalance,
        "weighted_mae": weighted_mae,
        "weighted_rmse": weighted_rmse,
        "mean_credibility": mean_z,
        "high_credibility_ratio": high_cred_ratio,
        "residuals": residuals,
        "variance_reduction_percentage": z * 100.0,
    }

    if hypothetical_mean_variance is not None:
        if not np.isfinite(hypothetical_mean_variance):
            raise ValueError("hypothetical_mean_variance must be finite")
        a = float(hypothetical_mean_variance)
        if a < 0.0:
            raise ValueError("hypothetical_mean_variance must be non-negative")
        diag["theoretical_mse"] = (1.0 - z) * a

    if process_variance is not None:
        if not np.isfinite(process_variance):
            raise ValueError("process_variance must be finite")
        s2 = float(process_variance)
        if s2 < 0.0:
            raise ValueError("process_variance must be non-negative")
        # Standardized residuals
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.sqrt(s2 / np.where(w > 0.0, w, 1.0))
            diag["standardized_residuals"] = np.where(denom > 0.0, residuals / denom, 0.0)

    return diag
