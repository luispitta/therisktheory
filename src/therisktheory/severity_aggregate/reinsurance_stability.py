"""Variance as a stability measure, optimal reinsurance, and reciprocity between insurers.

This module provides actuarial tools for quantifying portfolio stability under
reinsurance, determining optimal retention limits, and evaluating bilateral reciprocity
exchanges between direct insurers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def technical_result_variance(
    retention_rate: float | ArrayLike,
    expected_frequency: float | ArrayLike,
    variance_frequency: float | ArrayLike,
    expected_severity: float | ArrayLike,
    variance_severity: float | ArrayLike,
) -> NDArray[np.float64]:
    r"""Compute the variance of the technical result under proportional reinsurance.

    For an aggregate claim model :math:`S = \sum_{i=1}^N X_i` with claim frequency
    :math:`N` and severity :math:`X`, the law of total variance yields the gross loss
    variance:

    .. math::
        \text{Var}(S) = \mathbb{E}[N]\text{Var}(X) + \text{Var}(N)(\mathbb{E}[X])^2

    Under proportional reinsurance with retention rate :math:`\alpha \in [0.0, 1.0]`,
    the retained claims are :math:`\alpha S`, and the premium is deterministic.
    Hence, the variance of the technical underwriting result :math:`R = P_{\text{ret}} - \alpha S`
    equals the retained loss variance:

    .. math::
        \text{Var}(R) = \alpha^2 \text{Var}(S)
                      = \alpha^2 \left( \mathbb{E}[N]\text{Var}(X)
                        + \text{Var}(N)(\mathbb{E}[X])^2 \right)

    Args:
        retention_rate: Retained proportion :math:`\alpha \in [0.0, 1.0]`.
            Scalars and array-like inputs are accepted.
        expected_frequency: Expected claim count :math:`\mathbb{E}[N] \ge 0`.
        variance_frequency: Variance of claim count :math:`\text{Var}(N) \ge 0`.
        expected_severity: Expected individual loss :math:`\mathbb{E}[X] \ge 0`.
        variance_severity: Variance of individual loss :math:`\text{Var}(X) \ge 0`.

    Returns:
        NDArray[np.float64]: Retained technical result variance. NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), if ``retention_rate``
            is outside :math:`[0.0, 1.0]`, or if moments are negative.

    Notes:
        Validation status: experimental.

    References:
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.).
        Society of Actuaries, Chapter 14.
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory (3rd ed.).
        Chapter 6.

    Examples:
        >>> from therisktheory.severity_aggregate import technical_result_variance
        >>> var_ret = technical_result_variance(
        ...     retention_rate=0.8,
        ...     expected_frequency=10.0,
        ...     variance_frequency=10.0,
        ...     expected_severity=100.0,
        ...     variance_severity=400.0,
        ... )
        >>> float(var_ret)
        66560.0
    """
    alpha = np.asarray(retention_rate, dtype=np.float64)
    e_n = np.asarray(expected_frequency, dtype=np.float64)
    v_n = np.asarray(variance_frequency, dtype=np.float64)
    e_x = np.asarray(expected_severity, dtype=np.float64)
    v_x = np.asarray(variance_severity, dtype=np.float64)

    if not (
        np.all(np.isfinite(alpha))
        and np.all(np.isfinite(e_n))
        and np.all(np.isfinite(v_n))
        and np.all(np.isfinite(e_x))
        and np.all(np.isfinite(v_x))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("retention_rate must be between 0.0 and 1.0 inclusive")
    if np.any(e_n < 0.0) or np.any(v_n < 0.0):
        raise ValueError("Frequency moments must be non-negative")
    if np.any(e_x < 0.0) or np.any(v_x < 0.0):
        raise ValueError("Severity moments must be non-negative")

    gross_var = e_n * v_x + v_n * (e_x**2)
    return np.asarray((alpha**2) * gross_var, dtype=np.float64)


def optimal_reinsurance(
    expected_loss: float | ArrayLike,
    variance_loss: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    target_variance: float | ArrayLike | None = None,
    risk_aversion: float | ArrayLike | None = None,
    premium_budget: float | ArrayLike | None = None,
    criterion: Literal["target_variance", "mean_variance", "premium_budget"] = "target_variance",
) -> NDArray[np.float64]:
    r"""Determine optimal proportional reinsurance retention rate under stability constraints.

    Provides three actuarial optimization criteria for proportional retention
    :math:`\alpha^* \in [0.0, 1.0]`:

    1. ``"target_variance"``: Insurer specifies a maximum tolerable aggregate
       loss variance :math:`V_{\text{target}} > 0`:

       .. math::
           \alpha^* = \min\left(1.0, \sqrt{\frac{V_{\text{target}}}{\text{Var}(S)}}\right)

    2. ``"mean_variance"``: Trade-off between retained loss variance and reinsurance
       premium cost :math:`(1 - \alpha)(1 + \xi)\mathbb{E}[S]` with weight :math:`\lambda > 0`:

       .. math::
           \alpha^* = \text{clip}\left(
               \frac{\lambda (1 + \xi)\mathbb{E}[S]}{2 \text{Var}(S)}, 0.0, 1.0
           \right)

    3. ``"premium_budget"``: Insurer has maximum budget :math:`B \ge 0` allocated to
       purchase reinsurance:

       .. math::
           \alpha^* = \max\left(0.0, 1.0 - \frac{B}{(1 + \xi)\mathbb{E}[S]}\right)

    Args:
        expected_loss: Expected aggregate claims :math:`\mathbb{E}[S] > 0`.
        variance_loss: Variance of aggregate claims :math:`\text{Var}(S) > 0`.
        reinsurer_loading: Reinsurer relative safety loading :math:`\xi \ge 0`.
        target_variance: Maximum allowed variance :math:`V_{\text{target}} > 0`.
            Required if ``criterion="target_variance"``.
        risk_aversion: Trade-off weight :math:`\lambda > 0`.
            Required if ``criterion="mean_variance"``.
        premium_budget: Maximum reinsurance premium :math:`B \ge 0`.
            Required if ``criterion="premium_budget"``.
        criterion: Optimization criterion. Defaults to ``"target_variance"``.

    Returns:
        NDArray[np.float64]: Optimal retention proportion :math:`\alpha^* \in [0.0, 1.0]`.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), violate domain
            bounds, or if required parameters for the chosen criterion are missing.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Section 1.3.
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory (3rd ed.). Chapter 6.

    Examples:
        >>> from therisktheory.severity_aggregate import optimal_reinsurance
        >>> opt = optimal_reinsurance(
        ...     expected_loss=1000.0,
        ...     variance_loss=100000.0,
        ...     reinsurer_loading=0.20,
        ...     target_variance=40000.0,
        ...     criterion="target_variance",
        ... )
        >>> np.isclose(float(opt), np.sqrt(0.4))
        True
    """
    e_s = np.asarray(expected_loss, dtype=np.float64)
    v_s = np.asarray(variance_loss, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)

    if not (np.all(np.isfinite(e_s)) and np.all(np.isfinite(v_s)) and np.all(np.isfinite(xi))):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(e_s <= 0.0):
        raise ValueError("expected_loss must be strictly positive")
    if np.any(v_s <= 0.0):
        raise ValueError("variance_loss must be strictly positive")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")

    if criterion == "target_variance":
        if target_variance is None:
            raise ValueError("target_variance must be provided when criterion='target_variance'")
        v_tgt = np.asarray(target_variance, dtype=np.float64)
        if not np.all(np.isfinite(v_tgt)):
            raise ValueError("target_variance must contain only finite numbers")
        if np.any(v_tgt <= 0.0):
            raise ValueError("target_variance must be strictly positive")
        alpha_star = np.minimum(1.0, np.sqrt(v_tgt / v_s))
        return np.asarray(np.clip(alpha_star, 0.0, 1.0), dtype=np.float64)

    if criterion == "mean_variance":
        if risk_aversion is None:
            raise ValueError("risk_aversion must be provided when criterion='mean_variance'")
        lam = np.asarray(risk_aversion, dtype=np.float64)
        if not np.all(np.isfinite(lam)):
            raise ValueError("risk_aversion must contain only finite numbers")
        if np.any(lam <= 0.0):
            raise ValueError("risk_aversion must be strictly positive")
        alpha_star = (lam * (1.0 + xi) * e_s) / (2.0 * v_s)
        return np.asarray(np.clip(alpha_star, 0.0, 1.0), dtype=np.float64)

    if criterion == "premium_budget":
        if premium_budget is None:
            raise ValueError("premium_budget must be provided when criterion='premium_budget'")
        b = np.asarray(premium_budget, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            raise ValueError("premium_budget must contain only finite numbers")
        if np.any(b < 0.0):
            raise ValueError("premium_budget must be non-negative")
        re_cost_full = (1.0 + xi) * e_s
        alpha_star = np.maximum(0.0, 1.0 - (b / re_cost_full))
        return np.asarray(np.clip(alpha_star, 0.0, 1.0), dtype=np.float64)

    raise ValueError(
        f"criterion must be 'target_variance', 'mean_variance', or 'premium_budget', "
        f"got {criterion!r}"
    )


def insurer_reciprocity_metric(
    variance_1: float | ArrayLike,
    variance_2: float | ArrayLike,
    expected_loss_1: float | ArrayLike | None = None,
    expected_loss_2: float | ArrayLike | None = None,
    quota_1: float | ArrayLike = 0.5,
    correlation: float | ArrayLike = 0.0,
) -> dict[str, NDArray[np.float64]]:
    r"""Evaluate diversification efficiency in a bilateral reciprocity exchange.

    In an actuarial reciprocity treaty, two direct insurers exchange quotas of their
    portfolios :math:`S_1, S_2` without paying external reinsurance loading.
    Insurer 1 cedes proportion :math:`q_1 \in [0, 1]` to Insurer 2 and receives
    proportion :math:`q_2` of Insurer 2's portfolio.

    Financial balance (fair exchange) requires equal expected transfers:

    .. math::
        \mathbb{E}[q_1 S_1] = \mathbb{E}[q_2 S_2]
        \implies q_2 = q_1 \frac{\mathbb{E}[S_1]}{\mathbb{E}[S_2]}

    If expected losses are omitted, equal portfolio sizes are assumed (:math:`q_2 = q_1`).

    Post-exchange portfolio variances for given correlation :math:`\rho \in [-1, 1]` are:

    .. math::
        \text{Var}(Y_1) = (1 - q_1)^2 \text{Var}(S_1) + q_2^2 \text{Var}(S_2)
                          + 2(1 - q_1)q_2 \rho \sqrt{\text{Var}(S_1)\text{Var}(S_2)}
    .. math::
        \text{Var}(Y_2) = q_1^2 \text{Var}(S_1) + (1 - q_2)^2 \text{Var}(S_2)
                          + 2 q_1(1 - q_2) \rho \sqrt{\text{Var}(S_1)\text{Var}(S_2)}

    Args:
        variance_1: Aggregate loss variance of Insurer 1 (:math:`\text{Var}(S_1) > 0`).
        variance_2: Aggregate loss variance of Insurer 2 (:math:`\text{Var}(S_2) > 0`).
        expected_loss_1: Expected loss of Insurer 1 (:math:`\mathbb{E}[S_1] > 0`). Optional.
        expected_loss_2: Expected loss of Insurer 2 (:math:`\mathbb{E}[S_2] > 0`). Optional.
        quota_1: Cession proportion of Insurer 1 (:math:`q_1 \in [0.0, 1.0]`). Defaults to 0.5.
        correlation: Pearson correlation :math:`\rho \in [-1.0, 1.0]`. Defaults to 0.0.

    Returns:
        dict[str, NDArray[np.float64]]: Dictionary containing:
            - ``"post_variance_1"``: Variance of Insurer 1 after exchange.
            - ``"post_variance_2"``: Variance of Insurer 2 after exchange.
            - ``"variance_reduction_1"``: Percentage variance reduction for Insurer 1:
              :math:`1 - \text{Var}(Y_1)/\text{Var}(S_1)`.
            - ``"variance_reduction_2"``: Percentage variance reduction for Insurer 2:
              :math:`1 - \text{Var}(Y_2)/\text{Var}(S_2)`.
            - ``"total_reciprocity_gain"``: Overall system variance reduction:
              :math:`1 - (\text{Var}(Y_1) + \text{Var}(Y_2)) / (\text{Var}(S_1) + \text{Var}(S_2))`.
            - ``"quota_2"``: Balanced cession proportion for Insurer 2 (:math:`q_2`).

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), variances or expected
            losses are non-positive, :math:`q_1 \notin [0, 1]`, or :math:`\rho \notin [-1, 1]`.

    Notes:
        Validation status: experimental.

    References:
        De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory (3rd ed.). Chapter 6.

    Examples:
        >>> from therisktheory.severity_aggregate import insurer_reciprocity_metric
        >>> res = insurer_reciprocity_metric(
        ...     variance_1=100000.0,
        ...     variance_2=100000.0,
        ...     quota_1=0.5,
        ...     correlation=0.0,
        ... )
        >>> float(res["total_reciprocity_gain"])
        0.5
    """
    v1 = np.asarray(variance_1, dtype=np.float64)
    v2 = np.asarray(variance_2, dtype=np.float64)
    q1 = np.asarray(quota_1, dtype=np.float64)
    rho = np.asarray(correlation, dtype=np.float64)

    if not (
        np.all(np.isfinite(v1))
        and np.all(np.isfinite(v2))
        and np.all(np.isfinite(q1))
        and np.all(np.isfinite(rho))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(v1 <= 0.0) or np.any(v2 <= 0.0):
        raise ValueError("Portfolio variances must be strictly positive")
    if np.any((q1 < 0.0) | (q1 > 1.0)):
        raise ValueError("quota_1 must be between 0.0 and 1.0 inclusive")
    if np.any((rho < -1.0) | (rho > 1.0)):
        raise ValueError("correlation must be between -1.0 and 1.0 inclusive")

    if expected_loss_1 is not None and expected_loss_2 is not None:
        e1 = np.asarray(expected_loss_1, dtype=np.float64)
        e2 = np.asarray(expected_loss_2, dtype=np.float64)
        if not (np.all(np.isfinite(e1)) and np.all(np.isfinite(e2))):
            raise ValueError("expected_loss must contain only finite numbers")
        if np.any(e1 <= 0.0) or np.any(e2 <= 0.0):
            raise ValueError("Expected losses must be strictly positive")
        q2 = q1 * (e1 / e2)
        if np.any(q2 > 1.0):
            raise ValueError(f"Financial balance requires quota_2 > 1.0 ({q2}); reduce quota_1")
    else:
        q2 = q1

    cov = rho * np.sqrt(v1 * v2)

    var_post_1 = ((1.0 - q1) ** 2) * v1 + (q2**2) * v2 + 2.0 * (1.0 - q1) * q2 * cov
    var_post_2 = (q1**2) * v1 + ((1.0 - q2) ** 2) * v2 + 2.0 * q1 * (1.0 - q2) * cov

    red_1 = 1.0 - (var_post_1 / v1)
    red_2 = 1.0 - (var_post_2 / v2)
    tot_gain = 1.0 - ((var_post_1 + var_post_2) / (v1 + v2))

    return {
        "post_variance_1": np.asarray(var_post_1, dtype=np.float64),
        "post_variance_2": np.asarray(var_post_2, dtype=np.float64),
        "variance_reduction_1": np.asarray(red_1, dtype=np.float64),
        "variance_reduction_2": np.asarray(red_2, dtype=np.float64),
        "total_reciprocity_gain": np.asarray(tot_gain, dtype=np.float64),
        "quota_2": np.asarray(q2, dtype=np.float64),
    }
