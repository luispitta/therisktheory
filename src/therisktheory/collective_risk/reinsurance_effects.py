"""Reinsurance effects on the collective risk process, Lundberg bounds, and ruin.

This module formalizes the impact of proportional and excess-of-loss reinsurance on the
classical compound Poisson surplus process, deriving retained premium rates, retained
claim moments, the Lundberg adjustment coefficient, Lundberg ruin upper bounds, exact
Cramér-Lundberg ruin probabilities, and solvency capital relief.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, minimize_scalar


def proportional_reinsurance_process(
    claim_rate: float | ArrayLike,
    claim_mean: float | ArrayLike,
    claim_variance: float | ArrayLike,
    safety_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    retention_rate: float | ArrayLike = 1.0,
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    r"""Evaluate the collective risk process parameters under proportional reinsurance.

    For a classical compound Poisson surplus process :math:`U(t) = u + c t - S(t)` with
    claim rate :math:`\lambda > 0`, gross claim mean :math:`\mu_1 > 0`, claim variance
    :math:`\sigma^2 > 0`, and cedent safety loading :math:`\theta > 0`, gross premium income
    rate is:

    .. math::
        c = (1 + \theta) \lambda \mu_1

    Under proportional reinsurance with retention rate :math:`\alpha \in [0.0, 1.0]`, retained
    claims are :math:`X_{\text{ret}} = \alpha X`, yielding retained moments:

    .. math::
        \mu_{1, \text{ret}}(\alpha) = \alpha \mu_1, \quad
        \sigma_{\text{ret}}^2(\alpha) = \alpha^2 \sigma^2, \quad
        \mu_{2, \text{ret}}(\alpha) = \alpha^2 (\sigma^2 + \mu_1^2)

    The reinsurer charges safety loading :math:`\xi \ge 0` on ceded losses :math:`(1 - \alpha)X`,
    so the net retained premium income rate is:

    .. math::
        c_{\text{ret}}(\alpha) = c - c_{\text{re}}(\alpha)
        = (1 + \theta) \lambda \mu_1 - (1 + \xi) \lambda (1 - \alpha) \mu_1
        = \lambda \mu_1 \left[ \theta - \xi(1 - \alpha) + \alpha \right]

    The retained relative safety loading is:

    .. math::
        \theta_{\text{ret}}(\alpha) = \frac{c_{\text{ret}}(\alpha)}{\lambda \mu_{1,\text{ret}}} - 1
        = \frac{\theta - \xi(1 - \alpha)}{\alpha} \quad (\alpha > 0)

    To guarantee positive annual drift (preventing certain ruin :math:`\psi \equiv 1.0`), the net
    profit condition requires :math:`\theta - \xi(1 - \alpha) > 0`, yielding the minimum viable
    retention:

    .. math::
        \alpha_{\min} = \max\left(0.0, 1.0 - \frac{\theta}{\xi}\right)

    Args:
        claim_rate: Poisson claim intensity :math:`\lambda > 0`.
        claim_mean: Individual claim expected value :math:`\mu_1 > 0`.
        claim_variance: Individual claim variance :math:`\sigma^2 > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer safety loading :math:`\xi \ge 0`.
        retention_rate: Retained proportion :math:`\alpha \in [0.0, 1.0]`. Defaults to 1.0.

    Returns:
        dict[str, NDArray[np.float64] | NDArray[np.bool_]]: Dictionary containing:
            - ``"retained_premium_rate"``: Net premium income :math:`c_{\text{ret}}(\alpha)`.
            - ``"retained_safety_loading"``: Retained loading :math:`\theta_{\text{ret}}(\alpha)`.
            - ``"retained_claim_mean"``: :math:`\mu_{1,\text{ret}}(\alpha) = \alpha \mu_1`.
            - ``"retained_claim_variance"``: Retained claim variance :math:`\alpha^2 \sigma^2`.
            - ``"retained_claim_second_moment"``: :math:`\mu_{2, \text{ret}}(\alpha)`.
            - ``"expected_annual_drift"``: Net annual drift rate.
            - ``"minimum_viable_retention"``: Minimum viable retention :math:`\alpha_{\min}`.
            - ``"is_solvent_drift"``: Boolean indicator whether :math:`\theta_{\text{ret}} > 0`.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf) or violate domain bounds
            (:math:`\lambda \le 0`, :math:`\mu_1 \le 0`, :math:`\sigma^2 \le 0`,
            :math:`\theta \le 0`, :math:`\xi < 0`, or :math:`\alpha \notin [0.0, 1.0]`).

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). Modern Actuarial Risk Theory:
        Using R (2nd ed.). Springer, Chapter 4, Section 4.6 (pp. 100–104).
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Society of Actuaries,
        Chapter 14, Section 14.5.

    Examples:
        >>> from therisktheory.collective_risk import proportional_reinsurance_process
        >>> res = proportional_reinsurance_process(
        ...     claim_rate=100.0,
        ...     claim_mean=2.0,
        ...     claim_variance=4.0,
        ...     safety_loading=0.20,
        ...     reinsurer_loading=0.30,
        ...     retention_rate=0.80,
        ... )
        >>> float(res["retained_safety_loading"])
        0.175
    """
    lam = np.asarray(claim_rate, dtype=np.float64)
    mu1 = np.asarray(claim_mean, dtype=np.float64)
    var = np.asarray(claim_variance, dtype=np.float64)
    theta = np.asarray(safety_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)
    alpha = np.asarray(retention_rate, dtype=np.float64)

    if not (
        np.all(np.isfinite(lam))
        and np.all(np.isfinite(mu1))
        and np.all(np.isfinite(var))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(xi))
        and np.all(np.isfinite(alpha))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(lam <= 0.0):
        raise ValueError("claim_rate must be strictly positive")
    if np.any(mu1 <= 0.0):
        raise ValueError("claim_mean must be strictly positive")
    if np.any(var <= 0.0):
        raise ValueError("claim_variance must be strictly positive")
    if np.any(theta <= 0.0):
        raise ValueError("safety_loading must be strictly positive")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("retention_rate must be between 0.0 and 1.0 inclusive")

    mu1_ret = alpha * mu1
    var_ret = (alpha**2) * var
    mu2_ret = var_ret + (mu1_ret**2)

    # Net retained premium rate: c_ret = lambda * mu1 * [(1 + theta) - (1 + xi)*(1 - alpha)]
    c_ret = lam * mu1 * ((1.0 + theta) - (1.0 + xi) * (1.0 - alpha))

    # Retained safety loading: theta_ret = (theta - xi*(1 - alpha)) / alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_ret = np.where(
            alpha > 0.0,
            (theta - xi * (1.0 - alpha)) / alpha,
            np.where(theta >= xi, np.inf, -np.inf),
        )

    # Net annual drift: c_ret - lambda * mu1_ret
    net_drift = c_ret - lam * mu1_ret

    # Minimum viable retention: alpha_min = max(0, 1 - theta / xi)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_min = np.where(xi > 0.0, np.maximum(0.0, 1.0 - (theta / xi)), 0.0)

    is_solvent = net_drift > 0.0

    return {
        "retained_premium_rate": np.asarray(c_ret, dtype=np.float64),
        "retained_safety_loading": np.asarray(theta_ret, dtype=np.float64),
        "retained_claim_mean": np.asarray(mu1_ret, dtype=np.float64),
        "retained_claim_variance": np.asarray(var_ret, dtype=np.float64),
        "retained_claim_second_moment": np.asarray(mu2_ret, dtype=np.float64),
        "expected_annual_drift": np.asarray(net_drift, dtype=np.float64),
        "minimum_viable_retention": np.asarray(alpha_min, dtype=np.float64),
        "is_solvent_drift": np.asarray(is_solvent, dtype=np.bool_),
    }


def excess_of_loss_reinsurance_process(
    claim_rate: float | ArrayLike,
    claim_mean: float | ArrayLike,
    priority: float | ArrayLike,
    safety_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    distribution: Literal["exponential"] = "exponential",
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    r"""Evaluate the collective risk process parameters under excess-of-loss reinsurance.

    For an excess-of-loss treaty with priority (deductible) :math:`d > 0`, the insurer retains
    individual losses up to :math:`d`:

    .. math::
        X_{\text{ret}} = \min(X, d), \quad X_{\text{ced}} = \max(0, X - d)

    For exponentially distributed claim severities :math:`X \sim \text{Exp}(\beta)` with mean
    :math:`\mu_1 = 1 / \beta`:

    .. math::
        \mu_{1, \text{ret}}(d) = \mathbb{E}[\min(X, d)] = \mu_1 \left(1 - e^{-d / \mu_1}\right)
    .. math::
        \mathbb{E}[X_{\text{ced}}] = \mu_1 e^{-d / \mu_1}
    .. math::
        \mu_{2, \text{ret}}(d) = \mathbb{E}[\min(X, d)^2]
        = 2 \mu_1^2 \left[ 1 - e^{-d / \mu_1}\left(1 + \frac{d}{\mu_1}\right) \right]

    The retained premium rate and safety loading are:

    .. math::
        c_{\text{ret}}(d) = (1 + \theta)\lambda\mu_1 - (1 + \xi)\lambda\mathbb{E}[X_{\text{ced}}]
    .. math::
        \theta_{\text{ret}}(d) = \frac{c_{\text{ret}}(d)}{\lambda \mu_{1, \text{ret}}(d)} - 1

    Args:
        claim_rate: Poisson claim rate :math:`\lambda > 0`.
        claim_mean: Gross claim expected value :math:`\mu_1 > 0`.
        priority: Retention priority (attachment point) :math:`d > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer loading :math:`\xi \ge 0`.
        distribution: Individual claim distribution. Defaults to ``"exponential"``.

    Returns:
        dict[str, NDArray[np.float64] | NDArray[np.bool_]]: Dictionary containing:
            - ``"retained_premium_rate"``: Net premium rate :math:`c_{\text{ret}}(d)`.
            - ``"retained_safety_loading"``: Retained loading :math:`\theta_{\text{ret}}(d)`.
            - ``"retained_claim_mean"``: Retained expected severity :math:`\mu_{1, \text{ret}}(d)`.
            - ``"ceded_claim_mean"``: Expected ceded loss :math:`\mathbb{E}[X_{\text{ced}}]`.
            - ``"retained_claim_variance"``: Retained severity variance.
            - ``"retained_claim_second_moment"``: Retained raw second moment.
            - ``"expected_annual_drift"``: Net drift rate.
            - ``"is_solvent_drift"``: Boolean indicator whether :math:`\theta_{\text{ret}} > 0`.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf) or violate domain bounds
            (:math:`\lambda \le 0`, :math:`\mu_1 \le 0`, :math:`d \le 0`, :math:`\theta \le 0`,
            :math:`\xi < 0`), or if ``distribution`` is unsupported.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Section 4.6.
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Chapter 14.

    Examples:
        >>> from therisktheory.collective_risk import excess_of_loss_reinsurance_process
        >>> res = excess_of_loss_reinsurance_process(
        ...     claim_rate=50.0,
        ...     claim_mean=100.0,
        ...     priority=150.0,
        ...     safety_loading=0.20,
        ...     reinsurer_loading=0.30,
        ... )
        >>> float(res["retained_safety_loading"]) > 0.0
        True
    """
    lam = np.asarray(claim_rate, dtype=np.float64)
    mu1 = np.asarray(claim_mean, dtype=np.float64)
    d = np.asarray(priority, dtype=np.float64)
    theta = np.asarray(safety_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)

    if not (
        np.all(np.isfinite(lam))
        and np.all(np.isfinite(mu1))
        and np.all(np.isfinite(d))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(xi))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(lam <= 0.0):
        raise ValueError("claim_rate must be strictly positive")
    if np.any(mu1 <= 0.0):
        raise ValueError("claim_mean must be strictly positive")
    if np.any(d <= 0.0):
        raise ValueError("priority must be strictly positive")
    if np.any(theta <= 0.0):
        raise ValueError("safety_loading must be strictly positive")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")

    if distribution != "exponential":
        raise ValueError(f"Unsupported distribution: {distribution!r}")

    beta = 1.0 / mu1
    exp_factor = np.exp(-beta * d)

    mu1_ret = mu1 * (1.0 - exp_factor)
    mu1_ced = mu1 * exp_factor
    mu2_ret = 2.0 * (mu1**2) * (1.0 - exp_factor * (1.0 + beta * d))
    var_ret = np.maximum(0.0, mu2_ret - (mu1_ret**2))

    c_gross = (1.0 + theta) * lam * mu1
    c_re = (1.0 + xi) * lam * mu1_ced
    c_ret = c_gross - c_re

    with np.errstate(divide="ignore", invalid="ignore"):
        theta_ret = (c_ret / (lam * mu1_ret)) - 1.0

    net_drift = c_ret - lam * mu1_ret
    is_solvent = net_drift > 0.0

    return {
        "retained_premium_rate": np.asarray(c_ret, dtype=np.float64),
        "retained_safety_loading": np.asarray(theta_ret, dtype=np.float64),
        "retained_claim_mean": np.asarray(mu1_ret, dtype=np.float64),
        "ceded_claim_mean": np.asarray(mu1_ced, dtype=np.float64),
        "retained_claim_variance": np.asarray(var_ret, dtype=np.float64),
        "retained_claim_second_moment": np.asarray(mu2_ret, dtype=np.float64),
        "expected_annual_drift": np.asarray(net_drift, dtype=np.float64),
        "is_solvent_drift": np.asarray(is_solvent, dtype=np.bool_),
    }


def _excess_of_loss_lundberg_equation(
    r: float,
    beta: float,
    priority: float,
    claim_rate: float,
    reinsured_premium_rate: float,
) -> float:
    """Evaluates the Lundberg equation for Excess of Loss reinsurance with exponential claims.

    Lundberg equation: g(R) = lambda * (m_Xret(R) - 1) - c_ret * R = 0.
    At R = beta, the removable singularity evaluates to m_Xret(beta) = 1 + beta * priority.
    """
    if abs(r - beta) < 1e-12:
        mgf = 1.0 + beta * priority
    else:
        mgf = (beta - r * np.exp(-(beta - r) * priority)) / (beta - r)
    return float(claim_rate * (mgf - 1.0) - reinsured_premium_rate * r)


def reinsurance_adjustment_coefficient(
    claim_rate: float | ArrayLike,
    claim_mean: float | ArrayLike,
    safety_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    retention_param: float | ArrayLike,
    treaty_type: Literal["proportional", "excess_of_loss"] = "proportional",
    distribution: Literal["exponential"] = "exponential",
) -> NDArray[np.float64]:
    r"""Compute the Lundberg adjustment coefficient under reinsurance.

    The adjustment coefficient :math:`R > 0` is the unique positive root of Lundberg's
    fundamental equation:

    .. math::
        \lambda \left( m_{X_{\text{ret}}}(R) - 1 \right) - c_{\text{ret}} R = 0
        \iff 1 + (1 + \theta_{\text{ret}}) \mu_{1, \text{ret}} R = m_{X_{\text{ret}}}(R)

    1. **Proportional Reinsurance with Exponential Claims (:math:`X \sim \text{Exp}(\beta)`):**
       Since :math:`X_{\text{ret}} = \alpha X \sim \text{Exp}(\beta / \alpha)`, the retained
       adjustment coefficient has the exact closed-form solution:

       .. math::
           R(\alpha) = \frac{\beta}{\alpha} \frac{\theta_{\text{ret}}}{1 + \theta_{\text{ret}}}
           = \frac{\beta [\theta - \xi(1 - \alpha)]}{\alpha [\alpha + \theta - \xi(1 - \alpha)]}

       If :math:`\theta_{\text{ret}}(\alpha) \le 0` (net non-positive drift), :math:`R = 0.0`.

    2. **Excess-of-Loss Reinsurance with Exponential Claims (:math:`X \sim \text{Exp}(\beta)`):**
       The retained moment-generating function is:

       .. math::
           m_{\min(X, d)}(R) = \frac{\beta - R e^{-(\beta - R)d}}{\beta - R} \quad (R < \beta)

       Lundberg's equation is solved numerically on :math:`(0, \beta)` via Brent's method.

    Args:
        claim_rate: Poisson claim rate :math:`\lambda > 0`.
        claim_mean: Gross claim expected value :math:`\mu_1 > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer safety loading :math:`\xi \ge 0`.
        retention_param: Retention proportion :math:`\alpha \in [0.0, 1.0]` for proportional,
            or priority :math:`d > 0` for excess-of-loss.
        treaty_type: Type of reinsurance treaty (``"proportional"`` or ``"excess_of_loss"``).
        distribution: Claim distribution family. Defaults to ``"exponential"``.

    Returns:
        NDArray[np.float64]: Lundberg adjustment coefficient :math:`R \ge 0`.
        NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf) or violate domain bounds,
            or if ``treaty_type`` / ``distribution`` is unsupported.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer,
        Section 4.3 (eq. 4.13, p. 92) and Section 4.6 (eq. 4.45, p. 102).
        Rolski, T., Schmidli, H., Schmidt, V., & Teugels, J. (1999). Stochastic Processes
        for Insurance and Finance. Wiley, Chapter 5.

    Examples:
        >>> from therisktheory.collective_risk import reinsurance_adjustment_coefficient
        >>> # Exp claims mean 1.0 (beta = 1.0), theta = 0.20, no reinsurance (alpha = 1)
        >>> r_gross = reinsurance_adjustment_coefficient(
        ...     claim_rate=1.0,
        ...     claim_mean=1.0,
        ...     safety_loading=0.20,
        ...     reinsurer_loading=0.30,
        ...     retention_param=1.0,
        ... )
        >>> float(r_gross)
        0.16666666666666666
    """
    lam = np.asarray(claim_rate, dtype=np.float64)
    mu1 = np.asarray(claim_mean, dtype=np.float64)
    theta = np.asarray(safety_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)
    param = np.asarray(retention_param, dtype=np.float64)

    if not (
        np.all(np.isfinite(lam))
        and np.all(np.isfinite(mu1))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(xi))
        and np.all(np.isfinite(param))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(lam <= 0.0):
        raise ValueError("claim_rate must be strictly positive")
    if np.any(mu1 <= 0.0):
        raise ValueError("claim_mean must be strictly positive")
    if np.any(theta <= 0.0):
        raise ValueError("safety_loading must be strictly positive")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")

    if distribution != "exponential":
        raise ValueError(f"Unsupported distribution: {distribution!r}")

    beta = 1.0 / mu1

    if treaty_type == "proportional":
        if np.any((param < 0.0) | (param > 1.0)):
            raise ValueError("retention_param (alpha) must be between 0.0 and 1.0 inclusive")

        # Closed form: R(a) = beta * [theta - xi*(1 - a)] / (a * [a + theta - xi*(1 - a)])
        num = beta * (theta - xi * (1.0 - param))
        drift_factor = param + theta - xi * (1.0 - param)
        den = param * drift_factor

        with np.errstate(divide="ignore", invalid="ignore"):
            r_val = np.where(
                (param <= 0.0) | (num <= 0.0) | (drift_factor <= 0.0),
                0.0,
                num / den,
            )
        return np.asarray(r_val, dtype=np.float64)

    if treaty_type == "excess_of_loss":
        if np.any(param <= 0.0):
            raise ValueError("retention_param (priority d) must be strictly positive")

        # Solve root numerically for scalar / broadcasted inputs
        broadcasted = np.broadcast_arrays(lam, mu1, theta, xi, param)
        out = np.zeros(broadcasted[0].shape, dtype=np.float64)
        it = np.nditer(broadcasted, flags=["multi_index"])

        for _ in it:
            idx = it.multi_index
            l_val = float(broadcasted[0][idx])
            m_val = float(broadcasted[1][idx])
            th_val = float(broadcasted[2][idx])
            xi_val = float(broadcasted[3][idx])
            d_val = float(broadcasted[4][idx])

            b_val = 1.0 / m_val
            exp_term = np.exp(-b_val * d_val)
            mu1_ret = m_val * (1.0 - exp_term)
            mu1_ced = m_val * exp_term

            c_ret = (1.0 + th_val) * l_val * m_val - (1.0 + xi_val) * l_val * mu1_ced
            net_drift = c_ret - l_val * mu1_ret

            if net_drift <= 0.0:
                out[idx] = 0.0
                continue

            # Find upper bracket where lundberg equation is strictly positive
            r_upper = 0.1 * b_val
            while _excess_of_loss_lundberg_equation(r_upper, b_val, d_val, l_val, c_ret) <= 0.0:
                r_upper *= 2.0

            root = brentq(
                _excess_of_loss_lundberg_equation,
                1e-12,
                r_upper,
                args=(b_val, d_val, l_val, c_ret),
            )
            out[idx] = float(root)

        return np.asarray(out, dtype=np.float64)

    raise ValueError(f"treaty_type must be 'proportional' or 'excess_of_loss', got {treaty_type!r}")


def optimal_reinsurance_retention(
    claim_rate: float,
    claim_mean: float,
    safety_loading: float,
    reinsurer_loading: float,
    treaty_type: Literal["proportional", "excess_of_loss"] = "proportional",
    distribution: Literal["exponential"] = "exponential",
) -> dict[str, float]:
    r"""Find optimal retention maximizing the Lundberg adjustment coefficient.

    Maximizing the adjustment coefficient :math:`R` minimizes the asymptotic Lundberg
    upper bound on the ruin probability :math:`\psi(u) \le e^{-Ru}`:

    .. math::
        \alpha^* = \arg\max_{\alpha \in (\alpha_{\min}, 1]} R(\alpha)

    When :math:`\xi \le \theta`, the reinsurer provides cover cheaper than or equal to
    the direct insurer's loading. In the realistic actuarial case where :math:`\xi > \theta`,
    there exists a unique optimal retention :math:`\alpha^* \in (\alpha_{\min}, 1]` balancing
    variance reduction against loading cost dilution.

    Args:
        claim_rate: Poisson claim rate :math:`\lambda > 0`.
        claim_mean: Gross claim expected value :math:`\mu_1 > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer safety loading :math:`\xi > 0`.
        treaty_type: Treaty structure (``"proportional"`` or ``"excess_of_loss"``).
        distribution: Claim severity distribution. Defaults to ``"exponential"``.

    Returns:
        dict[str, float]: Optimization summary containing:
            - ``"optimal_retention"``: Optimal retention :math:`\alpha^*` or priority :math:`d^*`.
            - ``"optimal_adjustment_coefficient"``: Maximum achievable :math:`R(\alpha^*)`.
            - ``"gross_adjustment_coefficient"``: Gross benchmark :math:`R_{\text{gross}}`.
            - ``"relative_adjustment_gain"``: Relative gain :math:`(R^* - R) / R`.
            - ``"retained_safety_loading"``: Retained loading :math:`\theta_{\text{ret}}(\alpha^*)`.
            - ``"minimum_viable_retention"``: Minimum retention threshold :math:`\alpha_{\min}`.

    Raises:
        ValueError: If parameters are non-finite or violate domain bounds.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer,
        Section 4.6 (pp. 100–104).

    Examples:
        >>> from therisktheory.collective_risk import optimal_reinsurance_retention
        >>> opt = optimal_reinsurance_retention(
        ...     claim_rate=10.0,
        ...     claim_mean=100.0,
        ...     safety_loading=0.20,
        ...     reinsurer_loading=0.30,
        ...     treaty_type="proportional",
        ... )
        >>> opt["optimal_retention"] > opt["minimum_viable_retention"]
        True
        >>> opt["optimal_adjustment_coefficient"] > opt["gross_adjustment_coefficient"]
        True
    """
    for name, val in [
        ("claim_rate", claim_rate),
        ("claim_mean", claim_mean),
        ("safety_loading", safety_loading),
        ("reinsurer_loading", reinsurer_loading),
    ]:
        if not np.isfinite(val):
            raise ValueError(f"{name} must be a finite number")

    if claim_rate <= 0.0 or claim_mean <= 0.0 or safety_loading <= 0.0:
        raise ValueError("claim_rate, claim_mean, and safety_loading must be strictly positive")
    if reinsurer_loading <= 0.0:
        raise ValueError("reinsurer_loading must be strictly positive")

    beta = 1.0 / claim_mean
    r_gross = float((beta * safety_loading) / (1.0 + safety_loading))

    if treaty_type == "proportional":
        alpha_min = max(0.0, min(1.0 - 1e-12, 1.0 - (safety_loading / reinsurer_loading)))
        width = 1.0 - alpha_min
        lower_bound = alpha_min + 1e-4 * width

        def neg_r(alpha_val: float) -> float:
            r = reinsurance_adjustment_coefficient(
                claim_rate=claim_rate,
                claim_mean=claim_mean,
                safety_loading=safety_loading,
                reinsurer_loading=reinsurer_loading,
                retention_param=alpha_val,
                treaty_type="proportional",
                distribution=distribution,
            )
            return -float(np.asarray(r).item())

        opt_res = minimize_scalar(neg_r, bounds=(lower_bound, 1.0), method="bounded")
        alpha_opt = float(opt_res.x)
        r_opt = -float(opt_res.fun)

        th_ret = float((safety_loading - reinsurer_loading * (1.0 - alpha_opt)) / alpha_opt)
        gain = (r_opt - r_gross) / r_gross if r_gross > 0.0 else 0.0

        return {
            "optimal_retention": alpha_opt,
            "optimal_adjustment_coefficient": r_opt,
            "gross_adjustment_coefficient": r_gross,
            "relative_adjustment_gain": gain,
            "retained_safety_loading": th_ret,
            "minimum_viable_retention": alpha_min,
        }

    if treaty_type == "excess_of_loss":
        # Search priority in range [0.1 * claim_mean, 20.0 * claim_mean]
        def neg_r_xl(d_val: float) -> float:
            r = reinsurance_adjustment_coefficient(
                claim_rate=claim_rate,
                claim_mean=claim_mean,
                safety_loading=safety_loading,
                reinsurer_loading=reinsurer_loading,
                retention_param=d_val,
                treaty_type="excess_of_loss",
                distribution=distribution,
            )
            return -float(np.asarray(r).item())

        opt_res = minimize_scalar(
            neg_r_xl,
            bounds=(0.1 * claim_mean, 15.0 * claim_mean),
            method="bounded",
        )
        d_opt = float(opt_res.x)
        r_opt = -float(opt_res.fun)

        xl_proc = excess_of_loss_reinsurance_process(
            claim_rate=claim_rate,
            claim_mean=claim_mean,
            priority=d_opt,
            safety_loading=safety_loading,
            reinsurer_loading=reinsurer_loading,
            distribution=distribution,
        )
        th_ret = float(np.asarray(xl_proc["retained_safety_loading"]).item())
        gain = (r_opt - r_gross) / r_gross if r_gross > 0.0 else 0.0

        return {
            "optimal_retention": d_opt,
            "optimal_adjustment_coefficient": r_opt,
            "gross_adjustment_coefficient": r_gross,
            "relative_adjustment_gain": gain,
            "retained_safety_loading": th_ret,
            "minimum_viable_retention": 0.0,
        }

    raise ValueError(f"treaty_type must be 'proportional' or 'excess_of_loss', got {treaty_type!r}")


def reinsurance_ruin_bound(
    initial_capital: float | ArrayLike,
    claim_rate: float | ArrayLike,
    claim_mean: float | ArrayLike,
    safety_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    retention_rate: float | ArrayLike = 1.0,
    treaty_type: Literal["proportional"] = "proportional",
    method: Literal["cramer_lundberg_exact", "lundberg_upper_bound"] = "cramer_lundberg_exact",
) -> dict[str, NDArray[np.float64]]:
    r"""Evaluate ruin probabilities and Lundberg upper bounds under reinsurance.

    For initial reserve :math:`u \ge 0`, compares gross and reinsured ruin probabilities:

    1. **Lundberg Upper Bound:**
       $$\psi_{\text{bound}}(u) = \exp(-R u)$$

    2. **Exact Cramér-Lundberg Ruin Probability (Exponential Claims):**
       For exponential claims with proportional retention :math:`\alpha`:

       .. math::
           \psi_{\text{exact}}(u) = \frac{1}{1 + \theta_{\text{ret}}} \exp(-R_{\text{ret}} u)
           \quad (\text{for } u > 0, \theta_{\text{ret}} > 0)

       and :math:`\psi_{\text{exact}} = 1.0` if :math:`u = 0` or :math:`\theta_{\text{ret}} \le 0`.

    Args:
        initial_capital: Initial reserve capital :math:`u \ge 0`.
        claim_rate: Poisson claim rate :math:`\lambda > 0`.
        claim_mean: Gross claim expected value :math:`\mu_1 > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer safety loading :math:`\xi \ge 0`.
        retention_rate: Retained proportion :math:`\alpha \in [0.0, 1.0]`. Defaults to 1.0.
        treaty_type: Reinsurance treaty type. Defaults to ``"proportional"``.
        method: Evaluation method (``"cramer_lundberg_exact"`` or ``"lundberg_upper_bound"``).

    Returns:
        dict[str, NDArray[np.float64]]: Dictionary containing:
            - ``"reinsured_ruin_probability"``: :math:`\psi_{\text{ret}}(u)`.
            - ``"gross_ruin_probability"``: :math:`\psi_{\text{gross}}(u)`.
            - ``"ruin_reduction_ratio"``: Relative ruin reduction.
            - ``"reinsured_adjustment_coefficient"``: :math:`R_{\text{ret}}`.
            - ``"gross_adjustment_coefficient"``: :math:`R_{\text{gross}}`.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf) or violate domain bounds
            (:math:`u < 0`, :math:`\lambda \le 0`, :math:`\mu_1 \le 0`, :math:`\theta \le 0`,
            :math:`\xi < 0`, :math:`\alpha \notin [0.0, 1.0]`).

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer,
        Chapter 4, Section 4.6 (pp. 100–104).
        Asmussen, S., & Albrecher, H. (2010). Ruin Probabilities (2nd ed.). World Scientific.

    Examples:
        >>> from therisktheory.collective_risk import reinsurance_ruin_bound
        >>> res = reinsurance_ruin_bound(
        ...     initial_capital=50.0,
        ...     claim_rate=1.0,
        ...     claim_mean=1.0,
        ...     safety_loading=0.20,
        ...     reinsurer_loading=0.30,
        ...     retention_rate=0.70,
        ... )
        >>> float(res["reinsured_ruin_probability"]) < float(res["gross_ruin_probability"])
        True
    """
    u = np.asarray(initial_capital, dtype=np.float64)
    lam = np.asarray(claim_rate, dtype=np.float64)
    mu1 = np.asarray(claim_mean, dtype=np.float64)
    theta = np.asarray(safety_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)
    alpha = np.asarray(retention_rate, dtype=np.float64)

    if not (
        np.all(np.isfinite(u))
        and np.all(np.isfinite(lam))
        and np.all(np.isfinite(mu1))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(xi))
        and np.all(np.isfinite(alpha))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(u < 0.0):
        raise ValueError("initial_capital must be non-negative")
    if np.any(lam <= 0.0):
        raise ValueError("claim_rate must be strictly positive")
    if np.any(mu1 <= 0.0):
        raise ValueError("claim_mean must be strictly positive")
    if np.any(theta <= 0.0):
        raise ValueError("safety_loading must be strictly positive")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("retention_rate must be between 0.0 and 1.0 inclusive")

    beta = 1.0 / mu1

    # Gross calculations
    r_gross = (beta * theta) / (1.0 + theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "cramer_lundberg_exact":
            psi_gross = np.where(u == 0.0, 1.0, (1.0 / (1.0 + theta)) * np.exp(-r_gross * u))
        else:
            psi_gross = np.where(u == 0.0, 1.0, np.exp(-r_gross * u))

    # Reinsured calculations
    proc = proportional_reinsurance_process(lam, mu1, mu1**2, theta, xi, alpha)
    th_ret = proc["retained_safety_loading"]
    r_ret = reinsurance_adjustment_coefficient(
        lam, mu1, theta, xi, alpha, treaty_type="proportional"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "cramer_lundberg_exact":
            normal_psi_ret = (1.0 / (1.0 + th_ret)) * np.exp(-r_ret * u)
            psi_ret = np.where((u == 0.0) | (th_ret <= 0.0) | (r_ret <= 0.0), 1.0, normal_psi_ret)
        else:
            normal_psi_ret = np.exp(-r_ret * u)
            psi_ret = np.where((u == 0.0) | (th_ret <= 0.0) | (r_ret <= 0.0), 1.0, normal_psi_ret)

    psi_gross_clipped = np.clip(psi_gross, 0.0, 1.0)
    psi_ret_clipped = np.clip(psi_ret, 0.0, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = np.where(
            psi_gross_clipped > 0.0,
            1.0 - (psi_ret_clipped / psi_gross_clipped),
            0.0,
        )

    return {
        "reinsured_ruin_probability": np.asarray(psi_ret_clipped, dtype=np.float64),
        "gross_ruin_probability": np.asarray(psi_gross_clipped, dtype=np.float64),
        "ruin_reduction_ratio": np.asarray(reduction, dtype=np.float64),
        "reinsured_adjustment_coefficient": np.asarray(r_ret, dtype=np.float64),
        "gross_adjustment_coefficient": np.asarray(r_gross, dtype=np.float64),
    }


def reinsurance_capital_relief(
    ruin_tolerance: float | ArrayLike,
    claim_rate: float | ArrayLike,
    claim_mean: float | ArrayLike,
    safety_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    retention_rate: float | ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    r"""Compute required solvency capital and capital relief achieved via reinsurance.

    For a target maximum ruin probability :math:`\epsilon \in (0, 1)` under the exact
    Cramér-Lundberg formula for exponential claims:

    .. math::
        \psi(u) = \frac{1}{1 + \theta} e^{-R u} = \epsilon
        \implies u_{\epsilon} = \frac{1}{R} \ln\left(\frac{1}{(1 + \theta)\epsilon}\right)

    Reinsurance with retention :math:`\alpha` increases the adjustment coefficient to
    :math:`R_{\text{ret}} > R_{\text{gross}}`, reducing required capital to
    :math:`u_{\epsilon}^{\text{re}}`. The solvency capital relief is:

    .. math::
        \Delta u = u_{\epsilon}^{\text{gross}} - u_{\epsilon}^{\text{re}}

    Args:
        ruin_tolerance: Solvency ruin probability tolerance :math:`\epsilon \in (0, 1)`.
        claim_rate: Poisson claim rate :math:`\lambda > 0`.
        claim_mean: Gross claim expected value :math:`\mu_1 > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer safety loading :math:`\xi \ge 0`.
        retention_rate: Retained proportion :math:`\alpha \in (\alpha_{\min}, 1.0]`.

    Returns:
        dict[str, NDArray[np.float64]]: Dictionary containing:
            - ``"gross_required_capital"``: Required capital without reinsurance.
            - ``"reinsured_required_capital"``: Required capital with reinsurance.
            - ``"capital_relief"``: Absolute capital reduction :math:`\Delta u`.
            - ``"percentage_capital_relief"``: Percentage capital relief.

    Raises:
        ValueError: If inputs contain non-finite numbers or violate domain bounds
            (:math:`\epsilon \notin (0, 1)` or non-positive drift).

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Section 4.6.
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Chapter 14.

    Examples:
        >>> from therisktheory.collective_risk import reinsurance_capital_relief
        >>> relief = reinsurance_capital_relief(
        ...     ruin_tolerance=0.01,
        ...     claim_rate=1.0,
        ...     claim_mean=100.0,
        ...     safety_loading=0.20,
        ...     reinsurer_loading=0.30,
        ...     retention_rate=0.70,
        ... )
        >>> float(relief["capital_relief"]) > 0.0
        True
    """
    eps = np.asarray(ruin_tolerance, dtype=np.float64)
    lam = np.asarray(claim_rate, dtype=np.float64)
    mu1 = np.asarray(claim_mean, dtype=np.float64)
    theta = np.asarray(safety_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)
    alpha = np.asarray(retention_rate, dtype=np.float64)

    if not (
        np.all(np.isfinite(eps))
        and np.all(np.isfinite(lam))
        and np.all(np.isfinite(mu1))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(xi))
        and np.all(np.isfinite(alpha))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any((eps <= 0.0) | (eps >= 1.0)):
        raise ValueError("ruin_tolerance must be strictly between 0.0 and 1.0")
    if np.any(lam <= 0.0) or np.any(mu1 <= 0.0) or np.any(theta <= 0.0):
        raise ValueError("claim_rate, claim_mean, and safety_loading must be strictly positive")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")
    if np.any((alpha <= 0.0) | (alpha > 1.0)):
        raise ValueError("retention_rate must be strictly positive and <= 1.0")

    proc = proportional_reinsurance_process(lam, mu1, mu1**2, theta, xi, alpha)
    th_ret = proc["retained_safety_loading"]
    if np.any(th_ret <= 0.0):
        raise ValueError("retention_rate violates net profit condition (theta_ret <= 0)")

    r_ret = reinsurance_adjustment_coefficient(
        lam, mu1, theta, xi, alpha, treaty_type="proportional"
    )
    r_gross = reinsurance_adjustment_coefficient(
        lam, mu1, theta, xi, np.ones_like(alpha), treaty_type="proportional"
    )

    u_gross = np.maximum(0.0, (1.0 / r_gross) * np.log(1.0 / ((1.0 + theta) * eps)))
    u_ret = np.maximum(0.0, (1.0 / r_ret) * np.log(1.0 / ((1.0 + th_ret) * eps)))

    diff = u_gross - u_ret
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(u_gross > 0.0, (diff / u_gross) * 100.0, 0.0)

    return {
        "gross_required_capital": np.asarray(u_gross, dtype=np.float64),
        "reinsured_required_capital": np.asarray(u_ret, dtype=np.float64),
        "capital_relief": np.asarray(diff, dtype=np.float64),
        "percentage_capital_relief": np.asarray(pct, dtype=np.float64),
    }


def compare_de_finetti_lundberg_retention(
    claim_mean: float,
    claim_variance: float,
    safety_loading: float,
    reinsurer_loading: float,
    claim_rate: float = 1.0,
) -> dict[str, Any]:
    r"""Compare De Finetti safety index retention with Lundberg collective risk retention.

    Contrasts the optimal proportional retention obtained under two classical actuarial frameworks:

    1. **De Finetti's Theory:**
       Based on the discrete game / diffusion Brownian motion approximation, maximizing the
       local safety index:

       .. math::
           \theta(\alpha) = \frac{\mathbb{E}[\Delta U]}{\text{Var}(U_{\text{ret}})}
           = \frac{\lambda \mu_1 [\theta - \xi(1 - \alpha)]}{\lambda \alpha^2 (\sigma^2 + \mu_1^2)}

    2. **Lundberg's Collective Risk Theory:**
       Based on the exact compound Poisson jump process, maximizing the global adjustment
       coefficient :math:`R(\alpha)` solving the fundamental moment-generating function equation:

       .. math::
           \lambda \left( m_X(\alpha R) - 1 \right) - c_{\text{ret}}(\alpha) R = 0

    Args:
        claim_mean: Gross claim expected value :math:`\mu_1 > 0`.
        claim_variance: Gross claim variance :math:`\sigma^2 > 0`.
        safety_loading: Cedent safety loading :math:`\theta > 0`.
        reinsurer_loading: Reinsurer loading :math:`\xi > 0`.
        claim_rate: Poisson intensity :math:`\lambda > 0`. Defaults to 1.0.

    Returns:
        dict[str, Any]: Comparative dictionary containing:
            - ``"de_finetti_optimal_retention"``: Retention maximizing De Finetti index.
            - ``"lundberg_optimal_retention"``: Retention maximizing Lundberg coefficient R.
            - ``"retention_difference"``: Difference :math:`\alpha_{\text{DF}}^* - \alpha_L^*`.
            - ``"de_finetti_safety_index"``: Maximum De Finetti safety index.
            - ``"lundberg_adjustment_coefficient"``: Maximum Lundberg coefficient :math:`R^*`.
            - ``"commentary"``: Analytical explanation of model divergence.

    Raises:
        ValueError: If inputs contain non-finite numbers or violate domain bounds.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Section 4.6.
        De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory (3rd ed.).
    """
    for name, val in [
        ("claim_mean", claim_mean),
        ("claim_variance", claim_variance),
        ("safety_loading", safety_loading),
        ("reinsurer_loading", reinsurer_loading),
        ("claim_rate", claim_rate),
    ]:
        if not np.isfinite(val):
            raise ValueError(f"{name} must be a finite number")

    if (
        claim_mean <= 0.0
        or claim_variance <= 0.0
        or safety_loading <= 0.0
        or reinsurer_loading <= 0.0
        or claim_rate <= 0.0
    ):
        raise ValueError("All parameters must be strictly positive")

    alpha_min = max(0.0, min(1.0 - 1e-12, 1.0 - (safety_loading / reinsurer_loading)))
    width = 1.0 - alpha_min
    low_bound = alpha_min + 1e-4 * width

    # 1. De Finetti optimum: maximize safety index
    mu2 = claim_variance + (claim_mean**2)

    def neg_de_finetti_index(a: float) -> float:
        drift = safety_loading - reinsurer_loading * (1.0 - a)
        var_term = (a**2) * mu2
        return -float(drift / var_term)

    opt_df = minimize_scalar(neg_de_finetti_index, bounds=(low_bound, 1.0), method="bounded")
    alpha_df = float(opt_df.x)
    idx_df = -float(opt_df.fun)

    # 2. Lundberg optimum: maximize adjustment coefficient R(alpha)
    opt_lund = optimal_reinsurance_retention(
        claim_rate=claim_rate,
        claim_mean=claim_mean,
        safety_loading=safety_loading,
        reinsurer_loading=reinsurer_loading,
        treaty_type="proportional",
        distribution="exponential",
    )
    alpha_lund = float(opt_lund["optimal_retention"])
    r_lund = float(opt_lund["optimal_adjustment_coefficient"])

    diff = alpha_df - alpha_lund
    commentary = (
        "De Finetti maximizes the second-order diffusion safety index drift/variance, "
        "penalizing retained variance quadratically. Lundberg maximizes the exact decay R "
        "from the complete moment generating function, penalizing all jump moments. "
        "The two criteria converge when claim severities approach the Brownian diffusion limit."
    )

    return {
        "de_finetti_optimal_retention": alpha_df,
        "lundberg_optimal_retention": alpha_lund,
        "retention_difference": diff,
        "de_finetti_safety_index": idx_df,
        "lundberg_adjustment_coefficient": r_lund,
        "commentary": commentary,
    }
