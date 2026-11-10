"""De Finetti's risk and game-theoretic framework applied to proportional reinsurance.

This module provides human-validated, vectorized implementations for the proportional
reinsurance retention problem, objective functions, optimal retention derivation,
and ruin probability approximations following Bruno de Finetti's risk theory.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def proportional_reinsurance_objective(
    retention_rate: float | ArrayLike,
    expected_loss: float | ArrayLike,
    variance_loss: float | ArrayLike,
    insurer_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    criterion: Literal[
        "safety_index", "mean_variance", "expected_profit", "retained_variance"
    ] = "safety_index",
    risk_aversion: float | ArrayLike | None = None,
) -> NDArray[np.float64]:
    r"""Compute De Finetti's objective function for proportional reinsurance.

    Under De Finetti's game-theoretic and risk framework, an insurer with gross
    claims :math:`X` (mean :math:`\mu = \mathbb{E}[X]`, variance :math:`\sigma^2 = \text{Var}(X)`)
    and premium loading :math:`\eta \ge 0` cedes a proportion :math:`1 - \alpha` to a
    reinsurer charging loading :math:`\xi \ge 0`.

    The expected net drift (expected technical profit) is:

    .. math::
        \mathbb{E}[\Delta U(\alpha)] = [\eta - (1 - \alpha)\xi] \mu

    The variance of retained claims is:

    .. math::
        \text{Var}(\Delta U(\alpha)) = \alpha^2 \sigma^2

    Depending on the selected ``criterion``, the objective function evaluates:

    - ``"safety_index"``: Ratio of expected profit to variance (De Finetti's safety index):

      .. math::
          \theta(\alpha) = \frac{[\eta - (1 - \alpha)\xi] \mu}{\alpha^2 \sigma^2}
          = \frac{\mu}{\sigma^2} \left[ \frac{\eta - \xi}{\alpha^2} + \frac{\xi}{\alpha} \right]

      At zero retention (:math:`\alpha = 0`), retained variance is zero. The function evaluates
      the continuous limiting behavior :math:`\lim_{\alpha \to 0^+} \theta(\alpha)`:
      - If :math:`\eta > \xi`: The insurer eliminates all underwriting risk while locking in a
        strictly positive net premium margin :math:`(\eta - \xi)\mu > 0` (arbitrage/fronting
        margin), so :math:`\theta(0) = +\infty` (represented as ``np.inf``).
      - If :math:`\eta = \xi > 0`: Expected profit is :math:`\alpha \eta \mu` and variance is
        :math:`\alpha^2 \sigma^2`, so :math:`\theta(\alpha) = \frac{\eta \mu}{\alpha \sigma^2}
        \to +\infty` (represented as ``np.inf``).
      - If :math:`\eta = \xi = 0` (degenerate zero drift and zero loading): Both profit and
        variance are identically zero for all :math:`\alpha`, so :math:`\theta(0) = 0.0`.
      - If :math:`\xi > \eta`: The insurer pays more for reinsurance than it collects, incurring
        a guaranteed loss :math:`(\eta - \xi)\mu < 0` with zero variance, so
        :math:`\lim_{\alpha \to 0^+} \theta(\alpha) = -\infty` (represented as ``-np.inf``).

    - ``"mean_variance"``: Expected profit penalized by risk aversion parameter :math:`k > 0`:

      .. math::
          \Phi(\alpha) = [\eta - (1 - \alpha)\xi] \mu - \frac{k}{2} \alpha^2 \sigma^2

    - ``"expected_profit"``: Net expected drift :math:`\mathbb{E}[\Delta U(\alpha)]`.
    - ``"retained_variance"``: Retained risk variance :math:`\alpha^2 \sigma^2`.

    Args:
        retention_rate: Retained proportion :math:`\alpha \in [0.0, 1.0]`.
            Scalars and array-like inputs are accepted.
        expected_loss: Expected claim amount :math:`\mu = \mathbb{E}[X] > 0`.
            Scalars and array-like inputs are accepted.
        variance_loss: Variance of claim amount :math:`\sigma^2 = \text{Var}(X) > 0`.
            Scalars and array-like inputs are accepted.
        insurer_loading: Insurer relative safety loading :math:`\eta \ge 0`.
            Scalars and array-like inputs are accepted.
        reinsurer_loading: Reinsurer relative safety loading :math:`\xi \ge 0`.
            Scalars and array-like inputs are accepted.
        criterion: Objective criterion to compute. Must be one of ``"safety_index"``,
            ``"mean_variance"``, ``"expected_profit"``, or ``"retained_variance"``.
            Defaults to ``"safety_index"``.
        risk_aversion: Risk aversion coefficient :math:`k > 0`. Required if
            ``criterion="mean_variance"``.

    Returns:
        NDArray[np.float64]: Computed objective value. NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), or violate domain bounds
            (:math:`\alpha \notin [0, 1]`, :math:`\mu \le 0`, :math:`\sigma^2 \le 0`,
            :math:`\eta < 0`, :math:`\xi < 0`, :math:`k \le 0`), or if ``criterion`` is
            unrecognized.

    Notes:
        Validation status: experimental.

    References:
        De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
        IV Congresso dell'Unione Matematica Italiana, Sections 1–3.
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory (3rd ed.).
        Chapman and Hall, Chapter 6.
        Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer-Verlag.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.finetti_ruin import proportional_reinsurance_objective
        >>> obj = proportional_reinsurance_objective(
        ...     retention_rate=0.6,
        ...     expected_loss=100.0,
        ...     variance_loss=400.0,
        ...     insurer_loading=0.15,
        ...     reinsurer_loading=0.20,
        ...     criterion="safety_index",
        ... )
        >>> float(obj)
        0.04861111111111111
        >>> # Zero retention with eta > xi yields +inf (arbitrage limit)
        >>> float(proportional_reinsurance_objective(0.0, 1.0, 1.0, 0.20, 0.10))
        inf
    """
    alpha = np.asarray(retention_rate, dtype=np.float64)
    mu = np.asarray(expected_loss, dtype=np.float64)
    var = np.asarray(variance_loss, dtype=np.float64)
    eta = np.asarray(insurer_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)

    if not (
        np.all(np.isfinite(alpha))
        and np.all(np.isfinite(mu))
        and np.all(np.isfinite(var))
        and np.all(np.isfinite(eta))
        and np.all(np.isfinite(xi))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("retention_rate must be between 0.0 and 1.0 inclusive")
    if np.any(mu <= 0.0):
        raise ValueError("expected_loss must be strictly positive")
    if np.any(var <= 0.0):
        raise ValueError("variance_loss must be strictly positive")
    if np.any(eta < 0.0):
        raise ValueError("insurer_loading must be non-negative")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")

    expected_profit = (eta - (1.0 - alpha) * xi) * mu
    retained_var = (alpha**2) * var

    valid_criteria = {
        "safety_index",
        "mean_variance",
        "expected_profit",
        "retained_variance",
    }
    if criterion not in valid_criteria:
        raise ValueError(f"criterion must be one of {valid_criteria}, got {criterion!r}")

    if criterion == "expected_profit":
        return np.asarray(expected_profit, dtype=np.float64)

    if criterion == "retained_variance":
        return np.asarray(retained_var, dtype=np.float64)

    if criterion == "mean_variance":
        if risk_aversion is None:
            raise ValueError(
                "risk_aversion parameter must be provided when criterion='mean_variance'"
            )
        k = np.asarray(risk_aversion, dtype=np.float64)
        if not np.all(np.isfinite(k)):
            raise ValueError("risk_aversion must contain only finite numbers")
        if np.any(k <= 0.0):
            raise ValueError("risk_aversion must be strictly positive")
        res = expected_profit - 0.5 * k * retained_var
        return np.asarray(res, dtype=np.float64)

    # criterion == "safety_index"
    # When alpha == 0.0, retained variance is zero. We evaluate the continuous limit
    # lim_{alpha -> 0+} theta(alpha):
    # - If eta > xi: positive arbitrage drift / 0^+ -> +inf
    # - If eta == xi and eta > 0: eta*mu / (alpha*var) -> +inf
    # - If eta == xi == 0: zero drift / zero var -> 0.0 (degenerate null drift)
    # - If xi > eta: negative drift / 0^+ -> -inf
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = expected_profit / retained_var
        zero_limit = np.where(
            eta > xi,
            np.inf,
            np.where(
                eta == xi,
                np.where(eta > 0.0, np.inf, 0.0),
                -np.inf,
            ),
        )
        safety_idx = np.where(alpha == 0.0, zero_limit, raw_ratio)
    return np.asarray(safety_idx, dtype=np.float64)


def de_finetti_optimal_retention(
    expected_loss: float | ArrayLike,
    variance_loss: float | ArrayLike,
    insurer_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
    criterion: Literal["safety_index", "mean_variance"] = "safety_index",
    risk_aversion: float | ArrayLike | None = None,
) -> NDArray[np.float64]:
    r"""Compute De Finetti's optimal proportional retention rate.

    Under De Finetti's safety index criterion, the optimal retention proportion
    :math:`\alpha^* \in [0.0, 1.0]` maximizes the safety index :math:`\theta(\alpha)`
    over the compact domain :math:`[0.0, 1.0]`.

    The first-order condition on the interior :math:`\alpha \in (0, 1)` is:

    .. math::
        \frac{d\theta}{d\alpha} = \frac{\mu}{\sigma^2 \alpha^3} [2(\xi - \eta) - \xi \alpha] = 0
        \implies \alpha_0 = 2\left(1 - \frac{\eta}{\xi}\right)

    Evaluating global behavior and boundary values :math:`\alpha \in \{0, 1\}` yields
    three distinct economic regimes:

    1. **Cheap Reinsurance / Arbitrage (:math:`\xi \le \eta`):**
       As :math:`\alpha \to 0^+`, :math:`\theta(\alpha) \to +\infty`. The direct insurer
       eliminates all variance while earning positive or zero-risk drift, so the supremum
       is attained at the boundary :math:`\alpha^* = 0.0`, with :math:`\theta(0) = +\infty`.
       Since :math:`\theta(\alpha) < +\infty` for all :math:`\alpha > 0`, :math:`\alpha^* = 0.0`
       globally maximizes the criterion.

    2. **Moderate Reinsurance Cost (:math:`\eta < \xi < 2\eta`):**
       The stationary point :math:`\alpha^* = 2(1 - \eta/\xi)` lies strictly inside :math:`(0, 1)`.
       The second derivative is strictly negative:
       :math:`\theta''(\alpha^*) = -\frac{\mu \xi^4}{8 \sigma^2 (\xi - \eta)^3} < 0`.
       At the boundaries: :math:`\lim_{\alpha \to 0^+} \theta(\alpha) = -\infty`, while at full
       retention :math:`\theta(1) = \frac{\eta \mu}{\sigma^2} < \theta(\alpha^*)`.
       The interior point :math:`\alpha^*` is the unique global maximum.

    3. **Expensive Reinsurance (:math:`\xi \ge 2\eta`):**
       The stationary point satisfies :math:`\alpha_0 \ge 1.0`. The derivative
       :math:`\frac{d\theta}{d\alpha} > 0` for all :math:`\alpha \in (0, 1]`, making
       :math:`\theta(\alpha)` strictly increasing. At the left boundary,
       :math:`\theta(0) = -\infty`. The global maximum is attained at the upper boundary
       :math:`\alpha^* = 1.0` (full retention).

    In summary:

    .. math::
        \alpha^* = \begin{cases}
            0.0 & \text{if } \xi \le \eta \quad (\text{arbitrage / full reinsurance}) \\
            2\left(1 - \frac{\eta}{\xi}\right) & \text{if } \eta < \xi < 2\eta \quad
            (\text{interior optimum}) \\
            1.0 & \text{if } \xi \ge 2\eta \quad (\text{expensive reinsurance / full retention})
        \end{cases}

    Under the mean-variance criterion with risk aversion :math:`k > 0`, the optimal
    retention equates marginal profit to marginal risk:

    .. math::
        \alpha^* = \text{clip}\left(\frac{\xi \mu}{k \sigma^2}, 0.0, 1.0\right)

    Args:
        expected_loss: Expected claim amount :math:`\mu = \mathbb{E}[X] > 0`.
            Scalars and array-like inputs are accepted.
        variance_loss: Variance of claim amount :math:`\sigma^2 = \text{Var}(X) > 0`.
            Scalars and array-like inputs are accepted.
        insurer_loading: Insurer relative safety loading :math:`\eta \ge 0`.
            Scalars and array-like inputs are accepted.
        reinsurer_loading: Reinsurer relative safety loading :math:`\xi \ge 0`.
            Scalars and array-like inputs are accepted.
        criterion: Criterion for optimization. Must be either ``"safety_index"``
            or ``"mean_variance"``. Defaults to ``"safety_index"``.
        risk_aversion: Risk aversion coefficient :math:`k > 0`. Required if
            ``criterion="mean_variance"``.

    Returns:
        NDArray[np.float64]: Optimal retention proportion :math:`\alpha^* \in [0.0, 1.0]`.
        NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf) or violate domain bounds
            or if ``criterion`` is invalid.

    Notes:
        Validation status: experimental.

    References:
        De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
        IV Congresso dell'Unione Matematica Italiana.
        Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory.
        Chapman and Hall, Section 6.4.

    Examples:
        >>> from therisktheory.finetti_ruin import de_finetti_optimal_retention
        >>> opt = de_finetti_optimal_retention(
        ...     expected_loss=100.0,
        ...     variance_loss=500.0,
        ...     insurer_loading=0.15,
        ...     reinsurer_loading=0.20,
        ...     criterion="safety_index",
        ... )
        >>> float(opt)
        0.5
        >>> # Professor's case: mu=1, var=1, eta=0.2, xi=0.1 -> alpha* = 0.0
        >>> float(de_finetti_optimal_retention(1.0, 1.0, 0.20, 0.10))
        0.0
    """
    mu = np.asarray(expected_loss, dtype=np.float64)
    var = np.asarray(variance_loss, dtype=np.float64)
    eta = np.asarray(insurer_loading, dtype=np.float64)
    xi = np.asarray(reinsurer_loading, dtype=np.float64)

    if not (
        np.all(np.isfinite(mu))
        and np.all(np.isfinite(var))
        and np.all(np.isfinite(eta))
        and np.all(np.isfinite(xi))
    ):
        raise ValueError("inputs must contain only finite numbers (NaN and Inf not permitted)")

    if np.any(mu <= 0.0):
        raise ValueError("expected_loss must be strictly positive")
    if np.any(var <= 0.0):
        raise ValueError("variance_loss must be strictly positive")
    if np.any(eta < 0.0):
        raise ValueError("insurer_loading must be non-negative")
    if np.any(xi < 0.0):
        raise ValueError("reinsurer_loading must be non-negative")

    if criterion == "safety_index":
        with np.errstate(divide="ignore", invalid="ignore"):
            # When xi == 0: if eta > 0, reinsurance is free, full cession alpha*=0.0 is optimal.
            # If xi <= eta, theta(0) -> +inf, so alpha* = 0.0 achieves global supremum.
            # When xi > eta: stationary point is 2 * (1 - eta / xi).
            # Clipped to [0.0, 1.0], giving alpha*=1.0 when xi >= 2 * eta.
            ratio = np.where(xi == 0.0, np.inf, eta / np.where(xi == 0.0, 1.0, xi))
            alpha_star = np.where(
                xi <= eta,
                0.0,
                np.clip(2.0 * (1.0 - ratio), 0.0, 1.0),
            )
        return np.asarray(alpha_star, dtype=np.float64)

    if criterion == "mean_variance":
        if risk_aversion is None:
            raise ValueError(
                "risk_aversion parameter must be provided when criterion='mean_variance'"
            )
        k = np.asarray(risk_aversion, dtype=np.float64)
        if not np.all(np.isfinite(k)):
            raise ValueError("risk_aversion must contain only finite numbers")
        if np.any(k <= 0.0):
            raise ValueError("risk_aversion must be strictly positive")
        alpha_star = (xi * mu) / (k * var)
        return np.asarray(np.clip(alpha_star, 0.0, 1.0), dtype=np.float64)

    raise ValueError(f"criterion must be 'safety_index' or 'mean_variance', got {criterion!r}")


def de_finetti_retention_ruin_approximation(
    initial_capital: float | ArrayLike,
    retention_rate: float | ArrayLike,
    expected_loss: float | ArrayLike,
    variance_loss: float | ArrayLike,
    insurer_loading: float | ArrayLike,
    reinsurer_loading: float | ArrayLike,
) -> NDArray[np.float64]:
    r"""Compute the diffusion approximation to ruin probability under proportional retention.

    Following Bruno de Finetti (1940) and diffusion approximations in classical ruin theory,
    the surplus process :math:`U(t)` is approximated by a continuous-time Brownian motion
    with drift :math:`m(\alpha) = \mathbb{E}[\Delta U(\alpha)]` and diffusion variance
    :math:`s^2(\alpha) = \text{Var}(\Delta U(\alpha))`.

    For an initial capital reserve :math:`u \ge 0`, the first-passage ruin probability of this
    diffusion proxy is given analytically by:

    .. math::
        \psi_{\text{diff}}(u; \alpha) = \exp\left(- 2 \theta(\alpha) u\right)

    where :math:`\theta(\alpha) = \frac{m(\alpha)}{s^2(\alpha)}` is De Finetti's safety index,
    computed by :func:`proportional_reinsurance_objective`.

    **Important Distinction: Diffusion Approximation vs. Proven Lundberg Bound**:
    This expression is a **diffusion (heavy-traffic) approximation**, NOT a proven analytical
    upper bound such as the classical Lundberg inequality :math:`\psi(u) \le \exp(-R u)`
    (where :math:`R > 0` is the Cramér-Lundberg adjustment coefficient satisfying
    :math:`\mathbb{E}[e^{R(X - c)}] = 1`). For general compound Poisson processes with
    non-exponential claims, :math:`\psi_{\text{diff}}(u; \alpha)` can either underestimate or
    overestimate the exact ruin probability :math:`\psi(u)`.

    **Degenerate Boundary Cases (Zero Reserve and Zero Drift)**:
    - **Zero Initial Reserve (:math:`u = 0`)**: The initial surplus begins exactly on the ruin
      barrier :math:`U(0) = 0`. Under both discrete and continuous ruin formulations, ruin occurs
      instantaneously with certainty: :math:`\psi_{\text{diff}}(0) = 1.0`, regardless of drift
      rate :math:`\theta`.
    - **Zero Drift (:math:`\theta = 0.0`)**: When expected net premium equals expected retained
      claims (:math:`\mathbb{E}[\Delta U] = 0`), the surplus is a pure martingale. An unbiased
      Brownian motion on :math:`[0, \infty)` is recurrent and hits zero with probability 1.0
      almost surely, yielding :math:`\psi_{\text{diff}}(u) = 1.0` for all :math:`u \ge 0`.
    - **Zero Reserve and Zero Drift (:math:`u = 0, \theta = 0`)**: Both degenerate conditions
      coincide, giving :math:`\psi_{\text{diff}}(0; 0) = 1.0`.
    - **Negative Drift (:math:`\theta < 0.0`)**: Expected drift is negative; ruin is almost
      surely certain: :math:`\psi_{\text{diff}} = 1.0`.
    - **Infinite Safety Index (:math:`\theta = +\infty`)**: When :math:`\alpha = 0` with
      :math:`\eta \ge \xi > 0` (arbitrage fronting with zero variance):
      For :math:`u > 0`, :math:`\psi_{\text{diff}} = \exp(-\infty) = 0.0`.
      For :math:`u = 0`, the barrier boundary applies: :math:`\psi_{\text{diff}}(0) = 1.0`.

    Args:
        initial_capital: Initial reserve capital :math:`u \ge 0`.
            Scalars and array-like inputs are accepted.
        retention_rate: Retention proportion :math:`\alpha \in [0.0, 1.0]`.
            Scalars and array-like inputs are accepted.
        expected_loss: Expected claim amount :math:`\mu = \mathbb{E}[X] > 0`.
            Scalars and array-like inputs are accepted.
        variance_loss: Variance of claim amount :math:`\sigma^2 = \text{Var}(X) > 0`.
            Scalars and array-like inputs are accepted.
        insurer_loading: Insurer relative safety loading :math:`\eta \ge 0`.
            Scalars and array-like inputs are accepted.
        reinsurer_loading: Reinsurer relative safety loading :math:`\xi \ge 0`.
            Scalars and array-like inputs are accepted.

    Returns:
        NDArray[np.float64]: Diffusion approximation to ruin probability in :math:`[0.0, 1.0]`.
        NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf) or if ``initial_capital < 0``.

    Notes:
        Validation status: experimental.

    References:
        De Finetti, B. (1940). Il rischio nel giuoco e nelle assicurazioni.
        IV Congresso dell'Unione Matematica Italiana.
        Asmussen, S., & Albrecher, H. (2010). Ruin Probabilities (2nd ed.).
        World Scientific, Chapter 4 (Diffusion Approximations).
        Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer-Verlag.

    Examples:
        >>> from therisktheory.finetti_ruin import de_finetti_retention_ruin_approximation
        >>> prob = de_finetti_retention_ruin_approximation(
        ...     initial_capital=50.0,
        ...     retention_rate=0.5,
        ...     expected_loss=100.0,
        ...     variance_loss=400.0,
        ...     insurer_loading=0.15,
        ...     reinsurer_loading=0.20,
        ... )
        >>> float(prob) < 1.0
        True
        >>> # Degenerate zero reserve yields ruin certainty
        >>> float(de_finetti_retention_ruin_approximation(0.0, 0.5, 100.0, 400.0, 0.15, 0.20))
        1.0
    """
    u = np.asarray(initial_capital, dtype=np.float64)
    if not np.all(np.isfinite(u)):
        raise ValueError("initial_capital must contain only finite numbers")
    if np.any(u < 0.0):
        raise ValueError("initial_capital must be non-negative")

    theta = proportional_reinsurance_objective(
        retention_rate=retention_rate,
        expected_loss=expected_loss,
        variance_loss=variance_loss,
        insurer_loading=insurer_loading,
        reinsurer_loading=reinsurer_loading,
        criterion="safety_index",
    )

    # Evaluate diffusion approximation:
    # 1. theta <= 0.0 or u == 0.0 -> ruin is certain: psi = 1.0
    # 2. theta == +inf and u > 0 -> exp(-inf) = 0.0
    # 3. 0 < theta < +inf and u > 0 -> exp(-2.0 * theta * u)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_exponent = np.where((theta <= 0.0) | np.isposinf(theta), 0.0, -2.0 * theta * u)
        normal_approx = np.exp(safe_exponent)
        approx = np.where(
            (theta <= 0.0) | (u == 0.0),
            1.0,
            np.where(np.isposinf(theta), 0.0, normal_approx),
        )

    return np.asarray(np.clip(approx, 0.0, 1.0), dtype=np.float64)


# Backward-compatible alias retained with explicit documentation:
de_finetti_retention_ruin_bound = de_finetti_retention_ruin_approximation
