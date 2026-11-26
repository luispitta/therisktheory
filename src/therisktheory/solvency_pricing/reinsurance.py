"""Reinsurance treaties and cedent solvency evaluation functions.

This module provides human-validated, vectorized implementations for proportional
and non-proportional reinsurance treaties, as well as actuarial solvency metrics
following risk transfer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quota_share(
    losses: ArrayLike,
    retention_rate: float | ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute retained and ceded losses under a proportional quota share treaty.

    In a quota share arrangement, the direct insurer (cedent) retains a fixed
    proportion :math:`\\alpha \\in [0, 1]` of every claim, and cedes the remaining
    proportion :math:`1 - \\alpha` to the reinsurer:

    .. math::
        X_{\\text{ret}} = \\alpha X, \\quad X_{\\text{ced}} = (1 - \\alpha) X

    Args:
        losses: Gross claim amounts :math:`X`. Must be non-negative.
            Scalars and array-like inputs are accepted.
        retention_rate: Retention proportion :math:`\\alpha \\in [0, 1]`.
            Scalars and array-like inputs are accepted.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            A tuple ``(retained_losses, ceded_losses)`` of element-wise retained
            and ceded losses. NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), or if any loss
            value is negative, or if retention rate is not in [0.0, 1.0].

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., Goovaerts, M., Dhaene, J., & Denuit, M. (2008). Modern Actuarial
        Risk Theory: Using R (2nd ed.). Springer, Chapter 1.
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Society
        of Actuaries, Chapter 14.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.solvency_pricing import quota_share
        >>> claims = np.array([1000.0, 5000.0, 10000.0])
        >>> retained, ceded = quota_share(claims, retention_rate=0.7)
        >>> retained
        array([ 700., 3500., 7000.])
        >>> ceded
        array([ 300., 1500., 3000.])
    """
    loss_arr = np.asarray(losses, dtype=np.float64)
    alpha_arr = np.asarray(retention_rate, dtype=np.float64)

    if not (np.all(np.isfinite(loss_arr)) and np.all(np.isfinite(alpha_arr))):
        raise ValueError("losses and retention_rate must contain only finite numbers")
    if np.any(loss_arr < 0.0):
        raise ValueError("losses must be non-negative")
    if np.any((alpha_arr < 0.0) | (alpha_arr > 1.0)):
        raise ValueError("retention_rate must be between 0.0 and 1.0 inclusive")

    retained = np.asarray(loss_arr * alpha_arr, dtype=np.float64)
    ceded = np.asarray(loss_arr * (1.0 - alpha_arr), dtype=np.float64)
    return retained, ceded


def surplus_share(
    losses: ArrayLike,
    sum_insured: ArrayLike,
    retention_line: float | ArrayLike,
    max_lines: float | ArrayLike = np.inf,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute retained and ceded losses under a proportional surplus share treaty.

    In a surplus share treaty, the insurer defines a monetary retention line :math:`L`.
    For a policy with sum insured :math:`S`, if :math:`S \\le L`, the insurer retains 100%
    of the risk. If :math:`S > L`, the retention proportion is :math:`\\alpha = L / S`,
    and the reinsurer covers up to :math:`m` lines (maximum cession :math:`m L`):

    .. math::
        \\alpha = \\min\\left(1, \\frac{L}{S}\\right)
    .. math::
        \\beta = \\min\\left(1 - \\alpha, \\frac{m L}{S}\\right)
    .. math::
        X_{\\text{ret}} = (1 - \\beta) X, \\quad X_{\\text{ced}} = \\beta X

    Args:
        losses: Gross claim amounts :math:`X`. Must be non-negative.
            Scalars and array-like inputs are accepted.
        sum_insured: Total sum insured :math:`S` of each policy. Must be strictly positive.
            Scalars and array-like inputs are accepted.
        retention_line: Insurer monetary retention limit :math:`L`. Must be strictly positive.
            Scalars and array-like inputs are accepted.
        max_lines: Maximum treaty capacity in multiples of the retention line (:math:`m`).
            Defaults to infinity (unlimited surplus capacity). Must be non-negative.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            A tuple ``(retained_losses, ceded_losses)``. NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), or if losses are negative,
            sum_insured <= 0, retention_line <= 0, or max_lines < 0.

    Notes:
        Validation status: experimental.

    References:
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Society
        of Actuaries, Chapter 14.
        Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2019). Loss Models: From
        Data to Decisions (5th ed.). Wiley.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.solvency_pricing import surplus_share
        >>> claims = np.array([2000.0, 5000.0])
        >>> sums_insured = np.array([10000.0, 40000.0])
        >>> retained, ceded = surplus_share(
        ...     claims, sums_insured, retention_line=10000.0, max_lines=3
        ... )
        >>> retained
        array([2000., 1250.])
        >>> ceded
        array([   0., 3750.])
    """
    loss_arr = np.asarray(losses, dtype=np.float64)
    si_arr = np.asarray(sum_insured, dtype=np.float64)
    line_arr = np.asarray(retention_line, dtype=np.float64)
    m_arr = np.asarray(max_lines, dtype=np.float64)

    if not (
        np.all(np.isfinite(loss_arr))
        and np.all(np.isfinite(si_arr))
        and np.all(np.isfinite(line_arr))
    ):
        raise ValueError("losses, sum_insured, and retention_line must be finite numbers")
    if np.any(loss_arr < 0.0):
        raise ValueError("losses must be non-negative")
    if np.any(si_arr <= 0.0):
        raise ValueError("sum_insured must be strictly positive")
    if np.any(line_arr <= 0.0):
        raise ValueError("retention_line must be strictly positive")
    if np.any(m_arr < 0.0):
        raise ValueError("max_lines must be non-negative")

    # Retention proportion alpha
    alpha = np.minimum(1.0, line_arr / si_arr)

    # Ceded proportion beta bounded by available treaty capacity
    capacity_ratio = (m_arr * line_arr) / si_arr
    beta = np.minimum(1.0 - alpha, capacity_ratio)

    retained = np.asarray(loss_arr * (1.0 - beta), dtype=np.float64)
    ceded = np.asarray(loss_arr * beta, dtype=np.float64)
    return retained, ceded


def excess_of_loss(
    losses: ArrayLike,
    priority: float | ArrayLike,
    limit: float | ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute retained and ceded losses under an Excess of Loss (XL) treaty.

    Under an Excess of Loss arrangement, the cedent pays each loss up to a retention
    priority :math:`d`. The reinsurer indemnifies the portion above :math:`d` up to
    an optional layer cover limit :math:`l`:

    .. math::
        X_{\\text{ced}} = \\min\\left(\\max(0, X - d), l\\right)
    .. math::
        X_{\\text{ret}} = X - X_{\\text{ced}}

    Args:
        losses: Individual claim amounts :math:`X`. Must be non-negative.
            Scalars and array-like inputs are accepted.
        priority: Cedent deductible or retention attachment point :math:`d`.
            Must be non-negative. Scalars and array-like inputs are accepted.
        limit: Layer cover capacity :math:`l`. If None, cover is unlimited above
            the priority. If specified, must be strictly positive.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            A tuple ``(retained_losses, ceded_losses)``. NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers, or if losses or priority
            are negative, or if limit <= 0.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Chapter 1.
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Chapter 14.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.solvency_pricing import excess_of_loss
        >>> claims = np.array([300.0, 1200.0, 2500.0])
        >>> retained, ceded = excess_of_loss(claims, priority=500.0, limit=1000.0)
        >>> retained
        array([ 300.,  500., 1500.])
        >>> ceded
        array([   0.,  700., 1000.])
    """
    loss_arr = np.asarray(losses, dtype=np.float64)
    p_arr = np.asarray(priority, dtype=np.float64)

    if not (np.all(np.isfinite(loss_arr)) and np.all(np.isfinite(p_arr))):
        raise ValueError("losses and priority must contain only finite numbers")
    if np.any(loss_arr < 0.0):
        raise ValueError("losses must be non-negative")
    if np.any(p_arr < 0.0):
        raise ValueError("priority must be non-negative")

    excess = np.maximum(0.0, loss_arr - p_arr)

    if limit is not None:
        lim_arr = np.asarray(limit, dtype=np.float64)
        if not np.all(np.isfinite(lim_arr)):
            raise ValueError("limit must contain only finite numbers")
        if np.any(lim_arr <= 0.0):
            raise ValueError("limit must be strictly positive")
        ceded = np.minimum(excess, lim_arr)
    else:
        ceded = excess

    retained = np.asarray(loss_arr - ceded, dtype=np.float64)
    return retained, np.asarray(ceded, dtype=np.float64)


def stop_loss(
    aggregate_losses: ArrayLike,
    deductible: float | ArrayLike,
    limit: float | ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute retained and ceded losses under an aggregate Stop-Loss treaty.

    Stop-loss reinsurance protects the total aggregate portfolio losses :math:`S`
    over an operational period. The reinsurer pays the aggregate loss exceeding
    deductible :math:`D`, up to an optional capacity limit :math:`L_{\\text{cap}}`:

    .. math::
        S_{\\text{ced}} = \\min\\left(\\max(0, S - D), L_{\\text{cap}}\\right)
    .. math::
        S_{\\text{ret}} = S - S_{\\text{ced}}

    Args:
        aggregate_losses: Total portfolio losses :math:`S` in a period. Must be non-negative.
            Scalars and array-like inputs are accepted.
        deductible: Aggregate attachment point :math:`D`. Must be non-negative.
            Scalars and array-like inputs are accepted.
        limit: Maximum aggregate cover capacity :math:`L_{\\text{cap}}`.
            If None, cover is unlimited above deductible. If specified, must be strictly positive.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            A tuple ``(retained_losses, ceded_losses)``. NumPy broadcasting rules apply.

    Raises:
        ValueError: If inputs contain non-finite numbers, or if aggregate losses
            or deductible are negative, or if limit <= 0.

    Notes:
        Validation status: experimental.

    References:
        Kaas, R., et al. (2008). Modern Actuarial Risk Theory (2nd ed.). Springer, Section 1.3.
        Rolski, T., Schmidli, H., Schmidt, V., & Teugels, J. (1999). Stochastic Processes
        for Insurance and Finance. Wiley.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.solvency_pricing import stop_loss
        >>> portfolios = np.array([40000.0, 75000.0, 120000.0])
        >>> retained, ceded = stop_loss(portfolios, deductible=50000.0, limit=50000.0)
        >>> retained
        array([40000., 50000., 70000.])
        >>> ceded
        array([    0., 25000., 50000.])
    """
    s_arr = np.asarray(aggregate_losses, dtype=np.float64)
    d_arr = np.asarray(deductible, dtype=np.float64)

    if not (np.all(np.isfinite(s_arr)) and np.all(np.isfinite(d_arr))):
        raise ValueError("aggregate_losses and deductible must contain only finite numbers")
    if np.any(s_arr < 0.0):
        raise ValueError("aggregate_losses must be non-negative")
    if np.any(d_arr < 0.0):
        raise ValueError("deductible must be non-negative")

    excess = np.maximum(0.0, s_arr - d_arr)

    if limit is not None:
        lim_arr = np.asarray(limit, dtype=np.float64)
        if not np.all(np.isfinite(lim_arr)):
            raise ValueError("limit must contain only finite numbers")
        if np.any(lim_arr <= 0.0):
            raise ValueError("limit must be strictly positive")
        ceded = np.minimum(excess, lim_arr)
    else:
        ceded = excess

    retained = np.asarray(s_arr - ceded, dtype=np.float64)
    return retained, np.asarray(ceded, dtype=np.float64)


def reinsured_solvent_position(
    initial_capital: float | ArrayLike,
    retained_premium: float | ArrayLike,
    retained_losses: ArrayLike,
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Evaluate the net solvent position, solvency status, and deficit of the cedent.

    Calculates the financial terminal surplus of the insurer after taking into account
    initial risk reserves :math:`u`, net retained premium income :math:`P_{\\text{ret}}`,
    and net retained claim losses :math:`S_{\\text{ret}}`:

    .. math::
        U_{\\text{net}} = u + P_{\\text{ret}} - S_{\\text{ret}}

    The insurer is considered solvent if :math:`U_{\\text{net}} \\ge 0`. In case of
    insolvency, the deficit magnitude is :math:`|U_{\\text{net}}| = -U_{\\text{net}}`.

    Args:
        initial_capital: Initial solvency reserves :math:`u \\ge 0`.
            Scalars and array-like inputs are accepted.
        retained_premium: Net premium volume :math:`P_{\\text{ret}} \\ge 0`.
            Scalars and array-like inputs are accepted.
        retained_losses: Net losses incurred by the cedent :math:`S_{\\text{ret}} \\ge 0`.
            Scalars and array-like inputs are accepted.

    Returns:
        dict[str, NDArray[np.float64] | NDArray[np.bool_]]: A dictionary containing:
            - ``"net_surplus"`` (NDArray[np.float64]): Terminal capital :math:`U_{\\text{net}}`.
            - ``"is_solvent"`` (NDArray[np.bool_]): Boolean array indicating solvency.
            - ``"deficit"`` (NDArray[np.float64]): Shortfall when insolvent (0.0 otherwise).

    Raises:
        ValueError: If inputs contain non-finite numbers (NaN/Inf), or if initial capital,
            retained premium, or retained losses are negative.

    Notes:
        Validation status: experimental.

    References:
        Bowers, N. L., et al. (1997). Actuarial Mathematics (2nd ed.). Chapter 14.
        Dickson, D. C. (2016). Insurance Risk and Ruin (2nd ed.). Cambridge University Press.

    Examples:
        >>> import numpy as np
        >>> from therisktheory.solvency_pricing import reinsured_solvent_position
        >>> pos = reinsured_solvent_position(
        ...     initial_capital=1000.0,
        ...     retained_premium=500.0,
        ...     retained_losses=[1200.0, 1800.0],
        ... )
        >>> pos["net_surplus"]
        array([ 300., -300.])
        >>> pos["is_solvent"]
        array([ True, False])
        >>> pos["deficit"]
        array([  0., 300.])
    """
    u_arr = np.asarray(initial_capital, dtype=np.float64)
    p_arr = np.asarray(retained_premium, dtype=np.float64)
    l_arr = np.asarray(retained_losses, dtype=np.float64)

    if not (
        np.all(np.isfinite(u_arr)) and np.all(np.isfinite(p_arr)) and np.all(np.isfinite(l_arr))
    ):
        raise ValueError("inputs must contain only finite numbers")
    if np.any(u_arr < 0.0):
        raise ValueError("initial_capital must be non-negative")
    if np.any(p_arr < 0.0):
        raise ValueError("retained_premium must be non-negative")
    if np.any(l_arr < 0.0):
        raise ValueError("retained_losses must be non-negative")

    net_surplus = np.asarray(u_arr + p_arr - l_arr, dtype=np.float64)
    is_solvent = np.asarray(net_surplus >= 0.0, dtype=np.bool_)
    deficit = np.asarray(np.maximum(0.0, -net_surplus), dtype=np.float64)

    return {
        "net_surplus": net_surplus,
        "is_solvent": is_solvent,
        "deficit": deficit,
    }
