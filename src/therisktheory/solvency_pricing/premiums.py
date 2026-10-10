"""Pure premium and the expected value premium principle.

Collective model notation: ``N`` is the number of claims, ``X_1, X_2, ...``
are the claim amounts and ``S = X_1 + ... + X_N`` is the aggregate claim
amount (``S = 0`` if ``N = 0``). The claim amounts are i.i.d. and independent
of ``N`` (Kaas et al., 2008, Section 3.1, p. 41).

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J. and Denuit, M. (2008). *Modern
    Actuarial Risk Theory: Using R*, 2nd ed. Springer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory._validation import as_nonnegative_finite_array


def expected_aggregate_loss(
    frequency: ArrayLike,
    mean_severity: ArrayLike,
) -> NDArray[np.float64]:
    """Expected aggregate claims ``E[S] = E[N] * E[X]``.

    Follows from conditioning on ``N``, see Kaas et al. (2008), eq. (3.3),
    p. 42. Only the means are needed, so any claim count and claim size
    distributions with finite mean can be used.

    Args:
        frequency: Expected number of claims ``E[N]``.
        mean_severity: Expected claim amount ``E[X]``.

    Returns:
        ``E[S]`` as a float64 array (0-d for scalar inputs). Inputs are
        broadcast with the usual NumPy rules.

    Raises:
        ValueError: If an input is negative, NaN or infinite.

    Note:
        Validation status: experimental.
    """
    freq = as_nonnegative_finite_array(frequency, "frequency")
    sev = as_nonnegative_finite_array(mean_severity, "mean_severity")
    return np.asarray(freq * sev, dtype=np.float64)


def pure_premium(
    frequency: ArrayLike,
    mean_severity: ArrayLike,
) -> NDArray[np.float64]:
    """Pure (net) premium ``P = E[S] = E[N] * E[X]``.

    This is the net premium or equivalence principle of Kaas et al. (2008),
    Section 5.3, p. 119. It has no safety loading, and a premium without a
    positive loading leads to ruin with certainty (Section 5.3.1, p. 120).

    Args:
        frequency: Expected number of claims ``E[N]``.
        mean_severity: Expected claim amount ``E[X]``.

    Returns:
        The pure premium as a float64 array (0-d for scalar inputs).

    Raises:
        ValueError: If an input is negative, NaN or infinite.

    Note:
        Validation status: experimental.
    """
    return expected_aggregate_loss(frequency, mean_severity)


def expected_value_premium(
    expected_loss: ArrayLike,
    loading: ArrayLike,
) -> NDArray[np.float64]:
    """Expected value principle ``(1 + theta) * E[S]``.

    Kaas et al. (2008), Section 5.3, principle (b), p. 119. Its properties are
    summarised in Table 5.1, p. 122. The book asks for ``theta > 0``; here
    ``theta = 0`` is also allowed and gives back the pure premium.

    Args:
        expected_loss: Expected aggregate claims ``E[S]``, e.g. the output of
            :func:`pure_premium`.
        loading: Relative safety loading ``theta``.

    Returns:
        The loaded premium as a float64 array (0-d for scalar inputs).

    Raises:
        ValueError: If an input is negative, NaN or infinite.

    Note:
        Validation status: experimental.
    """
    mean = as_nonnegative_finite_array(expected_loss, "expected_loss")
    theta = as_nonnegative_finite_array(loading, "loading")
    return np.asarray((1.0 + theta) * mean, dtype=np.float64)
