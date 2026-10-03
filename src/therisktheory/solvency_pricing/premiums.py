"""Premium principles and basic pricing functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pure_premium(
    frequency: ArrayLike,
    mean_severity: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the pure premium under the frequency-severity decomposition.

    The pure premium is the expected aggregate loss:

        P = E[N] E[X]

    under the standard assumption that claim count ``N`` and claim severities
    are independent and identically distributed in the compound-sum model.

    Parameters
    ----------
    frequency
        Expected claim frequency ``E[N]``. Scalars and array-like inputs are
        accepted.
    mean_severity
        Expected claim severity ``E[X]``. Scalars and array-like inputs are
        accepted.

    Returns
    -------
    numpy.ndarray
        Element-wise pure premium. NumPy broadcasting rules apply.

    Raises
    ------
    ValueError
        If any frequency or mean severity value is negative.

    Notes
    -----
    Validation status: experimental.

    This function is included as the reference contribution pattern for the
    repository. Its mathematical source must be recorded before promotion to
    ``mathematically_validated``.
    """
    freq = np.asarray(frequency, dtype=np.float64)
    sev = np.asarray(mean_severity, dtype=np.float64)

    if np.any(freq < 0):
        raise ValueError("frequency must be non-negative")
    if np.any(sev < 0):
        raise ValueError("mean_severity must be non-negative")

    return np.asarray(freq * sev, dtype=np.float64)
