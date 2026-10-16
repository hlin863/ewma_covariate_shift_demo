from .park_mean_cusum import (
    mean_cusum_curves,
    check_cusum_unimodality,
    evaluate_mean_change,
)

import numpy as np

def calculate_cpss_score(
    maximum_values,
    average_values,
    b_max_values,
    b_avg_values,
    *,
    T: int,
    gamma: float = 1.0,
    require_peak_agreement: bool = True,
) -> float:
    """Calculate the proposed Calibrated Peak Stability Score.

    Parameters
    ----------
    maximum_values : array-like
        A_r^max: unimodality-violation counts for B runs.
    average_values : array-like
        A_r^avg: unimodality-violation counts for B runs.
    b_max_values : array-like
        Peak locations for the maximum CUSUM curves.
    b_avg_values : array-like
        Peak locations for the average CUSUM curves.
    T : int
        Number of observations in each run.
    gamma : float
        Non-negative exponential distance penalty.
    require_peak_agreement : bool
        Whether to enforce Park's peak-distance criterion.

    Returns
    -------
    float
        CPSS in [0, 1].

    Notes
    -----
    A zero product means at least ONE unimodality
    count is zero, following Park et al. (2023).

    This is a proposed score, not yet statistically
    calibrated against a target false-alarm rate.
    """
    a_max = np.asarray(maximum_values, dtype=float)
    a_avg = np.asarray(average_values, dtype=float)
    b_max = np.asarray(b_max_values, dtype=float)
    b_avg = np.asarray(b_avg_values, dtype=float)

    arrays = (a_max, a_avg, b_max, b_avg)

    if any(a.ndim != 1 for a in arrays):
        raise ValueError("All inputs must be 1D arrays.")

    B = len(a_max)

    if B == 0 or any(len(a) != B for a in arrays):
        raise ValueError(
            "All inputs must contain the same nonzero "
            "number B of perturbation results."
        )

    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("All inputs must be finite.")

    if not isinstance(T, (int, np.integer)) or T < 3:
        raise ValueError("T must be an integer >= 3.")

    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and >= 0.")

    if (
        np.any(a_max < 0)
        or np.any(a_avg < 0)
        or np.any(a_max != np.floor(a_max))
        or np.any(a_avg != np.floor(a_avg))
    ):
        raise ValueError(
            "Unimodality counts must be nonnegative integers."
        )

    if (
        np.any(b_max < 1)
        or np.any(b_avg < 1)
        or np.any(b_max >= T)
        or np.any(b_avg >= T)
        or np.any(b_max != np.floor(b_max))
        or np.any(b_avg != np.floor(b_avg))
    ):
        raise ValueError(
            "Peak locations must be integer split positions."
        )

    # Park's first condition:
    # A_max * A_avg == 0
    unimodality_passed = (a_max * a_avg == 0)

    # Normalised absolute peak separation
    gap = np.abs(b_max - b_avg)
    d = gap / (T - 1)

    # Park's second condition
    if require_peak_agreement:
        agreement_passed = gap < np.sqrt(T) / 2
    else:
        agreement_passed = np.ones(B, dtype=bool)

    valid = unimodality_passed & agreement_passed

    # Exponential distance penalty
    weights = np.exp(-gamma * d)

    # CPSS contributions and final score
    contributions = valid.astype(float) * weights

    S_cpss = float(np.mean(contributions))

    return S_cpss