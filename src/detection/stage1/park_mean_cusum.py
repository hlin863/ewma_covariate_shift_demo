"""Park et al. (2023) Section 2.1: exploratory aggregated mean-CUSUM curves.

This implements the paper's channelwise normalized statistic and pointwise
maximum / mean aggregation and their candidate argmax locations only.
It does NOT implement the paper's subsequent unimodality, peak-agreement
confirmation procedure, and is not a significance test or online alarm.
"""
import numpy as np


def mean_cusum_curves(values: np.ndarray, *, min_segment: int = 8) -> dict:
    """Calculate Eq. (2.2) for positive-mean multichannel observations.

    Rows are ordered trial features / time observations; columns are channels.
    The paper divides each channel statistic by its time-series mean. Raw,
    mean-centred or signed EEG should not be entered without a justified
    transformation: zero/non-positive means are rejected here.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected observations by at least two variables.")
    n, p = data.shape
    if not isinstance(min_segment, int) or min_segment < 2 or 2 * min_segment >= n:
        raise ValueError("min_segment must be >=2 and leave two valid segments.")
    if not np.isfinite(data).all():
        raise ValueError("Mean-CUSUM inputs must be finite.")
    means = data.mean(axis=0)
    if np.any(means <= 1e-12):
        raise ValueError("Eq. (2.2) requires positive, nonzero channel means; use positive power features.")
    positions = np.arange(min_segment, n - min_segment + 1, dtype=int)
    prefix = np.vstack((np.zeros((1, p)), np.cumsum(data, axis=0)))
    before = prefix[positions]
    after = prefix[-1] - before
    left = np.sqrt((n - positions) / (n * positions))[:, None]
    right = np.sqrt(positions / (n * (n - positions)))[:, None]
    curves = np.abs(left * before - right * after) / means[None, :]
    maximum = curves.max(axis=1)
    average = curves.mean(axis=1)
    imax, iavg = int(maximum.argmax()), int(average.argmax())
    return {
        "positions": positions.tolist(),
        "by_channel": curves.T.round(7).tolist(),
        "maximum": maximum.round(7).tolist(),
        "average": average.round(7).tolist(),
        "b_max": int(positions[imax]),
        "b_avg": int(positions[iavg]),
        "agreement_gap": abs(int(positions[imax]) - int(positions[iavg])),
        "channels": p,
        "observations": n,
        "min_segment": min_segment,
    }
