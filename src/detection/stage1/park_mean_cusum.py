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
    
    maximum_smoothed = smooth_cusum_curve(positions, maximum, span=0.1, degree=1)
    average_smoothed = smooth_cusum_curve(positions, average, span=0.1, degree=1)

    imax_smoothed, iavg_smoothed = int(maximum_smoothed.argmax()), int(average_smoothed.argmax())
    
    return {
        "positions": positions.tolist(),
        "by_channel": curves.T.round(7).tolist(),

        # Original raw CUSUM outputs
        "maximum": maximum.round(7).tolist(),
        "average": average.round(7).tolist(),
        "b_max": int(positions[imax]),
        "b_avg": int(positions[iavg]),
        "agreement_gap": abs(int(positions[imax]) - int(positions[iavg])),
        
        # new smoothed CUSUM outputs
        "maximum_smoothed": maximum_smoothed.round(7).tolist(),
        "average_smoothed": average_smoothed.round(7).tolist(),
        "b_max_smoothed": int(positions[imax_smoothed]),
        "b_avg_smoothed": int(positions[iavg_smoothed]),
        "agreement_gap_smoothed": abs(int(positions[imax_smoothed]) - int(positions[iavg_smoothed])),

        # Explicit smoothing configuration
        "smoothing": {
            "span": 0.1,
            "degree": 1,
            "kernel": "tricube",
            "domain": "retained_candidate_positions",
        },

        "channels": p,
        "observations": n,
        "min_segment": min_segment,
    }


def smooth_cusum_curve(positions, values, *, span=0.1, degree=1):
    """LOESS-style local polynomial fit with tricube weights.

    Park specifies span=0.1. Degree and weighting kernel are
    explicit reconstruction choices, not verified author settings.
    """
    x = np.asarray(positions, dtype=float)
    y = np.asarray(values, dtype=float)

    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("positions and values must be matching 1D arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite.")
    if len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("Need at least three strictly increasing positions.")
    if not 0 < span <= 1:
        raise ValueError("span must be in (0, 1].")
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2.")
    if len(x) < degree + 2:
        raise ValueError("Too few observations for the selected degree.")

    neighbours = min(len(x), max(degree + 2, int(np.ceil(span * len(x)))))
    smoothed = np.empty_like(y)

    for i, centre in enumerate(x):
        distance = np.abs(x - centre)
        indices = np.argsort(distance, kind="stable")[:neighbours]

        # Include all points tied at the neighbourhood boundary.
        radius = distance[indices[-1]]
        indices = np.flatnonzero(distance <= radius)
        offsets = (x[indices] - centre) / radius

        weights = (1 - np.abs(offsets) ** 3) ** 3
        design = np.vander(offsets, N=degree + 1, increasing=True)
        root_weights = np.sqrt(weights)

        coefficients, _, rank, _ = np.linalg.lstsq(
            design * root_weights[:, None],
            y[indices] * root_weights,
            rcond=None,
        )
        if rank < degree + 1:
            raise ValueError("Local polynomial fit is rank deficient.")

        smoothed[i] = coefficients[0]

    return smoothed



def check_cusum_unimodality(
    positions, smoothed_curve, *, peak=None, observations=None, k=10
) -> dict:
    """Park et al. (2023), Algorithm 1 Step 3: A statistic on smoothed CUSUM.

    Form D1 from k equally spaced *interior* points between 0 and peak,
    and D2 between peak and T-1; map each target to the nearest retained
    CUSUM candidate. For k=10, implement the paper's boundary-exclusion
    version of (2.3): sum_{i=6..10} I[D1[i-1] > D1[i]] +
    sum_{j=1..5} I[D2[j] < D2[j+1]], using 1-based paper indices.

    Duplicate nearest candidate positions render the diagnostic ineligible:
    the paper does not specify a remedy when min_segment truncation makes
    the nominal sample points indistinguishable. This is a conservative
    repository-specific guard, not an additional Park criterion.
    """
    x = np.asarray(positions, dtype=int)
    y = np.asarray(smoothed_curve, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("positions and smoothed_curve must be nonempty matching 1D arrays")
    if np.any(np.diff(x) <= 0) or not np.isfinite(y).all():
        raise ValueError("positions must be increasing and curve finite")
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer >= 2")
    if observations is None:
        raise ValueError("observations T is required for original [0,T-1] domain")
    if not isinstance(observations, int) or observations < 3 or x[0] < 0 or x[-1] >= observations:
        raise ValueError("invalid observations or candidate domain")
    if peak is None:
        peak = int(x[int(np.argmax(y))])
    if int(peak) not in x:
        raise ValueError("peak must be a retained candidate position")
    if k != 10:
        raise ValueError("Algorithm 1's boundary-exclusion comparison is defined here for k=10")

    def sample(a, b):
        target = a + (b - a) * np.arange(1, k + 1) / (k + 1)
        chosen = np.abs(x[:, None] - target[None, :]).argmin(axis=0)
        return x[chosen], y[chosen]

    left_x, left_y = sample(0, peak)
    right_x, right_y = sample(peak, observations - 1)
    unique = len(np.unique(left_x)) == k and len(np.unique(right_x)) == k
    if not unique:
        return {
            "eligible": False, "A": None, "unimodal": None,
            "reason": "sampling points collapse on retained candidate positions",
            "peak": int(peak), "D1": left_x.tolist(), "D2": right_x.tolist(),
        }

    # For 1-based i=6..10 compare D1[i-1] > D1[i].
    left_violations = [int(i) for i in range(6, 11) if left_y[i - 2] > left_y[i - 1]]
    # For 1-based j=1..5 compare D2[j] < D2[j+1].
    right_violations = [int(j) for j in range(1, 6) if right_y[j - 1] < right_y[j]]
    A = len(left_violations) + len(right_violations)
    return {
        "eligible": True, "A": int(A), "unimodal": A == 0,
        "reason": None, "peak": int(peak),
        "D1": left_x.tolist(), "D2": right_x.tolist(),
        "left_violation_indices": left_violations,
        "right_violation_indices": right_violations,
    }


def evaluate_mean_change(curves: dict) -> dict:
    """Park Algorithm 1 Steps 3–4: unimodality and strict peak agreement.

    Accepts the result of mean_cusum_curves(). The result is a paper-rule
    *decision*, not a calibrated online alarm or a numerical reproduction.
    On a failed mean rule, the full Park algorithm proceeds to its
    second-order method; that method is not implemented here.
    """
    required = ("positions", "maximum_smoothed", "average_smoothed", "observations")
    if any(key not in curves for key in required):
        raise ValueError("curves must contain positions, smoothed aggregates and observations")
    x = curves["positions"]
    T = int(curves["observations"])
    maximum = check_cusum_unimodality(x, curves["maximum_smoothed"], observations=T)
    average = check_cusum_unimodality(x, curves["average_smoothed"], observations=T)
    if not maximum["eligible"] or not average["eligible"]:
        return {
            "status": "ineligible", "mean_change": None, "estimated_position": None,
            "maximum_check": maximum, "average_check": average,
            "reason": "insufficient distinct sampling positions for Algorithm 1",
        }
    b_max, b_avg = maximum["peak"], average["peak"]
    separation = abs(b_max - b_avg)
    limit = np.sqrt(T) / 2
    passed = (maximum["A"] * average["A"] == 0) and separation < limit
    return {
        "status": "mean_change_candidate" if passed else "no_mean_change_by_rule",
        "mean_change": bool(passed),
        "estimated_position": (b_max + b_avg) / 2 if passed else None,
        "b_max_smoothed": b_max, "b_avg_smoothed": b_avg,
        "agreement_gap": separation, "agreement_threshold": float(limit),
        "agreement_passed": bool(separation < limit),
        "unimodality_passed": bool(maximum["A"] * average["A"] == 0),
        "maximum_check": maximum, "average_check": average,
        "next_step": "second_order_procedure_not_implemented" if not passed else None,
    }
