# Park (2023) mean-CUSUM unimodality — worked tutorial

**Work-plan location:** WP2 (CPD benchmarking), supporting WP1 (verified algorithm contracts). This adds the **first-order decision step** to existing Park calculations; it does not implement the paper's second-order wavelet/DPCA detector or establish published numerical reproduction.

**Primary source:** Park, Im & Lim (2023), *Change points detection for nonstationary multivariate time series*, §2.1 Eq. (2.2) and Eq. (2.3), pp. 372–373; §2.3, **Algorithm 1 Steps 2–5, p. 377**.

## Repository placement

- Core computation: `src/detection/stage1/park_mean_cusum.py` — `mean_cusum_curves`, `smooth_cusum_curve`, **`check_cusum_unimodality`**, **`evaluate_mean_change`**.
- Focused tests: `tests/detection/test_park_unimodality.py`.
- This tutorial: `docs/park_unimodality_tutorial.md`.
- Existing worked notebook: `notebooks/05_park_mean_cusum_smoothing.ipynb` (exploratory plots, not algorithm confirmation).

## Step 1: Calculate the channelwise CUSUM and aggregation

`mean_cusum_curves(X)` implements Park Eq. (2.2) for positive-mean columns. For each retained candidate t, aggregate over channels using max and average curves. Original raw outputs remain unchanged. The candidate domain is `min_segment <= t <= T-min_segment`.

## Step 2: Smooth both aggregate curves

Park Algorithm 1 Step 2 calls for local polynomial smoothing with span α=0.1. The repo uses local linear least squares and tricube weighting; **degree 1 and the kernel are reconstruction choices**, not published author specifications. The output fields are `maximum_smoothed`, `average_smoothed` and their peak locations.

## Step 3: Sample both sides of the smoothed peak

Let b denote the smoothed peak. Algorithm 1 Step 3(i) creates:

- D1: k=10 interior equally spaced target time points over [0,b].
- D2: k=10 interior equally spaced target time points over [b,T−1].

Each target maps to its **nearest available candidate time** in the retained CUSUM domain. Explicitly retain this indexing detail when explaining differences from a paper using all t. The paper excludes four boundary comparisons at each side for the implemented statistic. Since `min_segment` truncation can map multiple targets onto the same candidate, this implementation returns **ineligible** rather than inventing an interpolation rule.

## Step 4: Compute Eq. (2.3) with the paper's boundary exclusions

With paper-style 1-based indices:

```text
A = sum(i=6..10) I[D1[i-1] > D1[i]]
  + sum(j=1..5)  I[D2[j] < D2[j+1]]
```

These are **shape-violation counts**: an ideal rise towards the peak followed by a decrease has A=0. Compute A separately for the smoothed maximum and average curves. For this exact implemented k=10 version, inspect `left_violation_indices` and `right_violation_indices`.

A=0 is a diagnostic for the sampled shape under the paper's rule; it is not a distributional significance p-value.

## Step 5: Apply the strict mean-change rule

Park Algorithm 1 Step 4 accepts the mean-change branch if both conditions hold:

```text
A_max * A_avg == 0
abs(b_max_smoothed - b_avg_smoothed) < sqrt(T)/2
```

The first condition permits **either** smoothed aggregate to have no sampled shape violations; it does not require both. On acceptance, the estimated location is `(b_max + b_avg)/2`. The result status is `mean_change_candidate` to distinguish a paper-rule decision from a calibrated prospective online alarm.

On failure, status is `no_mean_change_by_rule`. **Park Algorithm 1 Step 5 then advances to wavelet local-spectral / dynamic-PCA second-order detection**, which is not implemented.

## Step 6: Run a reproducible example

```python
import numpy as np
from src.detection.stage1.park_mean_cusum import (
    mean_cusum_curves, check_cusum_unimodality, evaluate_mean_change,
)

rng = np.random.default_rng(15)
x = rng.normal(10, 0.7, (240, 3))
x[120:] += [3, 2, 4]  # planted mean change at zero-based row 120
curves = mean_cusum_curves(x, min_segment=8)
result = evaluate_mean_change(curves)
print(result["status"], result["estimated_position"])
print(result["maximum_check"]["A"], result["average_check"]["A"])
print(result.get("agreement_gap"), result.get("agreement_threshold"))
```

Do **not** assume that a specific random trace must pass; test multiple seeds and a no-change control before claiming detection performance.

## Step 7: Validate locally

```powershell
python -m pytest tests/detection/test_park_unimodality.py -q
python -m pytest -q
```

The tests cover a single-peaked reference, local reversals, close/distant peak locations, unsupported short candidate domains and an end-to-end synthetic computation. This tutorial is an executable, reviewable method walkthrough, **not a claim of reproducing Park's published results**.

## Known limits and next research tasks

1. **Paper indexing interpretation:** Algorithm 1 places k=10 points within [0,b] and [b,T−1] and then maps to available candidate indices. Retained-domain sampling, ties and strictly increasing positions deserve an additional parity study against paper-supplied code, if available.
2. **Short-series problem:** existing `mean_cusum_curves` may itself fail when smoothing has fewer than three candidates. The confirmation function returns ineligible when ten distinct sampling locations are impossible, but does not eliminate that earlier smoothing exception.
3. **False alarms:** `A=0` plus small peak separation can occur on noise. No validated false-alarm control exists yet.
4. **Second order:** the algorithm's conditional wavelet/dynamic-PCA step, Model B(ii), and complete matched-detector benchmark remain open.
5. **Performance:** evaluate repeated seeds, no-change and shift streams, change-position error, power, stream-level false alarms and runtime before promoting this as a WP2 completed benchmark.
