# Park et al. (2023) mean-change rule versus proposed CPSS

This is an **exploratory methodological benchmark**, not a reproduction of the
paper's Figure 3, a validated statistical test, or a demonstration of improved
clinical EEG/BCI outcomes.

## Research question

Does the proposed Calibrated Peak Stability Score (CPSS) improve discrimination
of first-order mean changes, compared with the original binary mean-change
decision of Park, Im and Lim (2023), Algorithm 1? Is a higher score associated
with less variability of peak estimates across small, chronology-preserving
perturbations?

- Original **Park binary**: calculate channelwise mean-CUSUM curves; smooth
  the maximum and average aggregates; check the unimodality product and strict
  peak-location agreement. Accept/reject once on the unperturbed stream.
- Proposed **CPSS**: repeat the same pipeline on B independently jittered
  versions of the stream, compute the existing CPSS formula combining each
  binary result and a normalized peak-separation penalty, and compare the mean
  score with a threshold learned **only** from independent stationary streams.
  The benchmark does not change either detector's implementation.

The gamma=0 and jitter=0 limit exactly recovers the original binary decision.

## Running

From the repository root, with the existing Python environment installed:

```bash
python -m pytest tests/detection/test_park_cpss_benchmark.py -q
python -m scripts.run_park_cpss_benchmark \
  --calibration-seeds 12 --evaluation-seeds 24 \
  --perturbations 8 --target-fpr 0.05 \
  --seed 2026 --output outputs/park_cpss/pilot
```

For a more stable evaluation, use **independent** larger calibration and
evaluation sets, e.g. 100/200, and consider gamma/jitter sensitivity analyses.
Do not reuse test seeds to select thresholds, gamma, or perturbation settings.
Use a fresh output directory for each run; existing outputs are never replaced.

## Data-generating assumptions

The dedicated synthetic generator creates strictly positive four-channel
features (think of power-like feature values rather than signed raw EEG),
with temporally correlated AR(1) noise. The midpoint is the known shift time.

| Scenario | Ground truth for **mean-change** decision | Role |
| --- | --- | --- |
| stationary | 0 | Independent null calibration and held-out false-alarm evaluation |
| mean_shift | 1 | Mean-shift sensitivity and localisation error |
| variance_shift | 0 for *mean change*, not 0 for distribution change | Second-order nuisance/failure-mode probe |

The mean change affects half the channels, respecting the moderately dense
setting discussed in Park Section 2.1. The variance-only generator increases
post-break variance around the unchanged expected mean. This is *not* a
negative case for general distribution-shift detection.

The benchmark deliberately does not transform Park's signed AR/MA generators
into positive signals and claim original-paper reproduction. For those
separate models see `docs/park_structural_simulation.md`.

## Reported measures

`summary.json` contains the exact configuration, null-calibration scores
and seed ranges, threshold, aggregate metrics and limitations. `trials.csv`
contains every paired evaluation stream, prediction, score, peak estimate and
spread value. These make it possible to rebuild plots without re-running.

**Classification (stationary versus mean-shift only):** confusion matrix
TP/TN/FP/FN, accuracy, balanced accuracy, sensitivity, specificity, precision,
F1 and empirical AUC (with ties). AUC for binary decisions is reported, although
it has only two operating points; CPSS AUC measures score ranking.

**Error and spread:** absolute change-point localisation error on *detected*
mean changes, across-seed standard deviation/IQR/10th and 90th percentiles
of CPSS scores and binary outcomes, within-stream CPSS contribution standard
deviation/IQR and peak-location IQR. Counts of ineligible original or perturbed
Park checks are exposed. Ineligible perturbations contribute zero; original
ineligible decisions are treated as no-alarm, not silently dropped.

**False mean-change alarms on variance-only streams** are shown separately
from stationary false alarms. Binary Park has no calibrated nominal alpha,
whereas CPSS uses a conservative separate-null threshold chosen from an
empirical upper-tail order statistic, with **strict score > threshold** and
ties rejected. These are not automatically matched-FPR operating points.

## Interpretation rules

- No empirical performance improvement should be claimed from a unit test or
  simply from implementing CPSS.
- With only 12 null calibration seeds, 5% upper-tail calibration may be
  extremely conservative or degenerate. Report both requested target and
  observed held-out false-positive rates.
- Perturbation spread is **sensitivity under added noise**, not a confidence
  interval, posterior distribution or measured sensor uncertainty.
- The benchmark is retrospective/offline: localisation error is not detection
  delay or ARL. No sequential alarm or energy advantage is inferred.
- Positive feature inputs are required by the current mean-normalised Park
  implementation. The paper's second-order wavelet/DPCA stage is not built
  here; variance-only outputs cannot be used to claim its reproduction.
- Repeated runs should vary shift size, noise/jitter, seed, channel sparsity,
  smoothing span, gamma and sample length. A held-out real EEG evaluation
  needs credible shift-time ground truth and preprocessing provenance.

## Literature and code mapping

- Park, Im & Lim (2023), Section 2.1, Eq. (2.2)–(2.3), Algorithm 1:
  `src/detection/stage1/park_mean_cusum.py`
- Proposed CPSS: `src/detection/stage1/park_cpss.py`
- Matched comparison: `src/experiments/park2023/cpss_benchmark.py`
- CLI: `scripts/run_park_cpss_benchmark.py`
- Focused tests: `tests/detection/test_park_cpss_benchmark.py`

This experiment supplies evidence toward WP1 Goal 3 (performance guarantees)
and Goal 5 (robustness), without treating the score as already statistically
calibrated or the first-order rule as a full covariance-shift detector.


## First GitHub Actions pilot: 18 October 2026

[Successful paired benchmark workflow and archived per-trial CSV/JSON](https://github.com/hlin863/ewma_covariate_shift_demo/actions/runs/39116958678)

Fixed configuration: 24 held-out seeds per scenario, 12 separate stationary
calibration seeds, 8 CPSS perturbations per evaluation, jitter_sd=0.12,
gamma=1, seed=2026, four channels and 160 observations. The CI artifact
contains the four output files for the standard and weaker scenarios.

| Mean shift | Method | Accuracy | Balanced accuracy | F1 | AUC | Held-out stationary false alarm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2.0 | Original Park binary | 1.000 | 1.000 | 1.000 | 1.000 | 0/24 |
| 2.0 | CPSS thresholded | 0.979 | 0.979 | 0.980 | 1.000 | 1/24 |
| 0.65 | Original Park binary | 0.625 | 0.625 | 0.400 | 0.625 | 0/24 |
| 0.65 | CPSS thresholded | 0.708 | 0.708 | 0.611 | 0.709 | 1/24 |

For the standard mean-shift scenario: between-seed CPSS mean score=0.9926,
standard deviation=0.0056 and IQR=0.0049; original binary positives=24/24.
Both methods had mean absolute location error=0.9375 among 24 detections.

For the weaker mean-shift scenario: original binary positives=6/24;
CPSS positives=11/24. CPSS score mean=0.2461, standard deviation=0.3610,
IQR=0.3074; binary decision standard deviation=0.4423, IQR=0.2500.
Mean absolute location error *conditional on detection* was 4.33 (n=6)
for binary and 3.27 (n=11) for CPSS, so those localisation summaries use
different selected cases and are **not** a paired claim of improved precision.
The variance-only mean-change alarm rate was zero for both methods here.

**Important qualification:** In BOTH pilots, the 12-sample stationary
calibration gave CPSS threshold=0. Therefore a nonzero score from *any*
accepted perturbation triggers a CPSS alarm. CPSS gains in the weaker
scenario may be explained by this more permissive effective decision rule;
they do **not** establish a stable or statistically significant improvement.
The 100% baseline accuracy in the standard case is a ceiling effect.
Future runs must use substantially larger independent calibration sets,
multiple seeds/perturbation settings and paired uncertainty intervals, and
preferably compare methods at matched empirical false-positive rates.
