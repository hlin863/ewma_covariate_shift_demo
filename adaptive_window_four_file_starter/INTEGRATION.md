# Feasibility-aware adaptive Stage-II — guided four-file implementation

These are **proposed research-extension files** grounded in the interfaces of the current GitHub repository `hlin863/ewma_covariate_shift_demo`. No repo changes were made.

## The four files

1. `src/detection/stage2/window_feasibility.py` — pure rule to select smallest feasible H using PCA features **only through the warning time**.
2. `src/detection/stage2/adaptive_window_hotelling.py` — event validator that uses the selected H to extract equal disjoint before/after windows and **reuses** `calculate_paper_two_sample_hotelling` already in your repository.
3. `tests/detection/stage2/test_adaptive_window_hotelling.py` — dimension checks, missing data, no-future-information selection, shift and pending-state regression tests.
4. `scripts/run_adaptive_window_study.py` — separate controlled fixed-H/adaptive-H comparison, supports a smoke-test synthetic example or precomputed **PCA-space** features + Stage-I warning event CSV.

## Mathematical contract

With d retained PCA components and equal validation window lengths H, require `2H-d-1>=target_df2`, so `H>=ceil((d+1+target_df2)/2)`.

The policy also optionally checks the condition number of a **reference-window covariance estimate** using the configured empirical or shrinkage estimator plus ridge. This does not establish exact F calibration. If `require_full_rank_empirical=True`, require the separate raw reference sample covariance to be full rank as a conservative extra condition.

## Run the four files without changing the baseline

Copy each file to its indicated location in your working repository. From the repo root:

```powershell
python -m pytest tests/detection/stage2/test_adaptive_window_hotelling.py -q
python -m scripts.run_adaptive_window_study --fixed-h 10 --target-df2 10 --alarm-gap 10
```

The default smoke test uses 160 synthetic three-dimensional observations with one **manually provided warning** at index 79 and a mean shift afterward. It deliberately **does not test Stage-I EWMA quality** and is not a reproduction of the published experiment.

For actual EEG, export retained-PCA-space 2D testing features `(trials, d)` from `CSEResult.testing_transformed`, the aligned time vector `testing_times` (as `.npy`), and Stage-I events `warning_results` with columns `time,stage_1_alarm` (as CSV). These must refer to the same trial order and feature space. Then:

```powershell
python -m scripts.run_adaptive_window_study `
  --features-npy outputs/experiments/my_run/testing_pca_features.npy `
  --times-npy outputs/experiments/my_run/testing_times.npy `
  --alarms-csv outputs/experiments/my_run/stage1_warnings.csv `
  --fixed-h 10 --target-df2 10 --max-window 40 --alarm-gap 10
```

The runner writes to a **new unique directory** for each run and stores `fixed_events.csv`, `adaptive_events.csv`, `stage1_alarm_input.csv`, `comparison.csv`, and `manifest.json`.

## Optional CSE core integration (separate from these four files)

Integrating the adaptive validator as a `run_cse()` mode requires a separate, small edit to `src/detection/core.py`, and optionally to `src/detection/stage2/__init__.py` to export its API:

1. Add `adaptive_windows` to allowed `CSEConfig.validation_mode` values.
2. Add an explicit field for the window selection policy (e.g., `adaptive_window_policy: WindowSelectionPolicy | None = None`).
3. In `_run_cse_validation_stage`, add a dedicated `if config.validation_mode == "adaptive_windows":` branch calling `validate_adaptive_window_alarms(testing_transformed, testing_times, warning_results, AdaptiveHotellingConfig(...))` **before** the old fallback to `validate_multivariate_alarms`.
4. Leave the pre-PCA `run_cse` contract alone. The shared `_validate_stage2_feasibility()` currently checks fixed `paper_two_sample` and `retrospective_windows` modes. Adaptive mode should instead check each actual selected window, **not** assume a global fixed H. Record `required_min_window`, selected H and df2 in output.
5. Add a compatibility test that the existing `paper_two_sample` path produces unchanged event results and counts. Do **not** modify the default Table 1 runner, output paths or its published comparison values.

**Design boundary:** in the four-file pilot, the adaptive comparison runner acts on the **same supplied warning times** as the fixed-window validator. Differences are attributable to Stage-II window policy, not Stage-I alarms. Later, use the real `run_cse` output from EEG datasets 2A and 2B after archival of exact settings, fixed seeds and label provenance.

## Research caveats

- Adaptive H selection from past covariance statistics and multiple overlapping warning tests can invalidate nominal fixed-sample p-value calibration. Calibrate the complete warning-selection + window-selection + test procedure using null simulations, measure stream-wise false alarm and detection delay, and consider sequential multiplicity control.
- The existing `calculate_paper_two_sample_hotelling` uses separately estimated covariances plus a classical Hotelling-to-F conversion as an *operational approximation*; do not claim exact p-values for general shrinkage/regularization/unequal-covariance settings.
- Nonzero denominator df and low covariance condition number are **not** sufficient proofs of valid inference or high detection power.
- The fixed `minimum_alarm_gap` in this experiment is a separate policy and is deliberately **not** coupled to selected H. Test overlap effects and alternative spacing policies independently.
- The code is an **implementation proposal**. The included tests exercise its intended behavior but successful local unit tests are not the same as scientific validation on EEG or a full run in your actual repository environment.
