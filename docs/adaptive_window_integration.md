# Adaptive-window Stage-II integration

The canonical adaptive-window modules live in `src/detection/stage2/` and reuse `calculate_paper_two_sample_hotelling` from the existing fixed-window validator. The study runner and tests live at the repository root under `scripts/` and `tests/`. The original Table 1 experiment is unchanged.

## Verification

```powershell
python -m pytest tests/detection/stage2/test_adaptive_window_hotelling.py -q
python -m scripts.run_adaptive_window_study --fixed-h 10 --target-df2 10 --alarm-gap 10
```

The study runner's default synthetic example uses a manually supplied Stage-I warning. It does not test EWMA warning detection or reproduce the paper's Table 1.

## EEG feature inputs

Supply the aligned testing-PCA representation from `CSEResult.testing_transformed`, testing timestamps, and `warning_results` CSV containing `time` and `stage_1_alarm`.

```powershell
python -m scripts.run_adaptive_window_study `
  --features-npy outputs/experiments/my_run/testing_pca_features.npy `
  --times-npy outputs/experiments/my_run/testing_times.npy `
  --alarms-csv outputs/experiments/my_run/stage1_warnings.csv `
  --fixed-h 10 --target-df2 10 --max-window 40 --alarm-gap 10
```

These input paths are examples; the original reproduction runner does not automatically write these arrays. Outputs are stored under a unique timestamp/identifier with fixed/adaptive event CSVs, comparison, and manifest.

## Mathematical contract and limitations

The window selector uses only PCA-space history available through the warning time and requires `2H-d-1 >= target_df2`. An optional reference-covariance condition-number bound and raw empirical full-rank check can further restrict selection. Satisfying these conditions does not establish calibrated p-values. The F conversion is the existing operational approximation; test stream-wise false alarms and delays with null and known-shift simulations. Keep alarm spacing independent of selected H.

## Source provenance

Imported and consolidated from the starter files in commit `d03dc519` so there is one canonical executable copy. Git history preserves the original starter tree.

## Integrated optional `run_cse()` mode

The shared CSE API now supports `CSEConfig(validation_mode="adaptive_windows")`.
All existing fixed-window modes and the published Table 1 runner retain their
original settings. The adaptive method uses the explicit policy when supplied,
otherwise builds a policy from `covariance_method` and `covariance_regularization`;
its default Stage-II alarm gap is 10 when `minimum_alarm_gap=None`.

```python
from src.cse import CSEConfig, run_cse
from src.detection.stage2 import WindowSelectionPolicy

config = CSEConfig(
    pca_components=3,
    validation_mode="adaptive_windows",
    minimum_alarm_gap=10,
    adaptive_window_policy=WindowSelectionPolicy(
        min_window=2, max_window=40, target_df2=10,
    ),
)
result = run_cse(training_features, testing_features, testing_times, config)
# Stage-I warning_results and adaptive Stage-II validation_results are retained.
```

This optional mode does not modify `scripts/run_bci_table1_reproduction.py`.
The experimental study runner remains the preferred fixed/adaptive comparator.
