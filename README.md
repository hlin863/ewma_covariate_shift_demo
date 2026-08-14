# EWMA Covariate Shift Demo

Research reproduction and extension project for EWMA-based covariate-shift estimation in non-stationary data, with a particular focus on the CSE and CSE-UAEL methods described by Raza et al. (2015, 2019).

## Implemented pipeline

- BCI Competition IV Dataset 2A/2B data loading helpers
- Cue-aligned EEG trial extraction for Dataset 2B
- 10-band filter-bank CSP feature extraction
  - 8th-order zero-phase Butterworth band-pass filters
  - overlapping bands from 8-12 Hz through 26-30 Hz
  - CSP spatial filtering and log-normalised variance features
- PCA fitted on training features and reused on evaluation data
- EWMA Stage-I covariate-shift warnings
  - data-driven lambda estimation
  - configurable control limits and variance updates
- Multivariate Stage-II Hotelling validation
- Diagnostic and sensitivity-analysis utilities
- Pytest coverage for the CSE/EWMA pipeline

## Stage-II validation modes

`CSEConfig.validation_mode` currently supports three explicitly separated interpretations:

- `paper_two_sample`: paper-aligned equal-length, disjoint multivariate subsequences around a Stage-I warning. This is the preferred mode when reproducing the two-sample Hotelling procedure described in the CSE methodology.
- `algorithm1_training_reference`: literal interpretation of the 2019 Algorithm 1 wording, comparing the current transformed feature vector against the training reference distribution.
- `retrospective_windows`: general retrospective before/after-window validation that allows unequal window sizes and additional experimental settings.

The separation is intentional because the 2019 paper's Algorithm 1 wording and the detailed two-sample methodology inherited from the cited 2015 CSE paper are not identical descriptions of Stage II.

## Paper-aligned two-sample Stage II

Use equal validation windows:

```python
from src.cse import CSEConfig, run_cse

config = CSEConfig(
    validation_mode="paper_two_sample",
    validation_before_size=50,
    validation_after_size=50,
    validation_alpha=0.05,
    covariance_method="empirical",
)

result = run_cse(
    training_features=training_features,
    testing_features=testing_features,
    testing_times=testing_times,
    config=config,
)
```

`paper_two_sample` rejects unequal sample sizes instead of silently converting a single feature vector into a sample. The validation result records the reference and current subsequence boundaries, validation time, Hotelling statistic, F statistic, p-value, and confirmed/rejected status.

## Current research boundary

The repository currently covers the signal-processing, feature-extraction, CSE warning, and Stage-II validation portions of the 2019 CSE-UAEL pipeline. The PWKNN-driven unsupervised adaptation, dynamic LDA ensemble growth, and weighted ensemble classification remain future implementation stages.
