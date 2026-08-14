# EWMA Covariate Shift Demo

Research reproduction and extension project for EWMA-based covariate-shift estimation in non-stationary data, with a particular focus on the CSE and CSE-UAEL methods described by Raza et al. (2015, 2019).

## Implemented pipeline

- BCI Competition IV Dataset 2A/2B data loading helpers
- Cue-aligned EEG trial extraction for Dataset 2A and Dataset 2B
- Dataset 2A paper channel selection: C3, FC3, CP3, C5, C1, C4, FC4, CP4, C2, C6
- Dataset 2A left/right-hand training from Session I and unlabeled Session II evaluation
- 10-band filter-bank CSP feature extraction
  - 8th-order zero-phase Butterworth band-pass filters
  - overlapping bands from 8-12 Hz through 26-30 Hz
  - CSP spatial filtering and log-normalised variance features
- PCA fitted on training features and reused on evaluation data
- EWMA Stage-I covariate-shift warnings
  - data-driven lambda estimation
  - configurable control limits and variance updates
- Multivariate Stage-II Hotelling validation
- Combined Dataset 2A/2B Table 1 reproduction output
- Diagnostic and sensitivity-analysis utilities
- Pytest coverage and GitHub Actions CI for the CSE/EWMA pipeline

## Reproduce the paper's Table 1 structure

The paper reports subject-level smoothing constant (lambda), covariate-shift warnings (CSW), and covariate-shift validations (CSV) for nine Dataset 2A subjects and nine Dataset 2B subjects. The repository now keeps those published values only as reference targets and writes the experiment's computed values into the same side-by-side 2A/2B structure.

Expected raw-data layout:

```text
data/raw/bci_competition_iv_2a/
  A01T.gdf ... A09T.gdf
  A01E.gdf ... A09E.gdf

data/raw/bci_competition_iv_2b/
  B0101T.gdf ... B0905E.gdf
```

Run the combined reproduction:

```bash
python scripts/run_bci_table1_reproduction.py
```

The default run uses the paper-oriented equal-window two-sample Stage-II mode. The validation interpretation can also be compared explicitly:

```bash
python scripts/run_bci_table1_reproduction.py --validation-mode paper_two_sample
python scripts/run_bci_table1_reproduction.py --validation-mode algorithm1_training_reference
python scripts/run_bci_table1_reproduction.py --validation-mode retrospective_windows
```

Outputs:

```text
outputs/metrics/bci_table1_reproduction.md
outputs/metrics/bci_table1_comparison.csv
```

`bci_table1_reproduction.md` contains the computed subject rows and mean row in the grouped Dataset 2A / Dataset 2B layout. `bci_table1_comparison.csv` contains published values, computed values, and CSW/CSV differences for calibration and reproducibility analysis.

The combined runner does not replace computed results with the publication targets.

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
    validation_before_size=10,
    validation_after_size=10,
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

## Testing

Run the complete suite locally with:

```bash
python -m pytest -q
```

The tests include Dataset 2A trial extraction, Dataset 2A FBCSP feature construction, Stage-II two-sample validation, Table 1 formatting, existing Dataset 2B regression tests, and EWMA/CSE unit tests. GitHub Actions also runs the suite on pushes and pull requests.

## Current research boundary

The repository currently covers the signal-processing, feature-extraction, CSE warning, and Stage-II validation portions of the 2019 CSE-UAEL pipeline. The PWKNN-driven unsupervised adaptation, dynamic LDA ensemble growth, and weighted ensemble classification remain future implementation stages.
