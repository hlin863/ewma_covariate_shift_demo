# EWMA Covariate Shift Demo

Research reproduction and extension project for EWMA-based covariate-shift estimation in non-stationary data, with a particular focus on the CSE and CSE-UAEL methods described by Raza et al. (2015, 2019).

## Project structure

The active code is organised by research responsibility rather than as one flat `src` directory:

```text
src/
├── bci/
│   ├── data.py                       # GDF loading and raw session models
│   ├── features.py                   # generic repo-specific EEG features
│   ├── fbcsp.py                      # paper-aligned Butterworth + CSP/FBCSP
│   └── datasets/
│       ├── dataset2a/
│       │   ├── experiment.py         # Dataset 2A trial/expt logic
│       │   ├── development_split.py  # Session-I 70/30 split
│       │   └── development_pipeline.py
│       └── dataset2b/
│           ├── experiment.py
│           ├── diagnostics.py
│           └── reference.py
├── detection/
│   ├── core.py                       # dataset-agnostic CSE orchestration
│   ├── preprocessing.py              # PCA fit/transform helpers
│   ├── stage1/
│   │   ├── sd_ewma.py
│   │   └── msd_ewma.py
│   └── stage2/                       # Hotelling validation API
├── reporting/
│   ├── table1.py                     # Table 1 output/comparison
│   └── metrics.py                    # detector evaluation metrics
├── simulation/
│   ├── gaussian.py                   # synthetic abrupt mean-shift streams
│   └── jumping_mean.py               # paper D2 AR jumping-mean stream
├── experiments/
│   └── paper2015/
│       └── d2.py                     # D2 generation/detection/evaluation
└── web/
    └── dashboard.py                  # Flask presentation layer
```

New code should use these structured imports, for example:

```python
from src.bci.data import load_bci_competition_iv_2a_session
from src.bci.datasets.dataset2a import extract_dataset_2a_trials
from src.detection import CSEConfig, run_cse
from src.detection.stage2 import validate_algorithm1_alarms
from src.reporting.table1 import Table1Row
```

The former flat modules such as `src.cse`, `src.fbcsp`, `src.bci_data`, `src.ewma`, and `src.table1_reproduction` are retained as compatibility shims so existing notebooks/tests do not have to migrate in one breaking change. See `docs/project_structure.md` for the responsibility boundaries and migration policy.

## Implemented pipeline

- BCI Competition IV Dataset 2A/2B data loading helpers
- Cue-aligned EEG trial extraction for Dataset 2A and Dataset 2B
- Dataset 2A paper channel selection: C3, FC3, CP3, C5, C1, C4, FC4, CP4, C2, C6
- Dataset 2A Session-I stratified 70/30 development split before FBCSP fitting
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
- Flask dashboard for published-versus-computed Table 1 results
- Diagnostic and sensitivity-analysis utilities
- Paper-aligned D2 jumping-mean generator with repeated-shift truth labels
- Full-stream SD-EWMA/TSSD-EWMA D2 experiment and event-level metrics
- Pytest coverage and GitHub Actions CI for the CSE/EWMA pipeline

## Reproduce the paper's Table 1 structure

The paper reports subject-level smoothing constant (lambda), covariate-shift warnings (CSW), and covariate-shift validations (CSV) for nine Dataset 2A subjects and nine Dataset 2B subjects. The repository keeps those published values only as reference targets and writes the experiment's computed values into the same side-by-side 2A/2B structure.

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

Dataset 2A now applies its Session-I 70/30 development split before FBCSP fitting. The validation fraction and reproducible split seed are explicit experiment controls:

```bash
python scripts/run_bci_table1_reproduction.py \
  --session1-validation-fraction 0.30 \
  --session1-split-seed 42
```

The paper specifies the 70/30 ratio but does not report the exact random partition, so the seed is a repository reproducibility choice rather than a paper-defined parameter.

The Stage-II interpretation can be compared explicitly:

```bash
python scripts/run_bci_table1_reproduction.py --validation-mode paper_two_sample
python scripts/run_bci_table1_reproduction.py --validation-mode algorithm1_training_reference
python scripts/run_bci_table1_reproduction.py --validation-mode retrospective_windows
```

PCA retention is an explicit experiment option. Stage I still monitors PC1 only in the univariate Algorithm-1 path:

```bash
python scripts/run_bci_table1_reproduction.py --pca-components 1
python scripts/run_bci_table1_reproduction.py --pca-components 3
python scripts/run_bci_table1_reproduction.py --pca-components 0.95
python scripts/run_bci_table1_reproduction.py --pca-components all
```

Each subject line reports the retained PCA component count and the PC1 explained-variance ratio so PCA behavior can be audited alongside CSW/CSV results.

Outputs:

```text
outputs/metrics/bci_table1_reproduction.md
outputs/metrics/bci_table1_comparison.csv
```

`bci_table1_reproduction.md` contains computed subject rows and the mean row in the grouped Dataset 2A / Dataset 2B layout. `bci_table1_comparison.csv` contains published values, computed values, and CSW/CSV differences for calibration and reproducibility analysis.

## Flask visualisation dashboard

The dashboard implementation lives in `src/web/dashboard.py`; root `app.py` remains a stable launch/compatibility entry point.

```bash
python -m pip install -r requirements-dashboard.txt
python scripts/run_bci_table1_reproduction.py --validation-mode paper_two_sample
python app.py
```

The dashboard reads `outputs/metrics/bci_table1_comparison.csv` on each request, so new experiment output updates the visualisation without copying values into the web application.

## Stage-II validation modes

`CSEConfig.validation_mode` supports three explicitly separated interpretations:

- `paper_two_sample`: equal-length disjoint multivariate subsequences around a Stage-I warning.
- `algorithm1_training_reference`: interpretation of the 2019 Algorithm 1 wording, comparing the current transformed feature vector against the training reference distribution.
- `retrospective_windows`: general before/after-window validation supporting additional experimental settings.

These remain separated because the 2019 Algorithm 1 wording and the detailed two-sample methodology are not identical descriptions of Stage II.

Example using the canonical detection namespace:

```python
from src.detection import CSEConfig, run_cse

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

## Reproduce the 2015 D2 jumping-mean experiment

Run the full testing-stream evaluation with lambda estimated from the configured
training section:

```bash
python scripts/run_2015_synthetic_reproduction.py --dataset d2
```

Use the paper's reported lambda of 0.40 as an explicit diagnostic mode:

```bash
python scripts/run_2015_synthetic_reproduction.py \
  --dataset d2 \
  --lambda-mode configured
```

The runner writes a JSON summary, Stage-I/Stage-II event tables and detector
traces beneath `outputs/metrics/paper2015/d2/`. See
[`docs/d2_reproduction.md`](docs/d2_reproduction.md) for the indexing,
final-regime and Table III evaluation-scope decisions.

## Testing

Run the architecture smoke test first, then the complete suite:

```bash
python -m pytest tests/test_project_structure.py -v
python -m pytest -q
```

The test suite covers Dataset 2A trial extraction and 70/30 development processing, FBCSP construction, Stage-II validation, Table 1 formatting, Flask dashboard rendering, PCA parsing, Dataset 2B regression behavior, EWMA/CSE logic, and legacy-import compatibility.

## Current research boundary

The repository currently covers signal processing, feature extraction, CSE warning, and Stage-II validation portions of the 2019 CSE-UAEL pipeline. PWKNN-driven unsupervised adaptation, dynamic LDA ensemble growth, and weighted ensemble classification remain future implementation stages.
