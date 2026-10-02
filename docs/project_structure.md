# Project structure

The repository is organised by research responsibility rather than by a flat list of implementation files. This active structure coexists with a preservation layer: the repository is a living computational literature review, so historically meaningful implementations, compatibility shims, paper-specific alternatives and milestone outputs are retained when they document how the research interpretation evolved.

See [Living computational literature review](living_literature_review/README.md) for the paper-to-code lineage and preservation rules.

```text
src/
├── bci/
│   ├── data.py                       # GDF/session loading and raw BCI models
│   ├── features.py                   # generic repo-specific EEG features
│   ├── fbcsp.py                      # paper-aligned Butterworth + CSP/FBCSP
│   ├── splitting.py                  # shared stratified development indices
│   └── datasets/
│       ├── dataset2a/
│       │   ├── experiment.py         # trial extraction + subject experiment
│       │   ├── development_split.py  # Session-I 70/30 split
│       │   └── development_pipeline.py
│       └── dataset2b/
│           ├── experiment.py
│           ├── development_split.py  # 2019 I–III pool / IV–V evaluation
│           ├── development_pipeline.py # fit on development training only
│           ├── diagnostics.py
│           └── reference.py
├── adaptation/
│   ├── classifier.py                 # classifier and retraining contracts, SVM
│   ├── supervised.py                 # supervised + PWKNN transductive adaptation
│   ├── experiments/                  # real and synthetic bagging comparisons
│   └── ensemble/
│       └── bagging.py                # bootstrap ensemble classifier
├── detection/
│   ├── core.py                       # dataset-agnostic CSE orchestration
│   ├── preprocessing.py              # PCA preprocessing
│   ├── stage1/
│   │   ├── sd_ewma.py
│   │   └── msd_ewma.py
│   └── stage2/                       # Hotelling validation API/variants
├── reporting/
│   ├── table1.py                     # paper-style Table 1 output
│   └── metrics.py                    # detector evaluation metrics
├── simulation/
│   ├── gaussian.py                   # synthetic abrupt mean-shift data
│   └── jumping_mean.py               # D2 AR jumping-mean data and truth
├── experiments/
│   └── paper2015/
│       └── d2.py                     # paper-specific D2 orchestration
└── web/
    ├── home.py                       # paper-grounded home-page model
    └── dashboard.py                  # Flask routes and visualisation layer
```

## Import policy

New code should use structured package imports:

```python
from src.bci.data import load_bci_competition_iv_2a_session
from src.bci.datasets.dataset2a import extract_dataset_2a_trials
from src.detection import CSEConfig, run_cse
from src.detection.stage1 import fit_sd_ewma
from src.detection.stage2 import validate_algorithm1_alarms
from src.reporting.table1 import Table1Row
```

Former flat modules remain import-compatible where practical so notebooks, tests and downstream scripts do not break abruptly. The old modules are migration shims, not the preferred location for new functionality.

## Responsibility boundaries

- `src/bci`: raw BCI data access, EEG preprocessing and feature construction.
- `src/bci/datasets`: dataset-specific assumptions such as sessions, channels, cue codes, development splits and published reproduction targets.
- `src/bci/splitting.py`: shared stratified index operation; paper-specific session assignments stay in dataset packages.
- `src/adaptation`: classifier contracts, supervised adaptation, PWKNN pseudo-labelled transductive updates, and separate real/synthetic bagging comparisons. Dynamic CSE-UAEL ensemble growth and weighted voting remain unimplemented.
- `src/detection`: dataset-agnostic covariate-shift estimation logic.
- `src/detection/stage1`: EWMA-based CS warning generation.
- `src/detection/stage2`: K-S/Hotelling CS warning validation methods.
- `src/detection/two_stage.py`: reusable univariate TSSD-EWMA orchestration.
- `src/reporting`: experiment metrics, paper-style output and published/computed comparisons.
- `src/simulation`: synthetic generators and ground-truth labels, independent of detector configuration.
- `src/experiments`: paper-specific dataset splits, detector settings and evaluation scope.
- `src/web`: presentation only; it consumes generated result files and does not execute EEG/CSE processing.
- `scripts`: executable experiment orchestration.
- `tests`: domain-grouped numerical, protocol, pipeline and regression checks; web, CLI and legacy compatibility checks are separate. See [test architecture](../tests/README.md).
- `archive`: superseded research workflows retained as provenance records. They are excluded from canonical execution, but remain part of the living literature review when they document an earlier methodological interpretation, data protocol, or experiment.

## Compatibility strategy

The refactor deliberately avoids a flag-day migration. Existing imports such as:

```python
from src.cse import run_cse
from src.fbcsp import fit_fbcsp
from src.bci_data import BCISessionData
```

continue to resolve through compatibility shims, while active orchestration such as `scripts/run_bci_table1_reproduction.py` uses the structured namespaces. This lets the test suite expose migration regressions without forcing every notebook and utility script to change in one commit.

Compatibility shims have a second role in this repository: they are implementation-lineage markers. A shim should remain when it records a meaningful transition from an earlier research layout to a canonical package. New functionality should still target the canonical package so preservation does not create two competing active implementations.

## Why this structure

The separation keeps paper-specific Dataset 2A/2B assumptions out of the generic CSE detector, distinguishes the paper-aligned FBCSP path from repo-specific generic EEG features, and prevents the Flask presentation layer from becoming coupled to signal processing. It also gives future CSE-UAEL adaptation/classification work a clean place to grow as a separate functional package instead of expanding the flat `src` namespace.
