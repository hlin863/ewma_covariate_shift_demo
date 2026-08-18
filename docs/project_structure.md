# Project structure

The repository is organised by research responsibility rather than by a flat list of implementation files.

```text
src/
├── bci/
│   ├── data.py                 # GDF/session loading and raw BCI data models
│   ├── features.py             # generic repo-specific EEG feature helpers
│   ├── fbcsp.py                # paper-aligned Butterworth + CSP/FBCSP path
│   └── datasets/
│       ├── dataset2a/          # Dataset 2A experiment/development interface
│       └── dataset2b/          # Dataset 2B experiment/diagnostics/reference interface
├── detection/
│   ├── __init__.py             # CSE orchestration API
│   ├── stage1/                 # EWMA warning detectors
│   └── stage2/                 # Hotelling validation methods
├── reporting/
│   └── table1.py               # paper-style Table 1 output/comparison helpers
└── web/
    └── dashboard.py            # Flask visualisation layer
```

## Import policy

New code should use the structured package imports:

```python
from src.bci.data import load_bci_competition_iv_2a_session
from src.bci.datasets.dataset2a import extract_dataset_2a_trials
from src.detection import CSEConfig, run_cse
from src.detection.stage2 import validate_algorithm1_alarms
from src.reporting.table1 import Table1Row
```

The former flat modules remain import-compatible where practical so existing notebooks, tests, and downstream scripts do not break abruptly. Core BCI data, generic feature, FBCSP, Table 1 reporting, and Flask implementation code now live in the structured packages; the old modules are compatibility entry points.

## Responsibility boundaries

- `src/bci`: data acquisition, EEG preprocessing and feature construction.
- `src/bci/datasets`: dataset-specific assumptions such as sessions, channels, cue codes and published reproduction targets.
- `src/detection`: dataset-agnostic covariate-shift estimation logic.
- `src/detection/stage1`: EWMA-based CS warning generation.
- `src/detection/stage2`: CS warning validation and Hotelling variants.
- `src/reporting`: formatting and comparison of experimental outputs.
- `src/web`: presentation only; it consumes generated result files and does not run the scientific pipeline itself.
- `scripts`: executable experiment orchestration.
- `tests`: unit, regression, integration and architecture smoke tests.
- `archive`: legacy workflows retained for reference but excluded from the active architecture.

## Why this structure

The separation keeps paper-specific Dataset 2A/2B assumptions out of the generic CSE detector and prevents the Flask presentation layer from becoming coupled to EEG processing. It also makes future CSE-UAEL work easier to place: adaptation/classification modules can be added beside `detection` as a dedicated package without expanding the current flat `src` namespace.
