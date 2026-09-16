# Test architecture

The existing working pytest cases are grouped by scientific and software responsibility. This migration preserves the original test contents and changes only their paths.

```text
tests/
├── bci/
│   ├── test_bci_2a_development_pipeline.py
│   ├── test_bci_2a_development_split.py
│   ├── test_bci_2a_experiment.py
│   ├── test_bci_2a_validation_calibration.py
│   ├── test_bci_2b_experiment.py
│   ├── test_bci_2b_features.py
│   ├── test_bci_data.py
│   └── test_cse_bci_features.py
├── detection/
│   ├── test_cse.py
│   ├── test_cse_algorithm1.py
│   ├── test_cse_lambda_override.py
│   ├── test_cse_paper_stage_2.py
│   ├── test_cse_preprocessing.py
│   ├── test_ewma_training.py
│   ├── test_ewma_variance_modes.py
│   ├── test_msd_ewma.py
│   ├── test_sd_ewma_detection.py
│   ├── test_stage_1_paper_cases.py
│   ├── test_stage_2.py
│   ├── test_tssd_ewma_pipeline.py
│   ├── test_ici_cdt.py
│   └── test_two_stage_namespace.py
├── integration/
│   ├── test_d2_reproduction.py
│   └── test_project_structure.py
├── reporting/
│   ├── test_dashboard.py
│   ├── test_evaluation.py
│   ├── test_repeated_shift_metrics.py
│   ├── test_table1_reproduction.py
│   └── test_table1_runner_cli.py
└── simulation/
    └── test_jumping_mean.py
```

The placeholder-only adaptation, regression, real-data, and 2018-flow test files were removed from this migration so that every `test_*.py` file in the grouped suite corresponds to an existing working case.

## Commands

Run the complete suite:

```bash
python -m pytest -q
```

Run a group:

```bash
python -m pytest tests/bci -v
python -m pytest tests/detection -v
python -m pytest tests/integration -v
python -m pytest tests/reporting -v
python -m pytest tests/simulation -v
```

Run selected cases:

```bash
python -m pytest tests/detection/test_cse.py -v
python -m pytest tests/detection/test_stage_2.py -v
python -m pytest tests/bci/test_bci_2a_experiment.py -v
python -m pytest tests/reporting/test_table1_reproduction.py -v
```
