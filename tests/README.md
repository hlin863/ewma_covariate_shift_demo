# Test architecture

The test suite is organised by scientific and software responsibility rather than by one flat list of source-module mirrors.

```text
tests/
├── adaptation/
│   ├── conftest.py
│   ├── test_classifier.py
│   ├── test_policies.py
│   ├── test_supervised_loop.py
│   └── test_validation.py
├── detection/
├── integration/
│   ├── test_cse_to_adaptation.py
│   └── test_2018_online_bci_flow.py
├── regression/
│   └── test_adaptation_regression.py
├── reporting/
├── simulation/
└── realdata/
    └── test_online_bci_dataset2a_smoke.py
```

## Roles

- **unit**: isolated deterministic behaviour of one class or function.
- **contract**: schema and interface compatibility between modules.
- **integration**: sequential behaviour across multiple modules.
- **regression**: protects deliberately established behaviour and accepted reproducibility outputs.
- **research**: methodological invariants, including protection against label leakage and invalid experiment construction.
- **realdata**: checks requiring external BCI Competition files.
- **slow**: computationally expensive tests that should not be required in the fastest development loop.

## Adaptation test boundaries

`tests/adaptation/test_classifier.py` verifies classifier lifecycle behaviour only. It must not execute EWMA or Stage-II validation.

`tests/adaptation/test_policies.py` treats update policies as pure decision rules. It must not fit classifiers or load EEG data.

`tests/adaptation/test_supervised_loop.py` verifies sequential adaptation semantics with deterministic synthetic features, including the invariant that prediction for trial `t` occurs before an update caused by trial `t`.

`tests/adaptation/test_validation.py` protects input integrity and research-validity constraints.

`tests/integration/test_cse_to_adaptation.py` checks the contract between CSE outputs and adaptation inputs.

`tests/integration/test_2018_online_bci_flow.py` verifies the complete synthetic detect -> validate -> adapt sequence without claiming reproduction of the original clinical experiment.

`tests/regression/test_adaptation_regression.py` should only lock values that have been deliberately established. It must not force uncertain published values to match.

`tests/realdata/test_online_bci_dataset2a_smoke.py` is reserved for a representative external-data smoke run and should not be part of the fastest unit-test loop.

## Suggested commands

```bash
pytest -m unit
pytest -m "unit or contract or integration"
pytest -m "not realdata and not slow"
pytest -m realdata
```
