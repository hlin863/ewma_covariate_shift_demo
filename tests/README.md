# Test architecture and research value

Tests are grouped by the component or protocol they protect. A short test is
valuable when its failure exposes a numerical, methodological, data, or
application regression. Test length and passing-test count are not measures of
research quality.

## Hierarchy

| Directory | Responsibility |
| --- | --- |
| `bci/` | Shared session loading and paper-specific filter-bank contracts |
| `bci/datasets/dataset2a/` | Channels, cues, Session-I development split, FBCSP fit scope and validation calibration |
| `bci/datasets/dataset2b/` | Session protocols, evaluation labels, features and subject experiments |
| `bci/datasets/chowdhury/` | Cohort metadata, file resolution and experiment preparation |
| `bci/pipelines/` | Exported feature files through PCA and CSE |
| `detection/preprocessing/` | PCA fitting, reuse, reconstruction and residual energy |
| `detection/stage1/` | EWMA recurrence, training, alarm boundaries and variance modes |
| `detection/stage2/` | K-S, training-reference Hotelling and paper two-sample Hotelling validation |
| `detection/pipelines/` | CSE/TSSD execution, parameter forwarding and window/decision consistency |
| `detection/baselines/` | ICI-CDT reference detector |
| `adaptation/` | Classifier contracts, supervised updates, PWKNN pseudo-labelling, confidence-gated transductive adaptation, policy decisions and prediction-before-update ordering |
| `adaptation/experiments/` | Diethe scenarios, matched policies, bagging protocols and exports |
| `experiments/paper2015/` | D1 detection and D2/Table III reproduction orchestration |
| `simulation/` | D2 recurrence, shift schedule, indexing and reproducibility |
| `reporting/` | False alarms, missed shifts, delay, event matching and Table 1 exports |
| `web/` | Pages/APIs, artifact provenance, result presentation and JUnit source inspection |
| `cli/` | Command-line parameter interpretation |
| `compatibility/` | Legacy imports resolving to canonical implementations |
| `support/` | Local retrieval and generation-context integration |

This is a domain hierarchy, not a claim that every case is an isolated unit
test. Pipeline and experiment cases exercise multiple modules. BCI cases use
synthetic signals, temporary feature/label files or mocked loaders; the committed
Chowdhury demographic CSV is also checked. In particular,
`test_bagging_bci_protocol.py` mocks the real-data loader. Its former name,
`test_bagging_bci_real.py`, did not mean it ran external EEG recordings.

## Commands

Run from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Run focused groups:

```bash
python -m pytest tests/detection/stage1 tests/detection/stage2 -v
python -m pytest tests/detection/pipelines tests/bci/pipelines -v
python -m pytest tests/bci/datasets/dataset2a tests/bci/datasets/dataset2b -v
python -m pytest tests/adaptation tests/experiments tests/reporting -v
python -m pytest tests/web tests/cli tests/compatibility -v
```

Refresh the application's test-results page with the current paths:

```bash
python -m pytest -q --junitxml=outputs/metrics/pytest_results.xml
```

`pytest.ini` uses `xunit1` because the dashboard consumes per-case
`record_property` values for actual/expected comparisons. Web fixtures restore
the shared Flask configuration and clear the support cache after each case.
The source-analysis test exercises a nested path under
`tests/detection/pipelines/`. CI still runs the full suite using its existing
`python -m pytest -q` command.

Use directory selection for these groups. Existing pytest markers are retained,
but annotation is not comprehensive: `-m research` or `-m integration` does not
select all relevant cases.

## Review decisions: 1 October 2026

Baseline: **250 passed** at `4db1878`. After review: **242 passed** on Python 3.12.
Five low-value cases were removed and three overlapping cases were consolidated.
No tests were skipped or excluded to obtain this result.

### Removed

| Former test | Reason and retained protection |
| --- | --- |
| `test_canonical_bci_namespace_exposes_core_components` | Class-name assertions do not establish behaviour. Filter-bank assertions duplicate `bci/test_filter_bank_variants.py`; loader and FBCSP cases exercise the classes. |
| `test_dataset_namespaces_expose_reproduction_entry_points` | Callable-only checks add little beside dataset extraction and development pipeline cases. |
| `test_detection_namespace_groups_cse_stages` | Names/callability do not verify detection. Stage-specific and pipeline cases exercise the implementations. |
| `test_reporting_and_web_namespaces_are_importable` | Class and Flask application names do not verify reports or routes. Table-export and web cases cover behaviour. |
| `test_public_simulation_namespace_exports_d2_generator` | Every retained D2 generator test imports and calls this same public entry point. |

The first four cases were in `integration/test_project_structure.py`. Its legacy
object-identity check remains in `compatibility/test_legacy_imports.py`, alongside
the two-stage compatibility check. These protect real import contracts even
though they are not experimental evidence.

### Consolidated

| Former standalone case | Retained or strengthened case |
| --- | --- |
| `test_multivariate_ewma_path_returns_expected_shapes` | Shapes now accompany a manually calculated two-step recurrence in `detection/stage1/test_msd_ewma.py`. Both states and both prediction errors are checked. |
| `test_session1_split_returns_explicit_development_split` | Type and 14/6 size assertions now accompany class balance, disjointness and complete trial preservation in the Dataset 2A split test. |
| `test_home_page_links_support_layer` | Support-link and label assertions now sit in the existing home-page application-map test. |

### Kept deliberately

- Equality-inclusive warning boundaries, initial limits, lambda selection and
  variance-update modes: these directly affect warning counts.
- PCA reuse, training-only FBCSP fitting, disjoint splits and evaluation-data
  exclusion: these protect the interpretation of results.
- Window ownership, pending validation, validation time, prediction-before-update
  and cooldown: these protect causal ordering and delay accounting.
- Evaluation-label exclusion, PWKNN confidence gating, rejected pseudo-labels
  and duplicate-admission protection: these protect the transductive adaptation
  boundary and knowledge-base update semantics.
- Official labels, channel layouts, dimensions and invalid inputs: these prevent
  silently evaluating a different dataset or protocol.
- False-positive/false-negative metrics and published/computed provenance:
  these protect the evidence presented by the application.
- Routes, CLI interpretation and import compatibility: useful software regressions
  kept separate from scientific claims.

Passing this suite demonstrates the checked software behaviour. It does not
establish reproduction accuracy or comparative detector performance. The next
substantive coverage gap is direct testing of complementary score and residual
monitoring, including calibration isolation and decision rules. The retained
PCA reconstruction checks do not cover the complementary detector itself.
