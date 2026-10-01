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
│   ├── splitting.py                  # shared stratified development indices
│   └── datasets/
│       ├── chowdhury/              # participant metadata, loaders, CSE experiment
│       ├── dataset2a/
│       │   ├── experiment.py         # Dataset 2A trial/expt logic
│       │   ├── development_split.py  # Session-I 70/30 split
│       │   └── development_pipeline.py
│       └── dataset2b/
│           ├── experiment.py
│           ├── development_split.py  # 2019 protocol and 70/30 split
│           ├── development_pipeline.py
│           ├── diagnostics.py
│           └── reference.py
├── adaptation/
│   ├── classifier.py                 # common classifier contracts and linear SVM
│   ├── supervised.py                 # supervised + PWKNN transductive adaptation
│   ├── experiments/                  # real and synthetic bagging comparisons
│   └── ensemble/
│       └── bagging.py                # bootstrap ensemble classifier
├── detection/
│   ├── core.py                       # dataset-agnostic CSE orchestration
│   ├── preprocessing.py              # PCA fit/transform helpers
│   ├── stage1/                    # SD/MSD EWMA and complementary PCA monitoring
│   ├── stage2/                    # Hotelling and K–S validation APIs
│   └── baselines/                 # ICI-CDT comparison
├── reporting/
│   ├── table1.py                     # Table 1 output/comparison
│   └── metrics.py                    # detector evaluation metrics
├── simulation/
│   ├── gaussian.py                   # synthetic abrupt mean-shift streams
│   ├── jumping_mean.py               # paper D2 AR jumping-mean stream
│   └── pca_subspace.py               # score/residual shift scenarios
├── experiments/
│   └── paper2015/
│       └── d2.py                     # D2 generation/detection/evaluation
├── support/
│   └── rag.py                        # local repository/proposal RAG support layer
└── web/
    ├── home.py                       # paper-grounded home-page model
    ├── support.py                    # local Llama support route
    ├── dashboard.py                  # Flask routes and result presentation
    ├── figure1.py                    # real 2A feature-distribution view
    ├── chowdhury.py                  # cohort metadata view
    └── test_results.py               # test-report views
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
- Dataset 2B 2019 protocol: Sessions I–III stratified 70/30 development split; IV–V held-out evaluation; FBCSP fitted on development training only
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
- Diagnostic and sensitivity-analysis utilities, including Dataset 2A validation-only control-limit calibration
- Complementary PCA score/residual Stage-I monitoring on synthetic and real 2A/2B features
- Chowdhury participant metadata loader and prepared-feature CSE experiment interface
- K–S Stage-II and ICI-CDT baseline for the 2015 D2 comparison
- Linear SVM, bagging classifier, supervised adaptation, and separate real/synthetic bagging experiments
- First-step CSE-UAEL transductive adaptation with labelled calibration data, unlabelled evaluation features, RBF-weighted PWKNN pseudo-labels, confidence-gated knowledge-base growth, and separate offline ground-truth scoring
- Dynamic LDA ensemble growth and weighted ensemble classification remain future CSE-UAEL stages
- Paper-aligned D2 jumping-mean generator with repeated-shift truth labels
- Full-stream SD-EWMA/TSSD-EWMA D2 experiment and event-level metrics
- Pytest coverage and GitHub Actions CI for the CSE/EWMA pipeline


## Is there evidence that the feature distribution has changed?

This question corresponds to the data-monitoring role in Diethe et al. (2019),
*Continual Learning in Practice*, Section 4.1. The repository answers it at
several evidence levels; a warning, a statistical confirmation, and a known
ground-truth shift are different observations.

### Implemented evidence chain

1. **Define the representation.** The development pipelines fit FBCSP on
   development-training trials and transform validation/evaluation trials with
   that fitted model. `src/detection/core.py` fits PCA on the supplied training
   features and reuses it on evaluation features. Claims concern this feature
   representation, not every property of the raw EEG.
2. **Screen for unusual observations.** In the PC1 path,
   `src/detection/stage1/sd_ewma.py` computes the prediction error
   `x_t - z_(t-1)` and warns when `x_t <= LCL_t` or `x_t >= UCL_t`.
   The limits use the previous EWMA state and previous error standard
   deviation. A warning is a candidate change, not statistical confirmation.
3. **Validate a candidate.** With `validation_mode="paper_two_sample"`,
   `src/cse_paper_stage_2.py` compares two disjoint samples in retained PCA
   space using a two-sample Hotelling T-squared test. For warning index i and
   window H, the reference is `features[i-H+1:i+1]` and the current sample is
   `features[i+1:i+1+H]`. The implemented decision is `p_value < alpha`.
   This is evidence against equal multivariate means under the test
   assumptions; it is not an omnibus test of every distributional change.
4. **Check information outside PC1.** The separate complementary Stage-I
   experiment in `src/detection/stage1/complementary.py` monitors PC1 prediction
   errors and squared reconstruction residuals, with limits calibrated on
   separate reference data. Its `alarm_source` identifies score, residual,
   both or neither. These alarms are not automatically Stage-II confirmations.
5. **Evaluate evidence against known changes where available.** Synthetic
   experiments measure false alarms, missed changes and delay.
   `src/reporting/metrics.py` includes repeated-shift event matching.
   Real EEG alarms have no supplied ground-truth shift-event labels.

The other Stage-II modes compare different objects. In particular,
`algorithm1_training_reference` compares a current vector to a training
reference; it must not be described as the same equal-window test. The
univariate K-S validation in the 2015 D2 experiment is a separate pathway.

### Interpret the validation status before answering

| Paper two-sample status | Defensible interpretation |
| --- | --- |
| `confirmed` | The implemented test rejects equal means at the configured alpha |
| `rejected` | The warning was not confirmed; this does not establish stationarity |
| `pending_current_window` | More subsequent observations are needed |
| `insufficient_reference_window` | Not enough preceding observations to test |
| `insufficient_degrees_of_freedom` | The window/dimension combination cannot support this test |
| `skipped_nearby_alarm` | No new test was performed because of the alarm-gap rule |

Confirmation becomes available at `validation_time`, after the second
window has arrived, not at `alarm_time`. The requirement
`2*H - d - 1 > 0` is necessary for the two-sample F conversion, but alone
does not guarantee well-conditioned covariance estimation or valid inference.
Repeated, data-selected tests and temporal dependence also require calibration;
a per-test alpha is not a stream-wide false-alarm guarantee. Shrinkage or
regularization can alter the classical reference distribution.

### What the committed results currently show

Snapshot inspected at source commit
`a3271c24b2aefebeaa45ca805536a6a76c17d0e6` (30 September 2026):

| Dataset | Total computed warnings | Total computed confirmations | Subjects with confirmations |
| --- | ---: | ---: | --- |
| 2A | 52 | 3 | A02 |
| 2B | 127 | 6 | B02, B03, B04, B06, B07, B08 |

Source: [committed Table 1 comparison](outputs/metrics/bci_table1_comparison.csv).
These are saved outputs, not a new experiment. The aggregate CSV does not
record validation mode, alpha, window sizes, per-event p-values or all
validation statuses. It therefore supports a statement about the recorded
confirmation counts, but cannot independently establish each event's test
configuration or explain every unconfirmed warning. Do not equate
`CSW - CSV` with the number of statistically rejected warnings.

The [100-seed complementary summary](outputs/complementary_100/summary.csv)
provides controlled synthetic evidence: PC2 shifts have detection rates of
21% for score-only and 100% for residual-only and combined monitoring.
For PC1 shifts the corresponding rates are 35%, 21% and 31%, so the combined
method does not dominate every scenario. The
[configuration](outputs/complementary_100/config.json) uses a 20-observation
detection horizon. Under no change, combined monitoring has a 1.06%
per-observation false-alarm rate but a 26% matched-window alarm rate.
These results demonstrate representation-dependent sensitivity, not real EEG
detection accuracy or universal false-alarm control.

### Export the evidence for an individual CSE run

After obtaining `result = run_cse(..., config=config)` on prepared features,
use the following snippet to preserve the decisions and their settings.
This is an explicit user-run export example, not an automatic runner feature.

```python
from dataclasses import asdict
from pathlib import Path
import json

out = Path("outputs/feature_shift_evidence/run_001")
out.mkdir(parents=True, exist_ok=False)  # use a new name for each run

result.warning_results.to_csv(out / "warnings.csv", index=False)
result.validation_results.to_csv(out / "validations.csv", index=False)
metadata = {
    "config": asdict(config),
    "effective_lambda": float(result.effective_lambda),
    "retained_dimensions": int(result.testing_transformed.shape[1]),
}
(out / "config.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print(result.validation_results["status"].value_counts(dropna=False))
```

Also record the actual source commit, dataset/subject, session roles, split
seed and feature-processing settings with the run. For a paper-two-sample
confirmation, inspect `reference_start_time`, `reference_end_time`,
`current_start_time`, `current_end_time`, `sample_size`, `n_features`,
`hotelling_t_squared`, `p_value` and `validation_time`.

A suitable conclusion is: "The configured monitor produced candidate warnings,
and the specified validation test confirmed a subset in the monitored feature
space." Name the run, representation, comparison windows and test threshold
before making a more specific statistical claim. Neither a confirmation nor a
change in P(X) establishes the covariate-shift condition that P(Y|X) is unchanged.
Classifier harm and the benefit of retraining require separate evaluation.

### Next evidence improvements

- Export event-level decisions and complete run provenance from the standard
  BCI runners, alongside the aggregate counts.
- Report mean differences or a clearly defined effect-size measure alongside
  p-values; p-values do not quantify change magnitude.
- Calibrate the full warning-plus-validation procedure on no-change streams,
  accounting for temporal dependence and repeated testing.
- Link confirmations to subsequent labelled prediction performance before
  concluding that adaptation is needed.

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
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels
```

Dataset 2A now applies its Session-I 70/30 development split before FBCSP fitting. The validation fraction and reproducible split seed are explicit experiment controls:

```bash
python scripts/run_bci_table1_reproduction.py \
  --labels-2a data/raw/bci_competition_iv_2a_labels \
  --session1-validation-fraction 0.30 \
  --session1-split-seed 42
```

Dataset 2B's three 2019 runners use the same named I–III training-pool / IV–V evaluation protocol. Set `--dataset2b-validation-fraction` and `--dataset2b-split-seed` in the combined runner, or `--validation-fraction` and `--split-seed` in either dedicated 2B runner. The validation subset is transformed and exposed for parameter work. Dataset 2A can optionally calibrate the Stage-I control-limit multiplier from its held-out validation errors with `--calibrate-control-limit-from-validation`; this is a repository extension, not a reported paper selection rule. The Table 1 runners otherwise use published lambda overrides and do not select K or T from the validation split. The 70/30 ratio comes from the paper; exact partition membership and seed are unreported. Changing from the previous whole-pool FBCSP fit changes computed results, so compare new counts to old output with that methodological difference in mind.

The separate bagging benchmark retains its stated classifier-comparison protocols: Dataset 2B defaults to I–II fitting with labelled III evaluation, and the optional released-label route uses III for calibration and IV–V for evaluation. It does not call the 2019 CSE-UAEL development split; its reported accuracy is a distinct experiment.

The Stage-II interpretation can be compared explicitly:

```bash
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --validation-mode paper_two_sample
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --validation-mode algorithm1_training_reference
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --validation-mode retrospective_windows
```

PCA retention is an explicit experiment option. Stage I still monitors PC1 only in the univariate Algorithm-1 path:

```bash
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --pca-components 1
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --pca-components 3
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --pca-components 0.95
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --pca-components all
```

Each subject line reports the retained PCA component count and the PC1 explained-variance ratio so PCA behavior can be audited alongside CSW/CSV results.

For a GDF-only diagnostic without the official 2A labels, explicitly set `--dataset2a-evaluation-mode gdf-only`. It retains the unlabelled Session-II cues and must not be interpreted as the paper's 144-trial left/right evaluation. The default paper mode requires `--labels-2a`.

Outputs:

```text
outputs/metrics/bci_table1_reproduction.md
outputs/metrics/bci_table1_comparison.csv
```

`bci_table1_reproduction.md` contains computed subject rows and the mean row in the grouped Dataset 2A / Dataset 2B layout. `bci_table1_comparison.csv` contains published values, computed values, and CSW/CSV differences for calibration and reproducibility analysis.

## Flask visualisation dashboard

The web interface uses `src/web/home.py` for the paper-grounded research overview and `src/web/dashboard.py` for Flask routes and result presentation; root `app.py` remains a stable launch/compatibility entry point.

```bash
python -m pip install -r requirements-dashboard.txt
python scripts/run_bci_table1_reproduction.py --labels-2a data/raw/bci_competition_iv_2a_labels --validation-mode paper_two_sample
python app.py
```

Open `/` for the research home page. It maps the paper lineage, implementation progress, current evidence boundary and links to every analytical page. The original Table 1 dashboard is preserved at `/table-1` and reads `outputs/metrics/bci_table1_comparison.csv` on each request, so new experiment output updates the visualisation without copying values into the web application.

The three source papers behind the 2015, 2018 and 2019 lineage panels are stored under `papers/`. Selecting a paper panel opens the repository PDF through `/papers/<filename>` for an inline browser preview. Bibliographic details and file mappings are documented in `papers/README.md`.

Open `/figure-1` for the real Dataset 2A A07 feature-distribution view (class-aware only when official evaluation labels are installed), `/chowdhury-demographics` for the included participant metadata, and `/tests` for a local JUnit XML report. The `/results/complementary-bci` page reads exported complementary-monitoring traces and synthetic summaries. Its real EEG view reports alarm counts and trajectories without ground-truth shift or detector accuracy claims. The `/results/paper2015-d2/ks-validation` page inspects generated K–S Stage-II evidence.

Open `/results` in the same Flask app to inspect the broader generated-result catalogue. That page groups the BCI Table 1 reproduction, Dataset 2B diagnostics, Dataset 2B control-limit sensitivity, synthetic lambda sensitivity and Raza 2015 D2/Table III outputs, including live CSV previews, tracked result fields and generated figure thumbnails when the corresponding files exist under `outputs/`.

Open `/data-processing` for the protocol atlas. It shows the 2019 Dataset 2A/2B development roles, the separate GDF-only bagging holdout, the 2016 published split, and a paper-to-implementation map. The B01 session counts and baseline results in `data/reference/bci_processing_2026-09-27.json` are a labelled summary of a user-provided run transcript. When available, `outputs/metrics/bagging_bci_real.csv` supplies the classifier chart and `outputs/metrics/bci_2b_table1_results.csv` supplies B01 diagnostic counts. The split illustration retains its transcript provenance; the page does not infer a new split from those result files.

## Local RAG research support layer

The research hub includes a separate local support layer at `/support`. It is
not part of the CSE detector or adaptive-learning model. The page retrieves
evidence from repository source files, documentation, generated metric tables,
local dataset inventories and an optional research-proposal PDF, then sends only
the retrieved context to a locally hosted Ollama model.

The default lightweight model is `llama3.2:1b`. Retrieval uses TF-IDF so the
support layer does not require a hosted embedding API or cloud vector database.
Raw GDF and MAT signal contents are not indexed; for the BCI Competition folders
the retriever records local inventory metadata such as file names and counts.

Install the support dependencies and local model:

```powershell
python -m pip install -r requirements-support.txt
ollama pull llama3.2:1b
```

Point the support layer at the local research proposal without committing the
PDF to the repository:

```powershell
$env:SUPPORT_RAG_PROPOSAL_PATH = "C:\path\to\Research Proposal Haocheng Lin(5).pdf"
python app.py
```

Then open `http://127.0.0.1:5000/support`. The page displays retrieved source
locations alongside every generated answer. If Ollama is not running, retrieval
still completes and the source evidence is shown with a local-model error rather
than falling back to a cloud service.

The support prompt explicitly separates implemented repository behaviour from
proposal intentions, and distinguishes BCI Competition benchmark EEG from
procedurally generated synthetic detector streams. Rebuild the in-memory index
from the page after changing code, generated result files or the configured
proposal.

## Diethe policy laboratory

Open `/results/diethe` to compare never-update, periodic, warning-triggered,
validated-shift and performance-drop policies on an identical synthetic stream.
The page reports rolling accuracy, update events, training size, classifier cost
and full Stage-II evidence, with a JSON download. It is an experiment inspired
by Diethe et al. (2019), not an implementation reported in that paper.

```bash
python -m scripts.run_diethe_experiment --scenario relationship_shift --seed 42
```

The CLI also accepts prepared EEG features and explicit provenance. See
[Diethe experiment protocols and data contract](docs/diethe_experiments.md).
Immediate labels and an expanding training set are explicit assumptions;
rollback, delayed labels and bounded-memory adaptation remain future work.

## First-step unsupervised CSE-UAEL adaptation

`src/adaptation/supervised.py` now contains a separate transductive adaptation
path alongside the supervised 2018-style loop. The learner is fitted from
labelled calibration data but `run_unsupervised_adaptation(...)` does not accept
evaluation ground-truth labels. When the selected adaptation policy triggers,
candidate evaluation observations are pseudo-labelled from the current
knowledge base using an RBF-weighted PWKNN rule. Only pseudo-labels whose
confidence is strictly above the configured threshold are admitted before
append-and-retrain.

The default `update_scope="all_seen"` mirrors the first CSE-UAEL knowledge-base
update idea by considering evaluation observations already seen at a trigger
while preventing an accepted trial from being appended twice. Alternative
`current_trial` and `since_last_update` scopes are retained as explicit
experimental variants.

Ground-truth evaluation labels are kept outside the online learner. Use
`evaluate_unsupervised_adaptation(...)` after the run to measure classifier
accuracy, pseudo-label accuracy and accepted-pseudo-label accuracy without
leaking test truth into adaptation.

This is a first-step reconstruction, not yet the full 2019 CSE-UAEL system.
PWKNN pseudo-labelling and confidence-gated knowledge-base updates are
implemented; dynamic LDA ensemble growth, dynamic weighted ensemble
classification, subject-specific K/threshold selection, and a complete 2A/2B
adaptive reproduction remain future stages.

## Complementary monitoring experiments

The complementary Stage-I experiment monitors the PCA score and the residual outside the retained subspace. Its synthetic stream has known shift scenarios, while the real BCI export supports descriptive alarm inspection only.

```bash
python -m scripts.run_complementary_monitoring --seeds 20
python -m scripts.run_complementary_bci --labels-2a data/raw/bci_competition_iv_2a_labels
```

The first command writes runs, calibration, summaries and scenario traces to `outputs/complementary/`. The second requires downloaded 2A and 2B recordings by default and writes trial traces, calibration and configuration to `outputs/complementary_bci/`. Use `--datasets 2b` for a 2B-only real-data export without 2A evaluation labels. View both through `/results/complementary-bci` after generation.

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

Install test dependencies and run the complete suite:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The test suite covers Dataset 2A trial extraction and 70/30 development processing, FBCSP construction, Stage-II validation, Table 1 formatting, Flask dashboard rendering, PCA parsing, Dataset 2B regression behavior, EWMA/CSE logic, supervised adaptation, PWKNN pseudo-labelling, confidence-gated transductive updates, evaluation-label isolation, and legacy-import compatibility.

Tests are grouped by dataset and component, with separate experiment, web, CLI,
and compatibility checks. See [`tests/README.md`](tests/README.md) for focused
commands, retention criteria and the review of removed or consolidated cases.

## Current research boundary

The repository covers signal processing, feature extraction, CSE warning, Stage-II validation, supervised adaptation, and a first-step PWKNN-driven transductive adaptation path for unlabelled evaluation features. The current pseudo-labelled path keeps evaluation truth outside the learner, applies confidence-gated knowledge-base growth, and supports post-run scoring against hidden labels. These components are still not a complete 2019 CSE-UAEL adaptive ensemble: dynamic LDA ensemble growth, weighted ensemble classification, paper-matched subject-specific PWKNN parameter selection, and an end-to-end 2A/2B adaptive reproduction remain future stages.
