# Diethe policy laboratory

Open `/results/diethe` from the research home page or results navigation.
This is an experimental application of Diethe et al. (2019), *Continual Learning
in Practice*, Sections 3–5. The paper proposes a reference architecture; it does
not supply the five-policy algorithm implemented here.

## Questions and responsibilities

| Paper question | Implementation | Remaining boundary |
| --- | --- | --- |
| Has the feature distribution changed? (§4.1) | Existing CSE detector, warnings and complete Stage-II records | Mean-shift evidence in retained PCA space; not all distribution changes |
| Did prediction quality change? (§4.2) | Per-trial correctness and full-window rolling accuracy | Immediate trustworthy labels assumed |
| When should a model update? (§5.2) | Never, periodic, warning, validated, performance-drop rules | No learned optimal policy or expected-utility claim |
| Which data enter training? (§5, horizon) | Current trial or all trials since last update | Both append to historical data; no forgetting or bounded reservoir |
| What does adaptation cost? (§5) | Initial fit, cumulative retraining, prediction timing, update counts and final training size | Measured wall time varies; memory/energy not measured |
| Can a decision be audited? (§3.3) | Configuration, validation windows, labels/predictions, versions and update events | Versions do not persist models; no rollback or acceptance gate |

## Controlled synthetic scenarios

- `none`: fixed input distribution and label rule.
- `feature_shift`: the first input feature's mean changes halfway through the
  evaluation stream. Labels always use `x0 + 0.5*x1 > 0`; P(Y|X) is unchanged.
  Class proportions may change. The shift need not harm classification.
- `relationship_shift`: the same input stream as `none`, with the label rule
  reversed halfway through. This changes P(Y|X), not P(X); an input detector may
  miss it. This is not pure covariate shift.

Synthetic data are generated feature vectors, not simulated raw EEG. Training
(120 observations), validation (80) and evaluation are separate draws. The
initial linear SVM's validation accuracy is frozen as the performance reference.
The feature representation and detector are fixed across all five policies.

## Temporal protocol

Each trial is predicted before its label is used. Its label is then available
for scoring and possible updating. The performance rule waits for a full rolling
window and requires accuracy strictly below `reference_accuracy - accuracy_drop`,
with at least `interval` trials since the last update. This is a heuristic
threshold, not a calibrated hypothesis test. It can respond without a feature
warning. The rolling window can span model versions after an update.

The validated policy acts at `validation_time`. The default detector compares
20 observations ending at a warning to 20 subsequent observations. Validation
is unavailable before the latter arrive. Pending, skipped, rejected and
ineligible records remain separate. An update at the final trial still costs
computation but cannot improve a future prediction within that run.

Initial model fitting and updating use the same linear SVM contract as the
existing supervised pathway. Original policy defaults remain unchanged.
The performance context and timing fields are additive.

## Running and preserving experiments

```bash
python -m scripts.run_diethe_experiment --scenario feature_shift --seed 42
python -m scripts.run_diethe_experiment --scenario relationship_shift --seed 42
python -m scripts.run_diethe_experiment --scenario none --seed 42
```

Optional controls: `--trials`, `--magnitude`, `--interval`,
`--performance-window`, `--accuracy-drop`, `--update-scope`, `--output`.
The interactive page bounds synthetic evaluations to 80–500 trials. It performs
no automatic run on GET. The JSON download contains the displayed run, not a
rerun with new timing measurements.

The CLI creates a new `outputs/diethe/<UTC timestamp>/` directory. An existing
output directory is rejected to prevent accidental overwrites. It writes:

- `experiment.json`: full run, settings, source commit if available and provenance.
- `summary.csv`: accuracy, gain versus never-update, update count, timing and size.
- `warnings.csv` and `validations.csv`: complete detector evidence.
- `<policy>_trials.csv` and `<policy>_updates.csv`: predictions and update history.

Use several seeds before interpreting performance differences. Detection and
classification conclusions are distinct. Timing is shared for the detector and
reported separately from policy-specific classifier costs.

## Prepared EEG features

The same comparison API accepts NumPy feature matrices. For the CLI, prepare
an NPZ with these exact keys (pickle is disabled):

| Key | Shape / meaning |
| --- | --- |
| `training_features` | n_train × d; preprocessing fitted on training only |
| `training_labels` | n_train; binary 0/1 with both classes present |
| `validation_features` | n_validation × d; disjoint pre-evaluation holdout |
| `validation_labels` | n_validation; binary 0/1 |
| `evaluation_features` | n_evaluation × d; chronological evaluation stream |
| `evaluation_labels` | n_evaluation; true binary labels revealed after prediction |
| `evaluation_times` | n_evaluation; strictly increasing integer trial indices |

Supply a nonempty provenance JSON describing dataset, subject, sessions, split
seed, feature settings and how holdout separation was enforced:

```bash
python -m scripts.run_diethe_experiment --features prepared.npz --provenance provenance.json
```

The code validates shapes, finite features, binary labels and ordered times; it
cannot infer overlap or data leakage from arbitrary prepared arrays. The caller
must ensure distinct splits and training-only preprocessing. Synthetic scenario,
trial-count and magnitude settings do not alter NPZ data. Actual evaluation size
is recorded separately; no ground-truth change index is assumed for EEG.
The Python `run_policy_comparison` API also accepts an explicit `CSEConfig`.
This does not automatically integrate raw 2A/2B loading or reproduce CSE-UAEL's
pseudo-labelling and dynamic ensemble growth.

## Files and verification

- `src/adaptation/experiments/diethe.py`: comparison and exports.
- `src/adaptation/policies.py`: additional performance-drop rule.
- `src/adaptation/supervised.py`: timing and rolling-performance context.
- `src/web/diethe.py`: bounded synthetic-run Flask blueprint.
- `templates/diethe.html`, `static/diethe.css`, `static/diethe.js`: page and charts.
- `scripts/run_diethe_experiment.py`: synthetic and prepared-feature CLI.

Run the targeted checks with:

```bash
python -m pytest tests/adaptation/test_diethe.py tests/reporting/test_diethe_page.py -q
```
