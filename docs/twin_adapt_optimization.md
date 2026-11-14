# Cost-aware TWIN-ADAPT optimisation for BCI Competition IV

This is a **development-only configuration search** layered on the operational
TWIN-ADAPT accuracy-triggered controller. The original fixed-parameter runner
(`scripts/run_twin_adapt_bci.py`) and archived result directories are unchanged.

## Execution

From the repository root, with the official BCI Competition IV GDF files
and evaluation-label MAT files in the existing `data/raw` structure:

```powershell
python scripts/run_twin_adapt_optimization.py --dataset 2a --subject 1 --batch-size 10 --budgets 20 40 --candidates 6 --seed 42
python scripts/run_twin_adapt_optimization.py --dataset 2b --subject 1 --batch-size 40 --budgets 40 80 --candidates 6 --seed 42
```

These commands define **example budgets**, not established optimal settings.
A budget must fit the available development-validation examples. For 2A, the
left/right subset and 70/30 development split make a 320-trial tuning budget
invalid. Reduce batch size and budgets only while preserving controller
eligibility (`0 < win_size < batch_size`). A short validation stream may have
too few events to reliably rank adaptation policies.

## Data separation and what is (not) causal

- 2A: Session I is divided into training and validation. Session II remains
  exclusively the final held-out evaluation session.
- 2B: Sessions I–III supply development training and validation; Sessions IV–V
  remain held out.
- The existing data-preparation functions use a **seeded stratified 70/30**
  split. Validation rows are sorted by their original trial index, but this
  *does not turn a scattered holdout into a contiguous session*. This is a
  feature-level **offline replay**, not real-time EEG acquisition. A future
  contiguous temporal holdout is a separate protocol requiring its own manifest.
- FBCSP is fitted only on development-training signals. The same fitted
  transform is used for validation and final evaluation. The final classifier
  also starts on the development-training observations, not a newly refitted
  full-development FBCSP model.
- The optimiser never receives final evaluation arrays; the final replay is
  performed **after** candidate selection. Validation feedback is revealed
  to the controller only at the configured release batch.

## Candidate search and scoring

The sample space contains the threshold ratios `gamma`/`delta` and
monitoring/window-capacity choices `win_size`/`win_size_max`. Invalid
`TwinAdaptConfig` combinations are omitted. Candidates are sampled without
replacement under `seed`, then evaluated on progressively longer
development-validation prefixes. Each trial starts from a fresh SVM trained on
the same training partition. Exact score ties preserve seeded candidate order.

```text
J = validation_accuracy
    - alpha * (number_of_updates / validation_trials)
    - beta * (cumulative_SVM_retraining_seconds / reference_seconds)
```

Default weights `alpha=0.2`, `beta=0.02`, `reference_seconds=1.0` are
**illustrative research coefficients**. They must be declared in any reported
experiment and subjected to sensitivity analysis, including `alpha=beta=0`.
This objective counts retraining time, **not** total configuration-search
expense. Candidate replay elapsed seconds and overall search elapsed seconds
are recorded separately. Runtime depends on hardware/load and should not be
treated as bitwise deterministic.

## Outputs

The independent output root is
`outputs/experiments/twin_adapt_optimization/<dataset>/subject_<id>/<run_id>/`.

- `search_space.json`, `objective_config.json`, `best_config.json`
- `development_split.json`: partition sizes, budgets and protocol caveats
- `candidate_history.csv`: every evaluation's parameters, budget, accuracy,
  update rate, retraining time, replay elapsed time, score and promotion
- `optimization_summary.json`: final evaluation metrics and search totals
- `final_evaluation/`: trial predictions, accuracy windows, transitions,
  adaptive buffers, updates and summary
- `figures/`: run-scoped SVG/PNG visualisations of candidate scores, stage-level
  prediction diversity (when available), accuracy/threshold timelines, and
  recorded update events, with a `manifest.json` pointing to source CSVs.

The runner now exports figures immediately after saving the CSV evidence.
To regenerate plots for a previously archived run (including the B01 run on
14 November 2026), use:

```powershell
python scripts/export_twin_adapt_run_figures.py --run-dir outputs/experiments/twin_adapt_optimization/2b/subject_01/20261114T145309426597Z_6535b2a3
```

The exporter writes only to the run's `figures/` subdirectory and retains
all original JSON/CSV files and their paths. Charts are derived solely from
saved run evidence. The stage-level prediction-diversity figure is not a
pairwise comparison matrix; it uses the current stage-wide summary field.
Presentation-specific exports continue to live under
`outputs/presentations/<event>/`, separate from experiment evidence.

Do **not** compare these results to historical fixed-parameter outputs as a
controlled performance improvement unless subjects, preprocessing splits,
classifier starts, label release protocol and evaluation conditions match.

## Tests

```powershell
python -m pytest tests/adaptation/test_twin_adapt_optimization.py -q
python -m pytest tests/adaptation/test_twin_adapt_integration.py -q
```

Tests using generated feature matrices validate software wiring; they do not
establish real EEG performance or reproduce paper-reported CSW/CSV counts.
