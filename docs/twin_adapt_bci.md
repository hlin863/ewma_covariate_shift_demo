# TWIN-ADAPT (Gupta et al., 2024) on BCI Competition IV 2A/2B

## Runtime path

`scripts/run_twin_adapt_bci.py` → `src/adaptation/experiments/twin_adapt_bci.py::run_bci_experiment` → existing `scripts.run_online_bci.prepare` (official 2A/2B GDF + MAT label protocol, fitted FBCSP) → `LinearSVMClassifier.fit` (development only) → `make_bci_dataflow` (chronological batches) → `src/adaptation/twin_adapt/controller.py::adapt_model` → immutable CSV/JSON evidence → `/results/twin-adapt`.

The Flask endpoint calls the **same experiment orchestrator** as the CLI for POST requests; GET reads recorded runs, never invents output. The existing online-learning page, research catalogue and Results navigation link to it. Raw data directories are not committed.

## Local commands

```powershell
python -m scripts.run_twin_adapt_bci --dataset 2a --subject 1 --batch-size 40 --win-size 10 --win-size-max 120 --gamma 0.95 --delta 0.85 --label-delay 0
python -m scripts.run_twin_adapt_bci --dataset 2b --subject 1 --batch-size 40 --win-size 10 --win-size-max 120 --gamma 0.95 --delta 0.85 --label-delay 0
python -m pytest tests/adaptation/test_twin_adapt_integration.py -q
python app.py
```

The 2A experiment requires `data/raw/bci_competition_iv_2a` and `data/raw/bci_competition_iv_2a_labels`. The 2B experiment requires `data/raw/bci_competition_iv_2b` and official evaluation MAT labels in `data/raw/bci_competition_iv_2b_labels`. The `prepare` helper validates label alignment and preserves the repository's 2A Session-I/II and 2B Sessions-I–III/IV–V split. All paths are relative to the repository working directory; use `--data-root` or `BCI_RAW_DATA_ROOT` app configuration for another location.

## Results and provenance

Each run is saved to `outputs/experiments/twin_adapt_bci/<dataset>/subject_<NN>/<UTC-run-id>/`:

- `trials.csv`: original trial indices, predictions, reference labels for post-run scoring and model versions.
- `accuracy_windows.csv`: PrevAcc/CurrAcc, gamma/delta, calculated thresholds and actions at feedback release.
- `state_transitions.csv`: state changes and reasons.
- `adaptive_windows.csv`: buffer length and collection/discard outcomes.
- `updates.csv`: selected trial IDs, training size, retraining time, classifier version.
- `summary.json` and `summary.csv`: metrics, parameter snapshot, feedback semantics, commit and dataset protocol.

A stable `latest_summary.csv` in the experiment root is updated so the results catalogue knows that real run evidence exists; historical per-run folders are never overwritten.

## Research semantics

This is **feature-level offline replay** from development-fitted FBCSP, not causal raw-EEG acquisition. Predictions are made *before* feedback release or an adaptation on their batch. `label_delay` is measured in **batches**, whereas `win_size_max` is measured in **trials**. `label_delay=-1` withholds all online feedback; post-run accuracy can still be scored against the separate official evaluation labels.

This code implements a documented **operational interpretation** of Gupta et al. (2024), §5.1 Algorithm 1. The paper's State-2 `f` indexing and adaptive-window semantics are ambiguous; the controller uses the first released post-update batch as reference and bounds the buffer in trials. This is **not** the paper's LightGBM implementation: the baseline uses the repository's existing linear SVM.

Accuracy-degradation warnings must not be conflated with EWMA Stage-I statistical warnings or Hotelling Stage-II confirmations. Historical Table-1 and B01/D2 discrepancies remain separate research questions.

## Web interface

Start `python app.py`, open `/results/twin-adapt`, select dataset, subject, batch/window parameters, gamma/delta and label delay, then submit. The server executes the exact CLI backend using local GDF/MAT files and shows saved run accuracy, window history, state transitions, updates and downloadable CSV/JSON evidence. Missing local raw data are reported as an error rather than replaced with synthetic values.

The existing `/results/online-learning` page links to this experiment for comparison. Do not compare policy results numerically until preprocessing, seeds, subject splits, label-release regime, batch sizes, and classifier baselines are aligned.
