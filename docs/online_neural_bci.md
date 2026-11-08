# Incremental neural classifier for BCI 2A/2B feature replay

This is an **experimental** alternative classifier, not a reproduction of
Zhang et al.'s RWOS-ELM or of Raza et al.'s reported CSE-UAEL results.

## Integration and evidence boundaries

- `src/adaptation/neural.py` implements a compact scikit-learn MLP using
  `partial_fit` on accepted labelled mini-batches. Calibration uses a
  training-fitted scaler, which remains frozen during evaluation.
- `src/adaptation/classifier.py::IncrementalLearner` defines the general
  `fit/predict/update/training_size` contract.
- An `append_and_retrain` compatibility method delegates to `update`,
  allowing existing `OnlineAdaptiveBCI` policies and delayed-label logic
  to remain unchanged.
- EWMA and Hotelling operate on the **same frozen PCA monitoring pathway**;
  neural updates affect only the classifier.
- EEG features are precomputed using offline zero-phase FBCSP; replay is
  chronological at the **feature level**, not causal raw-EEG acquisition.
- This implementation requires ground-truth labels after their configured
  release times. The separate transductive PWKNN experiment is not yet
  connected to this CLI.

## Run on BCI Competition IV

The original SVM CLI and output paths remain the default. To run MLP:

```powershell
python scripts/run_online_bci.py --dataset 2a --subject 1 --classifier mlp --policy validated --labels-2a data/raw/bci_competition_iv_2a_labels
python scripts/run_online_bci.py --dataset 2b --subject 1 --classifier mlp --policy validated --labels-2b data/raw/bci_competition_iv_2b_labels
```

The 2B label path must point to official class-label MAT files. The
evaluation labels may be used for post-run scoring, but the controller
only releases them for learning after the configured delay. Set
`--label-delay -1` to prohibit their use for training.

Run policy-matched comparators by varying `--classifier svm|mlp`
and `--policy never|periodic|warning|validated`; keep seed, feature
settings, PCA dimensions, windows and subject/session chronology fixed.

MLP results go under
`outputs/online/<dataset>/subject_<ID>/mlp/<policy>/`;
SVM results retain their existing paths. Inspect `trials.csv`,
`stage2.csv`, `updates.csv` and `summary.csv`. These runs alone do not
establish neural superiority, numerical CSE reproduction, causal
streaming or reduced energy expenditure.

## Tests

```bash
python -m pytest tests/adaptation/test_online_neural.py tests/online/test_adaptive_pipeline.py -q
```

Further controlled work: multi-subject comparisons, repeated seeds,
per-policy update cost and retention metrics, and separate evaluation
of adaptive neural EEG representations.
