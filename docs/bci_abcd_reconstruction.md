# BCI Competition IV reconstruction-loss experiment (Heyden et al. 2024)

This experiment **extends the repository's existing BCI pipeline**, rather than replacing the synthetic ABCD learning lab or modifying the 2019 CSE Table 1 reproduction.

## Data and pathway

- Dataset 2A: existing Session-I/T training/development split; official Session-II/E labels are required by the existing left/right FBCSP extractor. Evaluation labels control trial selection only and are **not used to fit PCA or compute loss**.
- Dataset 2B: existing Sessions I–III development protocol and held-out Sessions IV–V evaluation.
- Both: existing cue-aligned 3-second EEG extraction -> FBCSP trained only on development training trials -> frozen PCA fitted on training FBCSP features -> held-out feature reconstruction -> per-trial squared residual Q and mean reconstruction loss L=Q/d.
- Optional batch candidate scan reuses the existing annotated Bernstein Eq. (5)/(8) teaching baseline. No known shift-point ground truth is asserted merely because a session transition occurred.
- Validation features remain outside the fit in this first monitoring-only extension; no classifier updating or retraining occurs.

## Commands

```powershell
python -m pytest tests/bci/pipelines/test_reconstruction_loss.py -q
python -m scripts.run_bci_reconstruction_loss --dataset 2b --subjects 1 --data-2b data/raw/bci_competition_iv_2b --pca-components 1
python -m scripts.run_bci_reconstruction_loss --dataset 2a --subjects 1 --data-2a data/raw/bci_competition_iv_2a --labels-2a data/raw/bci_competition_iv_2a_labels --pca-components 1
```

The runner writes per-trial CSV traces, per-feature residual comparisons and a provenance manifest beneath `outputs/experiments/bci_reconstruction_loss`. The optional `--bernstein-M 1` is **only** a batch educational score and must not be presented as a calibrated p-value.

## Interpretation and limitations

High evaluation loss relative to training may indicate distributional mismatch in directions poorly reconstructed by frozen PCA, but it can also reflect preprocessing, noise or session effects. Low loss does not prove stationarity: changes within the retained PCA subspace may be invisible. Original EEG class labels are not distribution-change annotations. This output is neither a Heyden synthetic benchmark reproduction nor a new Table 1 result.

For genuinely online operation, implement sequentially valid scaling, adaptive ABCD windows, model resets, null calibration and independently established change-point truth. Reuse `src/detection/stage1/complementary.py` for subsequent score-versus-residual comparisons.
