# Optional KL-divergence diagnostics beside Stage-II Hotelling validation

## Purpose and location

Use `src/detection/stage2/kl_diagnostics.py` to examine **marginal** distributional
differences after EWMA candidate warnings. This is a *separate descriptive
diagnostic*, not a replacement for Raza et al. (2019) Stage-II Hotelling T²,
a joint density estimator, or a significance test. It implements no new
`confirmed_shift` decisions and triggers no SVM updates.

The baseline `src/detection/core.py::run_cse` is deliberately unchanged,
preserving the existing reproduction counts and experiments.

## Input/output interface

The input is the **same retained, training-fitted PCA space** used by Stage-II,
not raw EEG voltages or an independently refitted PCA representation.

```python
from src.detection import run_cse, CSEConfig
from src.detection.stage2.kl_diagnostics import (
    KLDiagnosticConfig, attach_kl_diagnostics,
)

result = run_cse(x_train, x_test, trial_times, CSEConfig(
    validation_mode="paper_two_sample",
    validation_before_size=10,
    validation_after_size=10,
))
diagnostic_frame = attach_kl_diagnostics(
    result.validation_results,
    result.testing_transformed,
    trial_times,
    result.pca_result.training_transformed,
    config=KLDiagnosticConfig(n_bins=8, pseudocount=0.5),
)
diagnostic_frame.to_csv("outputs/kl_stage2_diagnostics.csv", index=False)
```

Only `confirmed` and `rejected` Hotelling rows with two complete windows
receive KL scores; pending, skipped, or insufficient rows retain a reason.
The two disjoint windows for warning position `i`, length `H`, are
`X[i-H+1:i+1]` and `X[i+1:i+H+1]`. No evaluation labels are used.

Bins are set on the **calibration PCA data**; outer bins are open-ended to
include evaluation outliers. Positive pseudocount smoothing avoids infinite
plug-in values. The reported score is the **mean of per-feature**
`KL(P_before || P_after)` estimates, with a reverse-direction counterpart.
The per-feature vector is retained in `kl_per_feature` (list object; use JSON
when a machine-readable persisted vector is needed). This mean is not
`KL(P_joint || Q_joint)`, and can miss cross-feature dependency shifts.

## Interpretation and research boundaries

A positive KL estimate is *not* statistical evidence of a significant shift:
finite-sample histograms are noisy. Do not use an arbitrary KL threshold, or
treat raw KL magnitude as a p-value. Windows are selected after EWMA warnings,
so significance must be calibrated under the entire selection procedure.
For temporal EEG streams, IID permutations may invalidate false-positive
guarantees; consider block/bootstrap or stationary-control calibration.

For a benchmark: compare no-change versus known mean/variance shifts; sweep
window length, bin count and pseudocount on development data; evaluate against
existing Hotelling/MMD on matched windows and seeds; report detection delay,
false-alarm rate and adaptive SVM costs separately. Use train/validation/evaluation
splits to avoid tuning on held-out data.

## Research provenance

- Souza et al. (2020): supervised versus unsupervised drift detection,
  temporal stream considerations and window choice.
- Raza et al. (2019): unchanged EWMA warning and Hotelling validation baseline.
- van den Burg and Williams (2022): evaluate change locations and false alarms,
  not just diagnostic scores.
- Jayaram and Barachant (2018): matched BCI preprocessing and evaluation splits.

**Status:** new opt-in exploratory diagnostic. No claim of reproduced published
KL scores, improved BCI detection, improved classification, or energy savings.
