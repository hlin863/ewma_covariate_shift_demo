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

## Real BCI IV 2A/2B slide-12 evidence exports (18 November 2026)

Use the exact development-fitted dataset paths rather than making synthetic BCI
plots from mock values. Neither the official GDF recordings nor evaluation-label
MAT archives are committed to GitHub. The following commands require those
inputs on the executing machine:

```sh
python -m scripts.export_kl_bci_figures --dataset 2a --subject 1 \
  --data-2a data/raw/bci_competition_iv_2a \
  --labels-2a data/raw/bci_competition_iv_2a_labels

python -m scripts.export_kl_bci_figures --dataset 2b --subject 1 \
  --data-2b data/raw/bci_competition_iv_2b
```

Dataset 2A uses the **existing** `_load_dataset_2a_feature_timelines` pathway:
official left/right Session-II labels are needed to select trial membership.
KL itself still uses only FBCSP/PCA observations, not evaluation labels.

Dataset 2B reuses the **Raza-2019 development split**, with Sessions I–III for
labelled development and IV–V as an unlabelled evaluation stream. The 2B
run uses the same explicitly exploratory PCA2/H10/EWMA lambda 0.2 settings as
the 2A EDA inspector, *not* the subject-specific published lambda or the
PCA3/H10 Table-1 reproduction. The 2B dataflow is indexed by evaluation
trial ordinal; no false cue-onset-second metadata is supplied.

Every invocation exports to an input/configuration/revision-fingerprinted folder:

```text
outputs/experiments/kl_stage2/
  2a/A01/<run-key>/
  2b/B01/<run-key>/
    manifest.json
    kl_events.csv
    kl_events.json
    01_kl_event_magnitude.png (+ .svg)
    02_kl_per_pca_component.png (+ .svg)
    03_kl_before_after_probabilities.png (+ .svg)
    04_kl_slide12_composite.png (+ .svg)
```

The four figures are generated **only when an eligible event exists**.
Otherwise a manifest and event logs are written with
`status=no_eligible_windows`; no zero-valued dummy dots or misleading
synthetic plots are created. The default selected event is the eligible
warning with the greatest forward marginal KL, explicitly *for illustration*
and not as a tuned detector/ground-truth shift declaration. `--selected-alarm`
may select a different eligible warning consistently for B/C and the composite.

The **A** event plot displays each eligible warning with forward/reverse KL;
the **B** plot decomposes the chosen warning by retained PCA component;
the **C** plot shows before/after probabilities for the same event and its
highest-forward-KL component. The fourth image arranges A/B/C as one
slide-ready composite. Moving the chart legend above the plot eliminates
the overlapping scale annotation observed in the original browser image.

To inspect saved results, visit `/results/kl-stage2`. The BCI exploratory
page `/data-distributions` links to that gallery, and `/results`
discovers the latest manifest-backed export. The browser never silently
generates figures or claims a run exists unless its artifacts are present.

Saved KL values are **marginal descriptive divergences**, and the selected
observation still needs the post-warning comparison window to arrive.
The scores do not supersede any Hotelling outcome, alter reproduction
counts, confirm clinical concept drift, or initiate SVM retraining.
