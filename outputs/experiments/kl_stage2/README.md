# KL Stage-II real-BCI output archive

This folder is the canonical output root for the optional marginal KL diagnostic.
**Executable Python code lives in `scripts/`, `src/`, and `tests/`, not here.**
The figures are derived from the same training-fitted PCA feature-replay stream
used by the unchanged Hotelling validation.

## Generate subject-specific outputs

With official BCI Competition IV GDF recordings installed locally, run:

```powershell
python -m scripts.export_kl_bci_batch --dataset both --subjects 1
```

For individual runs:

```powershell
python -m scripts.export_kl_bci_figures --dataset 2a --subject 1 --labels-2a data/raw/bci_competition_iv_2a_labels
python -m scripts.export_kl_bci_figures --dataset 2b --subject 1 --data-2b data/raw/bci_competition_iv_2b
```

Each completed eligible run saves:

```text
outputs/experiments/kl_stage2/
  batch_report_<UTC>.json            # batch attempts and missing-input reasons
  2a/A01/<config-input-revision-key>/
  2b/B01/<config-input-revision-key>/
    manifest.json                     # input, parameter, selected-event and revision provenance
    kl_events.csv
    kl_events.json
    01_kl_event_magnitude.png / .svg
    02_kl_per_pca_component.png / .svg
    03_kl_before_after_probabilities.png / .svg
    04_kl_slide12_composite.png / .svg
```

Figures are produced **only** for complete, eligible before/after windows.
A run with no eligible events contains its manifest and event logs, but no
fabricated zero-divergence plots. If files are missing, the batch writes an
explicit failure record; it never manufactures EEG data. The application
reads only verified manifest-backed outputs at `/results/kl-stage2`.

**Interpretation:** mean marginal histogram KL is descriptive, not a
joint-distribution KL, calibrated p-value, Hotelling confirmation, or
classifier-update trigger. The displayed selected warning is chosen for
illustration, not as a ground-truth change point. These exploratory
settings are distinct from Raza Table-1 reproduction. See
[`docs/kl_stage2_diagnostics.md`](../../../docs/kl_stage2_diagnostics.md).
