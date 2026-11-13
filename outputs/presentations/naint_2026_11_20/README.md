# NAINT five-minute presentation figures — 20 November 2026

This directory contains **presentation assets** with explicitly separated evidence
provenance. Figures are not claimed to reproduce published counts unless supported
by controlled, event-level run evidence.

## Files already committed (no raw EEG files needed)

- [Synthetic D2 · EWMA warnings and K–S completion](synthetic_d2/05_synthetic_d2_ewma_ks_timeline.svg):
  last 1,000 paper-indexed observations from the **saved** 2015 D2 trace.
  The figure uses downsampled displayed curves (every 2–3 observations) but
  retains *all* plotted Stage-I warning points and Stage-II K–S completion
  markers. This is **not BCI Hotelling T²** and not a new experiment.
- [BCI IV 2B B01 · archived Stage-I warning events only](bci_2b/B01/06_b01_archived_warning_events_only.svg):
  all 17 saved alarm rows, not the full signal. The historical B01 Stage-II
  confirmation disagreement (0 versus 5; published 10) is **unresolved**.
  Do not join those aggregates to individual points on this figure.

## Generate four real BCI IV 2A figures and an independent D2 comparison

Place the official 2A GDF recordings and their released evaluation-label
MAT files in the existing local `data/raw/` subdirectories (these files are
intentionally .gitignored). Run from the repository root:

```powershell
python scripts/export_naint_slide_figures.py --source all --subject 1 \
  --data-2a data/raw/bci_competition_iv_2a \
  --labels-2a data/raw/bci_competition_iv_2a_labels
```

Use PowerShell's backtick for a multiline command, or run the following
single-line equivalent:

```powershell
python scripts/export_naint_slide_figures.py --source all --subject 1 --data-2a data/raw/bci_competition_iv_2a --labels-2a data/raw/bci_competition_iv_2a_labels
```

If raw EEG is unavailable but you want to export the **archived synthetic D2**
figure to PNG/SVG using the tracked CSV inputs:

```powershell
python scripts/export_naint_slide_figures.py --source d2
```

Outputs under `outputs/presentations/naint_2026_11_20/`:

| Path suffix | Description | Evidence |
| --- | --- | --- |
| `bci_2a/A01/01_fbcsp_chronological_feature.{svg,png}` | Development versus evaluation feature timeline | Real 2A GDF/MAT |
| `bci_2a/A01/02_stage1_ewma_warning_trace.{svg,png}` | PC1 observation, control limits, warnings | Real 2A GDF/MAT |
| `bci_2a/A01/03_stage2_validation_timeline.{svg,png}` | Warning versus validation-completion timing and status | Real 2A GDF/MAT |
| `bci_2a/A01/04_fbcsp_window_shift_heatmap.{svg,png}` | Window-standardised FBCSP differences | Real 2A GDF/MAT |
| `synthetic_d2/05_synthetic_d2_ewma_ks_timeline.{svg,png}` | Stage-I D2 and independent K–S validation | Archived D2 CSVs |

Subject selection uses `--subject 1..9`. Each local export creates
`manifest.json` recording its inputs, configuration and evidence status.
For reproducibility, record the Git commit, local dataset checksums and
environment alongside the manifest before using the plots in a manuscript.

## Important interpretation boundaries

1. The **2A** charts use the same development-fitted FBCSP rows as the existing
   `/data-distributions` page; detector recomputation uses its exploratory
   configuration (PCA=2, lambda=0.2, H=10, shrinkage Hotelling, alpha=0.05).
   The outputs are **not** a published-count reproduction. The 70/30
   development split is stratified, not a contiguous online stream.
2. The Stage-I EWMA graph shows recorded input observations and control limits;
   a threshold crossing is only a *warning*.
3. Stage-II validation may finish `H` observations after its warning.
   `pending`, `skipped` and `ineligible` do not mean hypothesis rejection.
   A small p-value is evidence against equal window means under test assumptions,
   **not proof** that accuracy decreased or retraining improves performance.
4. The heatmap is exploratory and does not test statistical significance.
5. The D2 chart is **synthetic** and uses a **K–S** validator; the 2A chart
   uses **Hotelling T²**. Never combine event counts from those different runs.
6. **No synthetic data are substituted for missing real EEG recordings.** The
   2A exporter fails clearly until the required GDF/MAT files are available.

Implementation: [`scripts/export_naint_slide_figures.py`](../../../scripts/export_naint_slide_figures.py).
Regression checks: [`tests/web/test_naint_slide_figures.py`](../../../tests/web/test_naint_slide_figures.py).

To validate the exporter:

```powershell
python -m pytest tests/web/test_naint_slide_figures.py tests/web/test_bci_shift_views.py -q
```
