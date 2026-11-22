# Research outputs: evidence, figures, and presentations

The `outputs/` directory is a versioned evidence archive. **Do not move or
rewrite old run CSV/JSON files**: the web results page, manuscript provenance,
and prior presentation material may reference their exact paths.

## Directory conventions

| Location | Content |
| --- | --- |
| `experiments/twin_adapt_bci/<dataset>/subject_<NN>/<run_id>/` | Fixed-configuration trial evidence and run-scoped figures |
| `experiments/twin_adapt_optimization/<dataset>/subject_<NN>/<run_id>/` | Candidate histories, run configuration, held-out evaluation and figures |
| `metrics/` | Detector/reproduction metrics, preserved in existing structures |
| `presentations/<event>/` | Slide and poster figures with their own source manifests |

Each new TWIN-ADAPT run has a `figures/` subfolder containing vector `*.svg`
and raster `*.png` assets, plus `manifest.json`. The manifest identifies the
exact source CSVs used for every graph. Figures are **derived artifacts**, not
additional experiments; the source run remains the authoritative evidence.

An optimised run retains these familiar source files:

```text
experiments/twin_adapt_optimization/2b/subject_01/<run_id>/
    candidate_history.csv
    search_space.json
    objective_config.json
    best_config.json
    development_split.json
    optimization_summary.json
    final_evaluation/
        trials.csv
        accuracy_windows.csv
        adaptive_windows.csv
        state_transitions.csv
        updates.csv
        summary.json
    figures/
        accuracy_windows.svg / .png
        update_events.svg / .png
        candidate_scores.svg / .png
        prediction_diversity.svg / .png  # when diagnostic field exists
        manifest.json
```

## November 2026 provenance additions

The latest research results include a **separate opt-in KL Stage-II diagnostic gallery** and a **matched BCI adaptation comparison runner**. Neither rewrites the historical 2015/2019 reproduction archive.

| Evidence family | Entry point | What is supported | What is not implied |
| --- | --- | --- | --- |
| Published-versus-computed BCI Table 1 | [comparison CSV](metrics/bci_table1_comparison.csv) and [baseline audit](baseline_2026_10_12/README.md) | Historical warning/confirmation aggregates and B01 conflicting archived records | A configuration-locked numerical reproduction or a resolved Stage-II event trace |
| BCI KL diagnostics | [export runner](../scripts/export_kl_bci_figures.py) and [method/manifest guide](../docs/kl_stage2_diagnostics.md) | Configuration-fingerprinted `experiments/kl_stage2/<dataset>/<subject>/<run-key>/` event CSV/JSON, manifest, eligible-event PNG/SVG figure panels and gallery at `/results/kl-stage2` | Hotelling confirmation, classifier improvement or simulated EEG presented as measured results |
| Matched BCI adaptation | [comparison runner](../scripts/run_matched_bci_comparison.py) and [implementation](../src/adaptation/experiments/matched_bci_comparison.py) | Common-runner framework for policy comparisons | Completed multi-seed causal comparison without checking archived configuration and run output |
| NAINT presentation exports | [presentation figure exporter](../scripts/export_naint_slide_figures.py) | Presentation-ready derived visualisations and associated source context | A substitute for experiment-level JSON/CSV evidence |

For KL experiments the official raw EEG files remain local, not committed. The 2B exploratory KL timeline is indexed by trial ordinal rather than reconstructed cue-onset seconds. If no eligible Stage-II comparison exists, only manifests and event records are saved; **do not draw zero-valued fake events**. The source guide specifies the exact required inputs and the per-run output contract.

**Evidence boundary:** retain immutable historical outputs. Published B01 warning/confirmation counts of **14/10** do not match the archived **17/0** cross-subject and **17/5** B01-only reports. Neither archived confirmation count can be endorsed without its corresponding event-level validation and effective configuration. See [baseline provenance](baseline_2026_10_12/README.md).

## Export figures for archived experiment runs

No raw EEG dataset is necessary because the exporter uses already saved CSVs.

```powershell
python scripts/export_twin_adapt_run_figures.py --run-dir outputs/experiments/twin_adapt_optimization/2b/subject_01/20261114T145309426597Z_6535b2a3
```

New experiments export these figures automatically after the CSV files are
written. Existing runs can be backfilled by explicitly naming their directory;
there is no automatic migration, renaming, or deletion of archived evidence.

For the NAINT slides, use the established
`scripts/export_naint_slide_figures.py` workflow under
`outputs/presentations/naint_2026_11_20/`. Presentation-ready figures are
separate from the run-specific charts used for scientific provenance.

## Batch export of BCI KL Slide-12 figures

To populate the canonical `experiments/kl_stage2/<dataset>/<subject>/<run-key>/`
folders across selected subjects, use:

```powershell
python -m scripts.export_kl_bci_batch --dataset both --subjects 1 2
```

The wrapper calls the existing real-data exporter for each subject and writes
`experiments/kl_stage2/batch_report_<UTC>.json` with per-subject success,
`no_eligible_windows`, or failure reasons. The subject manifests, original
Hotelling decisions, PNG/SVG assets, and event logs remain unchanged. Official
GDF recordings (and 2A evaluation-label files for trial selection) are needed
locally; absent inputs generate explicit failures, not synthetic figures. See
[KL output archive](experiments/kl_stage2/README.md).
