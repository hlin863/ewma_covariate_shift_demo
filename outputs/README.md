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
