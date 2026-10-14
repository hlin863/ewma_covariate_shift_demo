# Notebook index

These notebooks follow the repository's research workflow. Keep the numbered sequence stable: existing notebooks are preserved in their original paths, with new experiments appended rather than moved or renamed.

| Notebook | Focus | Role |
| --- | --- | --- |
| [01_gaussian_data_initialisation.ipynb](01_gaussian_data_initialisation.ipynb) | Gaussian data initialisation | Earlier data-generation exploration |
| [02_sd_ewma_detection.ipynb](02_sd_ewma_detection.ipynb) | SD/EWMA detection | Earlier monitoring exploration |
| [03_structural_changes_visualization.ipynb](03_structural_changes_visualization.ipynb) | Structural-change visualisation | Earlier structural-change exploration |
| [04_structural_changes_save_results.ipynb](04_structural_changes_save_results.ipynb) | Structural-change result saving | Earlier results workflow |
| [05_park_mean_cusum_smoothing.ipynb](05_park_mean_cusum_smoothing.ipynb) | Park (2023) aggregated mean-CUSUM and LOESS-style smoothing | Exploratory reproducible test and visualisation |

## Running the notebooks

Start Jupyter from the repository root or `notebooks/`. Install the repository's dependencies, including `numpy`, `matplotlib`, and Jupyter, in your environment. Execute cells sequentially. Notebook 05 locates the repository root automatically for either working directory and imports the production `src/detection/stage1/park_mean_cusum.py` directly, avoiding copied detector implementations.

## Evidence and interpretation

- Notebook 05 uses synthetic positive-mean multivariate data with a fixed random seed. Its test results are not published-paper reproduction claims.
- Park et al. specify span 0.1; local-linear degree 1 and tricube kernel are transparent implementation choices. The span operates on *retained candidate positions*, after the `min_segment` exclusion, not on the complete input time domain.
- Raw and smoothed peak estimates are separate. Peak locations alone do not establish a confirmed change. Unimodality and peak-agreement checks are future detector work.
- Avoid committing private EEG data or overwriting historical notebook outputs merely to reorganise this directory.
- A known short-series boundary issue exists in the production helper: some segment-valid inputs have fewer than three retained candidate positions and therefore cannot be smoothed. The notebook records this explicitly.

## Organisation policy

Keep notebook names numbered and descriptive. Put reusable detection implementations in `src/`, regression checks in `tests/`, and persistent experimental outputs in the appropriate `outputs/` location. Notebooks should act as documented, reproducible explorations rather than competing source-code implementations.
