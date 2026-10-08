# Dataset navigation and inspection

Open **Data structures** in the application navigation, or visit
`/data-structures` for the dataset overview. Dataset EDA is grouped here
instead of appearing as a single BCI entry on the home-page architecture.

| Dataset | View |
|---|---|
| BCI Competition IV 2A | Existing EEG EDA with 2A selected |
| BCI Competition IV 2B | Existing EEG EDA with 2B selected |
| NASA Turbofan / C-MAPSS | FD001–FD004 counts, column types, train/test previews, RUL and sensor summaries |
| Chowdhury stroke cohort | Existing participant metadata and cohort distributions |
| NASA Algae Raceway | MAT variables and per-raceway measurement shapes, sample counts, missing values and time ranges |
| Synthetic Gaussian mean shift | CSV schema and source observations |
| Park et al. (2023) · Multivariate structure | Demonstration or CSV-loaded time-ordered sensor channels, window mean/variance/correlation; **not** the paper's CUSUM/wavelet/dynamic PCA detector |

The submenu is shared across the research pages. The original BCI
`/data-distributions` URL and its subject selector remain supported.
Each BCI submenu option sets its dataset explicitly. Missing local files
show an unavailable-data panel while preserving navigation to the other datasets.

The new source inspectors use these optional Flask configuration keys:

| Configuration | Default source |
|---|---|
| `TURBOFAN_DATA_PATH` | `data/raw/turbofan_engine_degradation/` |
| `ALGAE_DATA_PATH` | `data/raw/Algae Raceway/algae.mat` |
| `SYNTHETIC_DATA_PATH` | `data/raw/gaussian_mean_shift.csv` |
| `PARK_MULTIVARIATE_DATA_PATH` | Unset: deterministic synthetic multivariate demonstration |

Inspection is read-only. Turbofan uses the validated data loader and
displays RUL as offline target information. Algae signals retain their
separate timestamps; this page does not align sampling rates or define
classification labels. Descriptive summaries do not run the shift detector
or retrain models.

See [Park-inspired multivariate inspection](park_multivariate_inspection.md) for the input contract, temporal-dependence research motivation, mathematical boundaries, and planned WP2 work. The view preserves the existing source-data menu and does not refit BCI transformations or modify adaptation policies.
