# Park et al. (2023): multivariate data inspection (WP2)

The existing `/data-structures` menu includes a **Park (2023) · Multivariate structure** entry at `/data-structures/park`.

**Literature:** Park, Y., Im, H., & Lim, Y. (2023). *Change points detection for nonstationary multivariate time series*. Communications for Statistical Applications and Methods, 30(4), 369–388. https://doi.org/10.29220/CSAM.2023.30.4.369

## Scope

This is **data-loading and exploratory inspection**, not a paper reproduction. Park et al. investigate mean changes with an aggregated CUSUM-type test and, conditionally, second-order changes using locally stationary wavelet spectral matrices and dynamic PCA. None of those three estimators has been implemented by this inspection page.

The current view provides an intentionally simple diagnostic: mean across channels, average channel sample variance, and Pearson correlation of the first two channels within consecutive non-overlapping windows. Changes in those descriptors are *not* statistical change-point confirmations.

## Source selection

By default, the page generates a deterministic 240-row, three-channel synthetic demonstration with a midstream **cross-correlation** change and approximately unchanged marginal distributions (not the paper's published data). No external data download is required.

To inspect your own CSV, configure Flask:

```python
app.config["PARK_MULTIVARIATE_DATA_PATH"] = "/absolute/path/to/multivariate.csv"
```

CSV format: two or more numeric channel columns, with an optional numeric, strictly increasing `time` column. Without `time`, a row-index time is generated. The loader refuses missing or non-finite values, repeated or unordered timestamps, and streams shorter than eight observations; it does **not** resample irregular observations. `?window=40` changes descriptor window length (minimum 4, maximum number of samples). Remaining incomplete observations are excluded.

## EEG integration boundary

BCI IV 2A/2B continue to use the existing GDF/session loaders and cue-aligned trial extractors. Do **not** flatten different trials into one continuous sequence: those boundaries generate artificial spectral changes. A future EEG-specific spectral module should compute each channel's representation within a trial, align by cue/condition, aggregate replicate periodograms in permitted splits, and keep evaluation labels out of training/calibration. The attached discussion of continuous-time locally stationary wavelet processes (CLSWP) suggests continuous-scale EWS as another **proposed** representation; it is distinct from Park et al.'s locally stationary wavelet/dynamic-spectral construction. Neither method is claimed implemented here.

## Follow-on WP2

1. Introduce controlled mean-only, variance-only, cross-dependence-only and simultaneous changes, with known ground truth.
2. Implement and independently test aggregated CUSUM only after verifying its exact form in Park et al.
3. Study local spectral matrix estimation and dynamic PCA separately; compare against existing EWMA/Hotelling and PCA score/residual monitors.
4. Calibrate false-alarm rates at the **stream** level with repeated seeds; report detection time, localisation error, false alarms and runtime.
5. Preserve read-only descriptive EDA and the current Table 1 EEG reproduction contracts.

## Files

- `src/detection/multivariate_inspection.py`: strictly numeric CSV loader, deterministic example, and window descriptors.
- `src/web/data_structures.py`: existing catalogue and read-only view.
- `templates/data_structures.html`: window table and selector.
- `tests/detection/test_multivariate_inspection.py`: loader/descriptor regression tests.
- `tests/web/test_park_data_structures.py`: route checks.

