# MATLAB — Park et al. (2023), Section 2.1 mean-change explorer

Independent, educational MATLAB translation of the Python module `src/detection/stage1/park_mean_cusum.py`, using the same **positive-mean, observation-by-variable** input contract. MATLAB stores the per-variable CUSUM values as **positions × variables**, rather than the Python JSON interface's variables × positions.

## Start

Open MATLAB in the repository root and run:

```matlab
addpath('matlab/park_mean_change');
selftest_park_mean_cusum
run_park_mean_change
```

For **real BCI Competition IV 2A or 2B**, first export cue-aligned trial features via the Python script, then open the generated CSV in MATLAB:

```powershell
python -m scripts.export_park_matlab_features --dataset 2b --subject 1 --session 1 --output outputs/matlab/park_2b_subject01_session01.csv
```

```matlab
run_park_mean_change('csv','outputs/matlab/park_2b_subject01_session01.csv',8,'outputs/matlab/figures')
```

2A has sessions 1=T and 2=E; 2B has sessions 1–5. The original BCI GDF directories must be available to Python, and the already existing Python session extraction pipeline remains authoritative. **Do not concatenate different trials or recording sessions as if they were continuous raw EEG samples.** Each row of the exported CSV represents *one ordered cue-aligned trial* with positive `mu_power_uv2`, `beta_power_uv2`, and `rms_uv`. These are **not** physical electrodes.

If local GDF files are unavailable, the demonstration mode still works. CSV files should have numeric `time` and at least two strictly positive-mean feature columns. No MAT files or Signal Processing Toolbox are required for the core figures.

## Statistics

For split `t` between the `t`-th and `(t+1)`-th observations (1-based MATLAB rows):

```text
nu_t(d) = abs( sqrt((T-t)/(T*t)) * sum(X(1:t,d)) ...
            - sqrt(t/(T*(T-t))) * sum(X(t+1:T,d)) ) / mean(X(:,d))
nu_max(t) = max_d nu_t(d)
nu_avg(t) = mean_d nu_t(d)
b_max = argmax_t nu_max(t), b_avg = argmax_t nu_avg(t)
```

Only candidate splits with `minSegment <= t <= T-minSegment` are evaluated. Visualisations show input features, each normalised per-variable CUSUM, and maximum/average aggregation with estimated peaks.

**Scope:** exploratory retrospective candidate localisation. Not a calibrated hypothesis test, not the paper's full unimodality/peak-agreement decision rule, not spectral/wavelet change detection, and not an online EEG classifier adaptation. Maximum/average scores may peak even in stationary data; the figures alone do not prove covariate shift.

**Reference:** Park, Y., Im, H., & Lim, Y. (2023). *Change points detection for nonstationary multivariate time series*. *Communications for Statistical Applications and Methods*, 30(4), 369–388. https://doi.org/10.29220/CSAM.2023.30.4.369

## Files

- `park_mean_cusum.m`: vectorised statistic and candidate maxima.
- `load_park_features.m`: deterministic example or feature CSV loading.
- `plot_park_mean_change.m`: three linked research figures.
- `run_park_mean_change.m`: entry point, optional CSV/PNG export.
- `selftest_park_mean_cusum.m`: synthetic step and algebraic equivalence checks.
- `scripts/export_park_matlab_features.py`: uses the existing Python BCI loader to create a reusable feature CSV.
