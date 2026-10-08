> **Historical document — updated review 2026-10-08.** The original gap analysis below records an earlier implementation snapshot and is retained for methodological provenance. It is **not** a current list of unfixed defects. Consult the dated resolution register first. No paper-level numerical reproduction is claimed.


---

## Dated resolution register and research workplan update (8 October 2026)

**Purpose.** This register distinguishes (1) a code change committed, (2) tests or experiment evidence, and (3) whether the published Table 1 numerical result has been reproduced. Dates are the GitHub commit timestamps in UTC; they are *completion evidence for the cited code change*, not dates on which published reproduction was experimentally proven. Source review: the connected `main` branch on 2026-10-08.

### Commit-based timeline / project diary

| Date (UTC) | Commit evidence | Earlier issue | Current assessment |
| --- | --- | --- | --- |
| 2026-08-01 | [58b46b6](https://github.com/hlin863/ewma_covariate_shift_demo/commit/58b46b602) | Stage-II multivariate test absent | **Implemented initial Hotelling validator**; interpretation compared later |
| 2026-08-02 | [89bc792](https://github.com/hlin863/ewma_covariate_shift_demo/commit/89bc7928b) | Lambda sensitivity unavailable | **Implemented diagnostic sweep**, not final parameter-selection proof |
| 2026-08-03 | [7695075](https://github.com/hlin863/ewma_covariate_shift_demo/commit/769507507) | Training lambda differs from test lambda | **Code fix introduced**: `lambda_override` in EWMA fitting; confirm effective values and results per saved run |
| 2026-08-03 | [33b3f50](https://github.com/hlin863/ewma_covariate_shift_demo/commit/33b3f503a) | Fixed L / variance sensitivity | **Diagnostic runner implemented** for Dataset 2B; no evidence that the paper's unpublished L selection has been recovered |
| 2026-08-14 | [1a562a1](https://github.com/hlin863/ewma_covariate_shift_demo/commit/1a562a14f) | Single-vector Stage II differs from paper | **Two-sample Hotelling interpretation implemented**; reference/current-window choices remain experimental assumptions |
| 2026-08-18 | [43ca4f8](https://github.com/hlin863/ewma_covariate_shift_demo/commit/43ca4f8c1), [30a7172](https://github.com/hlin863/ewma_covariate_shift_demo/commit/30a717272), [b710269](https://github.com/hlin863/ewma_covariate_shift_demo/commit/b7102699a) | No Dataset 2A 70/30 development protocol | **Implemented and wired into 2A FBCSP and Table 1 runner**; exact original partition seed is unreported |
| 2026-09-13 | [8153b02](https://github.com/hlin863/ewma_covariate_shift_demo/commit/8153b0273), [78773cf](https://github.com/hlin863/ewma_covariate_shift_demo/commit/78773cf52) | Variance handling assumptions | **Further fixes and validation**; numerical impact must be identified by mode-controlled experiments |
| 2026-09-22 | [f0d7a3e](https://github.com/hlin863/ewma_covariate_shift_demo/commit/f0d7a3ee6) | Insufficient lambda analysis | **SSE visualisation added**; it is a diagnostic, not evidence of published per-subject parameter selection |
| 2026-09-27 | [2e4e699](https://github.com/hlin863/ewma_covariate_shift_demo/commit/2e4e699bf) | Reproducible split handling | **Shared stratified split helper added**; compare its sampling assumption with each paper protocol |
| 2026-10-02 | [751687f](https://github.com/hlin863/ewma_covariate_shift_demo/commit/751687fc1) | Unclear online/frozen variance semantics | **Assumptions documented in code**; statistical calibration remains open |
| 2026-10-03 | [7112138](https://github.com/hlin863/ewma_covariate_shift_demo/commit/711213807) | Stage-II variants intertwined | **Refactored into explicit canonical modules** with legacy compatibility imports |
| 2026-10-04 | [2a8d20b](https://github.com/hlin863/ewma_covariate_shift_demo/commit/2a8d20b6ddb88d3d94eb4cb5f301018e99101046) | Reporting the reproduction boundary | **Progress review deck updated**; this is presentation provenance, not experimental resolution |

### Current status against the seven original priorities

| Original priority | Code status on latest inspected branch | Empirical/reproduction status | Next check |
| --- | --- | --- | --- |
| 1. Control-limit scale L | **Partially addressed**: L sensitivity runner and validation-based calibration options available | **Open**: published training-time selection of L not established | Preserve fixed lambda; sweep L as diagnostics, select only on development validation; report CSW and limit widths |
| 2. Online variance absorbs shifts | **Implemented alternatives**, including frozen/online modes | **Open**: which interpretation most faithfully matches paper and calibrates false alarms | Same-feature, same-split mode ablation; inspect limit widths and event counts |
| 3. Inconsistent training/test lambda | **Code fix committed** 2026-08-03 | **Verification needed** for saved legacy outputs and effective training/test metadata | Assert lambda consistency in each new run manifest and an automated regression test |
| 4. Missing 70/30 development split | **Implemented for 2A** on 2026-08-18; repository README also documents 2B I–III stratified development / IV–V evaluation | **Open**: exact paper trial membership/random seed unknown; no claim of matching historical data split | Freeze split seed and hashes/indices; compare protocols by dataset without evaluation leakage |
| 5. Stage-II differs from two-sample test | **Implemented** on 2026-08-14 and reorganized 2026-10-03; other interpretations retained by design | **Open**: sample-window choice, dependence, covariance regularization, selection effects and exact counts | Run modes separately with status counts, p-values, window boundaries, validation time and null calibration |
| 6. CSP component normalisation/h sensitivity | **Feature extraction exists** and parameters are exposed in BCI pipelines | **Open**: paper-specific choices not proven from source | Run explicit `h`/normalisation ablation with feature dimensions and retained PCA explained variance logged |
| 7. Missing result-level diagnostics | **Partially addressed**: event-level Stage-II records and several experiment manifests exist | **Open**: consistent per-subject manifest and all event-level exports in the standard Table 1 runners | Produce immutable run directory containing complete configuration, event files, SHA, seeds, session roles and feature settings |

The existing README describes committed combined Table 1 counts of **2A: 52 warnings / 3 confirmations** and **2B: 127 warnings / 6 confirmations**, but notes that the aggregate file omits critical event-level provenance. These are saved computational results, **not** confirmation of a successful published Table 1 reproduction. The earlier B01 files disagree on confirmation counts; they must be traced to their generating configurations, not silently combined or superseded.

### Workplan alignment and next calendar notes

The first-year PhD plan (October 2026–September 2027) proposes **WP1 theory and architecture**, **WP2 change-point detection benchmarking**, **WP3 continual learning/sustainability**, **WP4 digital twin co-evolution**, **WP5 cross-domain evaluation**, and **WP6 synthesis/dissemination**. The following are *proposed planning windows, not scheduled calendar events or already achieved results*:

| Proposed window | Workplan | Deliverable and acceptance criterion |
| --- | --- | --- |
| **October 2026 — first** | WP1 + WP2 | Lock Table 1 experiment specification: raw dataset revision, subject/session split, FBCSP filters/CSP h/PCA, effective lambda, L, variance mode, Stage-II mode/windows/alpha/covariance, seed and code SHA. Save one immutable manifest per subject and run. |
| **October 2026 — second** | WP2 | Audit B01's 17-warning/5-vs-0-validation discrepancy by regenerating each interpretation; report `confirmed`, `rejected`, `pending_current_window`, `insufficient_reference_window`, `insufficient_degrees_of_freedom` and `skipped_nearby_alarm` separately. |
| **November 2026** | WP2 | Calibrate the full warning-plus-validation pipeline under no-change and synthetic shifts (multiple seeds); report event-level false positives, detection delay and sensitivity to L/lambda/variance/window settings. |
| **November–December 2026** | WP2 + WP3 | Run matched adaptation policies across no-shift, covariate-shift and changed-relationship settings; measure predictive gain, retrain count, memory and timing, distinguishing distribution detection from model utility. |
| **After evidence is locked** | WP3–WP5 | Extend label-delay, bounded memory/replay, and digital-twin/industrial comparisons without changing Table 1 reference configuration retrospectively. |
| **Ongoing** | WP6 | Link every slide/manuscript Table 1 assertion to manifest/run ID/commit; retain historical contradictory runs with explicit explanations. |

### Minimum next implementation contract

Suggested `outputs/experiments/<run_id>/` contents (not yet claimed as a universal implemented runner feature): `manifest.json`, `warnings.csv`, `validations.csv`, `subject_metrics.csv`, `configuration.json` and `environment.json`. Assign a unique run ID **before execution**, never overwrite an earlier result, and record code SHA, timestamp, dataset, subject, source-file identifiers, trial indices, feature representation, selected and effective EWMA settings, covariance method, number of retained PCA components and all validation statuses.

**Research decision gate:** do not mark this Table 1 investigation *experimentally resolved* until one script reproduces each saved value from its recorded configuration, sensitivity to undocumented paper parameters is reported separately, and a controlled comparison against published per-subject values is generated without tuning directly on held-out evaluation counts.

---

# Dataset 2B Table 1 algorithm gap analysis

Current real-data result: mean CSW 1.33 versus 17.56 published, and mean CSV 1.00 versus 10.33 published. The feature matrices now have one row per cue-aligned trial and 20 FBCSP columns, so the remaining gap is concentrated in the CSE interpretation and parameterisation.

## Priority 1: control-limit scale is assumed, not reproduced

`run_dataset_2b_subject` currently uses a universal control-limit multiplier of 3.0. The paper describes `L` as a CSE parameter obtained during training, but Table 1 publishes only lambda. A fixed 3-sigma limit is therefore an assumption and is the most direct explanation for almost no Stage-I warnings.

Required diagnostic: sweep `L` per subject while keeping the published lambda fixed, and report the resulting CSW count. Do not select `L` directly against the Table 1 test count in the final experiment; use this only to identify the plausible range, then reproduce the paper's training/validation selection process.

## Priority 2: online error-variance adaptation can absorb shifts

The current SD-EWMA updates the error variance after every testing observation:

`variance_i = theta * error_i**2 + (1-theta) * variance_(i-1)`

A large shifted observation immediately widens subsequent limits. This can turn a persistent distribution change into one initial alarm followed by no alarms. The paper's notation is ambiguous about the exact variance recursion and whether the test-phase scale is meant to adapt indefinitely.

Required diagnostic: compare three explicit modes:

1. frozen training residual variance;
2. online variance updated on every observation;
3. online variance updated only when the observation is inside the limits.

Record CSW counts and limit widths for every mode.

## Priority 3: the training residual scale is calculated with the wrong lambda path

`fit_sd_ewma` estimates its own lambda and uses that lambda to calculate the training residual variance. Later, Dataset 2B overrides only the test-time lambda with the published subject value. This means the initial residual variance and the test EWMA can be based on different lambda values.

For a controlled Table 1 comparison, the training path, residual variance, final state, and test path must all be calculated with the same published lambda. Add `lambda_override` to the fitting function or refit the training EWMA after choosing the effective lambda.

This is a concrete code defect, not merely an unpublished-paper ambiguity.

## Priority 4: the paper's 70/30 parameter-selection split is missing

The code currently fits PCA, EWMA state and residual scale using all trials from sessions I--III. The paper says the existing training dataset was divided into 70% training and 30% validation for parameter estimation.

Required change: create a deterministic subject-level 70/30 split that preserves trial order unless the paper or original implementation establishes randomisation. Fit the feature/PCA reference on the training portion and choose CSE parameters on validation only.

## Priority 5: Stage-II is a defensible substitute, not the published test

The implementation validates one test feature vector against the complete training distribution using a predictive one-sample Hotelling/Mahalanobis statistic and Ledoit-Wolf covariance. The manuscript describes a multivariate two-sample Hotelling T-squared test using equal-sized samples, but does not fully specify their construction.

The present Stage-II result must therefore be labelled `training_reference_predictive`, not exact Algorithm 1 reproduction. Stage II should not be calibrated until Stage I is near the published warning counts.

## Priority 6: CSP component normalisation may differ from Equation 11

The current code selects the extreme CSP components first, then normalises each selected component variance by the sum of only those selected variances. Equation 11 can be read as normalising the selected component by the sum across the `2h` selected components, which matches the code, but the precise Dataset 2B value of `h` is unpublished. With three channels, the current default is `h=1`, producing two features per band.

Required diagnostic: retain `h=1` as the main three-channel interpretation, but report a sensitivity comparison using all three CSP components with a clearly different feature definition. Do not claim either variant is confirmed by the manuscript.

## Priority 7: no result-level diagnostics are currently saved

The runner saves only counts. Add per-subject diagnostics containing:

- effective lambda;
- lambda used to fit training residuals;
- initial residual standard deviation;
- mean and median control-limit width;
- PC1 training/test standard deviations;
- warning indices;
- session IV/V boundary;
- Stage-II p-values.

These outputs are necessary to distinguish a feature-scale problem from a control-limit problem.

## Recommended implementation order

1. Make the training EWMA use the effective published lambda consistently.
2. Add Stage-I variance modes and an `L` sensitivity runner.
3. Save per-subject Stage-I diagnostics.
4. Add the paper's 70/30 parameter-selection path.
5. Bring CSW close before revisiting Stage II.
