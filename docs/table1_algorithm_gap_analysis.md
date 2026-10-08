1| > **Historical document — updated review 2026-10-08.** The original gap analysis below records an earlier implementation snapshot and is retained for methodological provenance. It is **not** a current implementation claim and should be read as a historical reference rather than the present status of the project.
2| 
3| 
4| ---
5| 
6| ## Dated resolution register and research workplan update (8 October 2026)
7| 
8| **Purpose.** This register distinguishes (1) a code change committed, (2) tests or experiment evidence, and (3) whether the published Table 1 numerical result has been reproduced. Dates are the Git commit dates in UTC and therefore may differ from the date in the document record.
9| 
10| ### Commit-based timeline / project diary
11| 
12| | Date (UTC) | Commit evidence | Earlier issue | Current assessment |
13| | --- | --- | --- | --- |
14| | 2026-08-01 | [58b46b6](https://github.com/hlin863/ewma_covariate_shift_demo/commit/58b46b602) | Stage-II multivariate test absent | **Implemented initial Hotelling validator**; interpretation context remains under review |
15| | 2026-08-02 | [89bc792](https://github.com/hlin863/ewma_covariate_shift_demo/commit/89bc7928b) | Lambda sensitivity unavailable | **Implemented diagnostic sweep**, not final parameter-selection procedure |
16| | 2026-08-03 | [7695075](https://github.com/hlin863/ewma_covariate_shift_demo/commit/769507507) | Training lambda differs from test lambda | **Code fix introduced**: `lambda_override` in EWMA fitting path; still needs verification |
17| | 2026-08-03 | [33b3f50](https://github.com/hlin863/ewma_covariate_shift_demo/commit/33b3f503a) | Fixed L / variance sensitivity | **Diagnostic runner implemented** for Dataset 2B; no evidence that the paper value is reproduced |
18| | 2026-08-14 | [1a562a1](https://github.com/hlin863/ewma_covariate_shift_demo/commit/1a562a14f) | Single-vector Stage II differs from paper | **Two-sample Hotelling interpretation implemented**; results still require calibration against the paper |
19| | 2026-08-18 | [43ca4f8](https://github.com/hlin863/ewma_covariate_shift_demo/commit/43ca4f8c1), [30a7172](https://github.com/hlin863/ewma_covariate_shift_demo/commit/30a717272), [b710269](https://github.com/hlin863/ewma_covariate_shift_demo/commit/b710269) | Missing Calibrated 70/30 split and dataset cross-validation coverage | **Dataset 2A and 2B split utilities are in place**; the exact paper-trial matching remains under review |
20| | 2026-09-13 | [8153b02](https://github.com/hlin863/ewma_covariate_shift_demo/commit/8153b0273), [78773cf](https://github.com/hlin863/ewma_covariate_shift_demo/commit/78773cf52) | Variance handling and paper assumptions ambiguous | **Code paths made explicit**; final statistical interpretation remains unresolved |
21| | 2026-09-22 | [f0d7a3e](https://github.com/hlin863/ewma_covariate_shift_demo/commit/f0d7a3ee6) | Insufficient lambda analysis | **SSE visualisation added**; it is a diagnostic, not evidence of publication-level reproduction |
22| | 2026-09-27 | [2e4e699](https://github.com/hlin863/ewma_covariate_shift_demo/commit/2e4e699bf) | Reproducible split handling | **Shared stratified split helper added**; compare its sampling assumptions against the paper before treating it as authoritative |
23| | 2026-10-02 | [751687f](https://github.com/hlin863/ewma_covariate_shift_demo/commit/751687fc1) | Unclear online/frozen variance semantics | **Assumptions documented in code**; statistical calibration and final interpretation remain open |
24| | 2026-10-03 | [7112138](https://github.com/hlin863/ewma_covariate_shift_demo/commit/711213807) | Stage-II variants intertwined | **Refactored into explicit canonical modules** with legacy compatibility retained |
25| | 2026-10-04 | [2a8d20b](https://github.com/hlin863/ewma_covariate_shift_demo/commit/2a8d20b6ddb88d3d94eb4cb5f301018e99101046) | Reporting the reproduction boundary | **Progress review deck updated**; evidence still falls short of a full Table 1 reproduction claim |
26| 
27| ### Current status against the seven original priorities
28| 
29| Progress legend: 🟢 = resolved / strong evidence, 🟡 = partial progress / in progress, 🔴 = open / unresolved.
30| 
31| | Original priority | Progress | Code status on latest inspected branch | Empirical/reproduction status | Next check |
32| | --- | --- | --- | --- | --- |
33| | 1. Control-limit scale L | 🟡 | **Partially addressed**: L sensitivity runner and validation-based calibration options available | **Open**: published training-time selection of L not established | Please verify the exact training-time L policy against the source paper |
34| | 2. Online variance absorbs shifts | 🟡 | **Implemented alternatives**, including frozen/online modes | **Open**: which interpretation most faithfully matches paper and calibrates false alarms | Same as above; compare with paper conventions before finalising |
35| | 3. Inconsistent training/test lambda | 🟡 | **Code fix committed** 2026-08-03 | **Verification needed** for saved legacy outputs and effective training/test metadata | Assert lambda consistency in each saved run output |
36| | 4. Missing 70/30 development split | 🟡 | **Implemented for 2A** on 2026-08-18; repository README also documents 2B I–III stratified development / IV–V evaluation | **Open**: exact paper trial method and split reproducibility still need confirmation | Confirm the exact development/evaluation split used by the reference study |
37| | 5. Stage-II differs from two-sample test | 🟡 | **Implemented** on 2026-08-14 and reorganized 2026-10-03; other interpretations retained by design | **Open**: sample-window choice, dependence, covariance assumptions and paper equivalence | Re-run each Stage-II variant and compare to the original paper's statistic |
38| | 6. CSP component normalisation/h sensitivity | 🟡 | **Feature extraction exists** and parameters are exposed in BCI pipelines | **Open**: paper-specific choices not proven from source | Run explicit sensitivity checks for `h` and CSP normalisation options |
39| | 7. Missing result-level diagnostics | 🟡 | **Partially addressed**: event-level Stage-II records and several experiment manifests exist | **Open**: consistent per-subject manifest and all event-level outputs are still not locked down | Add the missing per-subject manifest and result-level validation metadata |
40| 
41| The existing README describes committed combined Table 1 counts of **2A: 52 warnings / 3 confirmations** and **2B: 127 warnings / 6 confirmations**, but notes that the aggregate file omits critical run metadata. The documentation therefore cannot currently claim a full published-value reproduction without closer audit.
42| 
43| ### Workplan alignment and next calendar notes
44| 
45| The first-year PhD plan (October 2026–September 2027) proposes **WP1 theory and architecture**, **WP2 change-point detection benchmarking**, **WP3 continual learning/sustainability**, **WP4 digital-twin/industrial deployment**, **WP5 publication and reproducibility**, and **WP6 documentation/governance**. This update aligns the implementation review to the plan.
46| 
47| | Proposed window | Workplan | Deliverable and acceptance criterion |
48| | --- | --- | --- |
49| | **October 2026 — first** | WP1 + WP2 | Lock Table 1 experiment specification: raw dataset revision, subject/session split, FBCSP filters/CSP h/PCA, effective lambda, L, variance mode, Stage-II interpretation, and reporting convention |
50| | **October 2026 — second** | WP2 | Audit B01's 17-warning/5-vs-0-validation discrepancy by regenerating each interpretation; report `confirmed`, `rejected`, `pending_current_window`, `insufficient_evidence`, and raw run IDs |
51| | **November 2026** | WP2 | Calibrate the full warning-plus-validation pipeline under no-change and synthetic shifts (multiple seeds); report event-level false positives, detection delay and sensitivity to threshold scaling |
52| | **November–December 2026** | WP2 + WP3 | Run matched adaptation policies across no-shift, covariate-shift and changed-relationship settings; measure predictive gain, retrain count, memory and compute cost |
53| | **After evidence is locked** | WP3–WP5 | Extend label-delay, bounded memory/replay, and digital-twin/industrial comparisons without changing Table 1 reference configuration retrospectively. |
54| | **Ongoing** | WP6 | Link every slide/manuscript Table 1 assertion to manifest/run ID/commit; retain historical contradictory runs with explicit explanations. |
55| 
56| ### Minimum next implementation contract
57| 
58| Suggested `outputs/experiments/<run_id>/` contents (not yet claimed as a universal implemented runner feature): `manifest.json`, `warnings.csv`, `validations.csv`, `subject_metrics.csv`, `config.yaml`, and `notes.md`.
59| 
60| **Research decision gate:** do not mark this Table 1 investigation *experimentally resolved* until one script reproduces each saved value from its recorded configuration, sensitivity to undocumented assumptions, and matching metadata chain from commit to output. |
61| 
62| ---
63| 
64| # Dataset 2B Table 1 algorithm gap analysis
65| 
66| Current real-data result: mean CSW 1.33 versus 17.56 published, and mean CSV 1.00 versus 10.33 published. The feature matrices now have one row per cue-aligned trial and 20 FBCSP columns, so the reproduction gap is no longer due to a shape mismatch alone.
67| 
68| ## Priority 1: control-limit scale is assumed, not reproduced
69| 
70| `run_dataset_2b_subject` currently uses a universal control-limit multiplier of 3.0. The paper describes `L` as a CSE parameter obtained during training, but Table 1 publishes only lambda. A fixed multiplier may not reproduce the original pipeline and may hide the actual sensitivity to `L`.
71| 
72| Required diagnostic: sweep `L` per subject while keeping the published lambda fixed, and report the resulting CSW count. Do not select `L` directly against the Table 1 test count in the final experiment.
73| 
74| ## Priority 2: online error-variance adaptation can absorb shifts
75| 
76| The current SD-EWMA updates the error variance after every testing observation:
77| 
78| `variance_i = theta * error_i**2 + (1-theta) * variance_(i-1)`
79| 
80| A large shifted observation immediately widens subsequent limits. This can turn a persistent distribution change into one initial alarm followed by no alarms. The paper's notation is ambiguous about whether the online variance should be frozen or updated after each residual.
81| 
82| Required diagnostic: compare three explicit modes:
83| 
84| 1. frozen training residual variance;
85| 2. online variance updated on every observation;
86| 3. online variance updated only when the observation is inside the limits.
87| 
88| Record CSW counts and limit widths for every mode.
89| 
90| ## Priority 3: the training residual scale is calculated with the wrong lambda path
91| 
92| `fit_sd_ewma` estimates its own lambda and uses that lambda to calculate the training residual variance. Later, Dataset 2B overrides only the test-time lambda with the published subject value. This creates a mismatch between the effective lambda used to fit the model and the lambda used to assess the run.
93| 
94| For a controlled Table 1 comparison, the training path, residual variance, final state, and test path must all be calculated with the same published lambda. Add `lambda_override` to the fitting function and validate its effect on thresholds and CSW counts.
95| 
96| This is a concrete code defect, not merely an unpublished-paper ambiguity.
97| 
98| ## Priority 4: the paper's 70/30 parameter-selection split is missing
99| 
100| The code currently fits PCA, EWMA state and residual scale using all trials from sessions I--III. The paper says the existing training dataset was divided into 70% training and 30% validation for parameter selection. The missing split likely explains part of the discrepancy between the code and the published output.
101| 
102| Required change: create a deterministic subject-level 70/30 split that preserves trial order unless the paper or original implementation establishes randomisation. Fit the feature/PCA reference on the first proportion and tune parameters on the validation proportion.
103| 
104| ## Priority 5: Stage-II is a defensible substitute, not the published test
105| 
106| The implementation validates one test feature vector against the complete training distribution using a predictive one-sample Hotelling/Mahalanobis statistic and Ledoit-Wolf covariance. The manuscript text may describe a two-sample or repeated-measures procedure, and the present implementation is therefore not yet a faithful reproduction of the published Table 1 algorithm.
107| 
108| The present Stage-II result must therefore be labelled `training_reference_predictive`, not exact Algorithm 1 reproduction. Stage II should not be calibrated until Stage I is near the published warning counts.
109| 
110| ## Priority 6: CSP component normalisation may differ from Equation 11
111| 
112| The current code selects the extreme CSP components first, then normalises each selected component variance by the sum of only those selected variances. Equation 11 can be read as normalising the selected components by the total variance across all components. The result is a design choice that may change the effective feature scale and control-limit widths.
113| 
114| Required diagnostic: retain `h=1` as the main three-channel interpretation, but report a sensitivity comparison using all three CSP components with a clearly different feature definition. Do not calibrate Stage II until the Stage I feature scaling is stable.
115| 
116| ## Priority 7: no result-level diagnostics are currently saved
117| 
118| The runner saves only counts. Add per-subject diagnostics containing:
119| 
120| - effective lambda;
121| - lambda used to fit training residuals;
122| - initial residual standard deviation;
123| - mean and median control-limit width;
124| - PC1 training/test standard deviations;
125| - warning indices;
126| - session IV/V boundary;
127| - Stage-II p-values.
128| 
129| These outputs are necessary to distinguish a feature-scale problem from a control-limit problem.
130| 
131| ## Recommended implementation order
132| 
133| 1. Make the training EWMA use the effective published lambda consistently.
134| 2. Add Stage-I variance modes and an `L` sensitivity runner.
135| 3. Save per-subject Stage-I diagnostics.
136| 4. Add the paper's 70/30 parameter-selection path.
| 5. Bring CSW close before revisiting Stage II.
| 
