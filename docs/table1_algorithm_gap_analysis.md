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
