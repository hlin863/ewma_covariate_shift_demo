# Hotelling T² with GPR uncertainty and kernel diagnostics

## Placement and method boundary

This opt-in extension keeps the existing repository layers and paper paths:

| Layer | Integration point | Behaviour |
|---|---|---|
| Stage II | `src/detection/stage2/gpr_kernel_hotelling.py` | Wraps the unchanged equal-window Hotelling validator; adds MMD and GPR diagnostics |
| Adaptation policy | `src/adaptation/policies.py::RetrainOnGPRKernelRecommendation` | Checks an experimental recommendation only after its validation time |
| Experiment | `src/adaptation/experiments/gpr_kernel.py` | Runs the existing EWMA warning implementation and the new validator |
| CLI | `scripts/run_gpr_kernel_validation.py` | Synthetic suite or prepared arrays; saves configuration, events, versions and provenance |
| Dashboard | `/results/gpr-kernel` | Interactive laboratory linked in Results → Validation and reproduction |
| Verification | `tests/detection/stage2/test_gpr_kernel.py`, `tests/web/test_gpr_kernel_page.py` | Numerical, timing, isolation and reachable-page checks |

The paper-reproduction runners and original `confirmed_shift` values remain unchanged.
This is an engineering extension, not a GPR method attributed to Chowdhury,
Raza or Diethe. Diethe et al. (2019), *Continual Learning in Practice*, §4.1
(p. 4 in the supplied PDF), motivates shift magnitude, uncertainty and explanation
as monitoring outputs; it does not specify this GPR/MMD combination.

## Three independent signals

1. **T²:** calls `validate_paper_two_sample_alarms` on raw prepared features.
   Reference is `[i-H+1:i+1]`; current is `[i+1:i+1+H]`. Its existing approximate
   F conversion and degrees-of-freedom limitations are inherited. No new T²
   calibration claim is made.
2. **Kernel diagnostic:** training-fitted standardisation, RBF kernel
   `k(x,y)=exp(-gamma*||x-y||²)`, with `gamma=1/median(positive squared training
   distances)`. The unbiased MMD² estimator excludes within-sample diagonal terms.
   Negative estimates are valid and retained. An IID permutation null uses the
   plus-one p-value `(1+exceedances)/(B+1)`. This implementation is motivated by
   [Gretton et al. (2012)](https://www.jmlr.org/papers/v13/gretton12a.html);
   its permutation procedure is not a reproduction of all tests in that paper.
3. **GPR uncertainty:** a fixed RBF GPR predicts the supplied continuous target.
   `optimizer=None` keeps the length scale explicit, `normalize_y=True` sets
   target units, and `alpha` is the specified observation-noise variance in
   normalised target units. With no WhiteKernel, returned standard deviation
   describes posterior latent-function uncertainty, not future observation noise.
   Mean std is compared with the 95th percentile of nonoverlapping calibration
   WINDOW means, not the percentile of individual predictions. See the
   [official GPR API](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).

Binary motor-imagery targets may be supplied as a regression surrogate; neither
its mean nor std is a calibrated classification probability. Prefer a documented
continuous target when available. Training is bounded to 500 rows for this prototype.
The caller must select a training-only subset and record any subsampling.

## Experimental adaptation gate

`adaptation_recommended = (confirmed_shift OR kernel_shift) AND high_uncertainty`.
The baseline status is preserved. This is an exploratory decision heuristic,
not a joint level-alpha test. The module does not retrain, query labels, compare
candidate models, or accept an update. The separate policy can be passed into an
adaptation loop. That loop must release events at `validation_time`, consume each
once, handle cooldown and label availability, and evaluate a candidate before use.
For warning index `i`, the earliest decision is at `i+H`; results must never be
backdated to the warning. Incomplete/skipped windows cannot recommend adaptation.

## Run and export

```bash
python -m scripts.run_gpr_kernel_validation --all-scenarios
python -m scripts.run_gpr_kernel_validation --features prepared.npz --provenance provenance.json
python -m pytest -q tests/detection/stage2 tests/adaptation/test_policies.py tests/web/test_gpr_kernel_page.py tests/web/test_diethe_page.py
```

The NPZ schema is `training_features (n,D)`, `training_targets (n,)`,
`calibration_features (m,D)`, `evaluation_features (k,D)`, `evaluation_times (k,)`.
Load uses `allow_pickle=False`. At least five full calibration windows are required.
Use separate chronological splits, fit CSP/PCA/filter-selection transforms on
training only, preserve evaluation order and unique integer times, and record
those decisions. Provenance JSON requires `dataset`, `subject`, `split`,
`preprocessing` and `target_definition`. The module cannot independently verify
that arrays are disjoint or transformations were fitted correctly.

Exports go to a new timestamped `outputs/gpr_kernel/.../validation.json`, with
revision, dirty-tree flag, versions, command, settings, warning and validation
records. Existing experimental outputs are not replaced. Web runs are transient;
the CLI creates a durable auditable result in the repository.

## What validation establishes

The synthetic suite separates no-change, input mean, variance, support and
relationship-change scenarios. It uses disjoint training/calibration/evaluation
samples and never injects a warning at the known intervention. Therefore Stage I
can miss a change and prevent subsequent diagnostics. The relationship scenario
leaves inputs unchanged and represents an unobserved target-rule reversal;
input-only diagnostics cannot distinguish it from the no-change scenario.

Unit checks cover the unbiased MMD formula, a same-mean variance contrast,
GPR extrapolation uncertainty, unchanged baseline records, pending windows,
future-window isolation, delayed policy eligibility and input contracts.
One synthetic seed verifies execution, not detector power, calibration or EEG
performance. Recommendation counts are event counts, not accuracy improvements.

Next research validation: run several seeds and independent no-change streams;
measure false alarms, pre-change confirmations and decision delays; ablate T²,
MMD and the combined uncertainty gate; then apply subject-held-out 2A/2B features.
Temporal EEG dependence needs a justified block/resampling null rather than IID
permutation. Repeated warnings need a stream-level error budget. Label/residual
monitoring is needed to assess changes in P(y|x), and adaptation benefit requires
matched retraining experiments plus forgetting/cost evaluation.

## Executed validation — 12 October 2026

The full suite passed **358 tests**. A subsequently added prepared-array CLI
export test passed separately (**1 test**). `git diff --check` passed.
The audited seed-42 suite uses H=20 and 199 permutations:

| Synthetic scenario | EWMA warnings | Evaluated diagnostic windows | T² confirmations | MMD confirmations | High-uncertainty windows | Review recommendations |
|---|---:|---:|---:|---:|---:|---:|
| No change | 6 | 3 | 0 | 0 | 1 | 0 |
| Mean shift | 7 | 4 | 1 | 1 | 3 | 1 |
| Variance shift | 55 | 6 | 0 | 0 | 5 | 0 |
| Support shift | 8 | 4 | 1 | 1 | 3 | 1 |
| Unobserved relationship shift | 6 | 3 | 0 | 0 | 1 | 0 |

The variance scenario is a retained pipeline miss, despite a direct same-mean
variance MMD unit check succeeding. Warning placement, windows and bandwidth
need sensitivity analysis; the run does not isolate a single cause. The final
relationship scenario is intentionally indistinguishable from no-change to this
input-only monitor. These outcomes do not establish superiority over T².

Complete events: [`outputs/gpr_kernel/20261012_audited_validation/validation.json`](../outputs/gpr_kernel/20261012_audited_validation/validation.json).
Its revision denotes the base commit and `working_tree_dirty=true` identifies
execution with the extension present. Source SHA-256 hashes identify the executed
code exactly. An earlier initial run is also retained in its own output folder.
