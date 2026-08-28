# D2 jumping-mean reproduction

The active D2 generator is `src/simulation/jumping_mean.py`. It implements

```text
x(t) = 0.6 x(t-1) - 0.5 x(t-2) + epsilon(t)
```

for 5,000 paper-indexed observations with `x(1) = x(2) = 0`, innovation
standard deviation 1.5, and a cumulative innovation-mean change every 100
observations.

## Explicit reproduction decisions

- `time` is one-based, matching the paper.
- Regime 1 covers observations 1--100, so the first truth marker is at 101.
- The paper prints regimes 2--49 despite specifying 5,000 observations. The
  generator completes regime 50 using the same recurrence, making the final
  block and the shift at 4,901 explicit.
- The default experiment uses 500 observations for parameter fitting and
  evaluates every remaining truth marker. This produces 45 testing-stream
  shifts.
- The paper's Table III discusses only nine evaluated shifts without identifying
  their indices. The runner therefore reports the published values as reference
  metadata but does not claim that its full-stream metrics reproduce that
  undocumented subset.

## Run the experiment

Estimate lambda from the configured training section:

```bash
python scripts/run_2015_synthetic_reproduction.py --dataset d2
```

Run a diagnostic comparison using the paper's reported lambda of 0.40:

```bash
python scripts/run_2015_synthetic_reproduction.py \
  --dataset d2 \
  --lambda-mode configured
```

Results are written beneath `outputs/metrics/paper2015/d2/`:

```text
summary.json
stage1_events.csv
stage1_trace.csv
stage2_events.csv
stage2_validations.csv
```

Stage-I event time is the EWMA alarm time. Stage-II event time is
`validation_time`, after its future K-S window is available, so its recognition
capability index includes the validation delay.

## Implementation boundaries

- `src/simulation/jumping_mean.py`: generator and truth labels only.
- `src/detection/stage1/sd_ewma.py`: generic online EWMA warnings.
- `src/detection/stage2/ks.py`: generic K-S validation.
- `src/detection/two_stage.py`: reusable TSSD-EWMA pipeline.
- `src/reporting/metrics.py`: one-to-one repeated-event evaluation.
- `src/experiments/paper2015/d2.py`: D2 split, detector configuration and scoring.
- `scripts/run_2015_synthetic_reproduction.py`: command-line and output writing.
