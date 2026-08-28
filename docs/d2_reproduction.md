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
- The paper's Table III discusses nine evaluated shifts without identifying
  their indices. Figure 5(b) shows a late, high-mean D2 window. The Table III
  runner therefore uses the final 1,000 observations (4,001--5,000), containing
  the last nine truth markers (4,101--4,901). This is an explicit repository
  inference, not a parameter stated verbatim by the paper.
- ICI-CDT is implemented as the minimum-variance, zeroth-order ICI rule over
  means of disjoint blocks. D2 changes the innovation mean, so this is the
  relevant feature. Repeated monitoring resets the interval intersection after
  an alarm because the cited single-change ICI algorithm otherwise terminates.

## Run the experiment

Generate computed Table III rows for SD-EWMA, TSSD-EWMA and ICI-CDT. The
default uses the paper's reported D2 lambda of 0.40:

```bash
python scripts/run_2015_synthetic_reproduction.py --dataset d2
```

Run the separate full-stream diagnostic with a fitted lambda:

```bash
python scripts/run_2015_synthetic_reproduction.py \
  --dataset d2 \
  --evaluation-scope full-stream \
  --lambda-mode estimate
```

Results are written beneath `outputs/metrics/paper2015/d2/`:

```text
summary.json
stage1_events.csv
stage1_trace.csv
stage2_events.csv
stage2_validations.csv
ici_cdt_trace.csv
table3_computed.csv
table3_comparison.csv
table3_computed.md
table3_sd_ewma_events.csv
table3_tssd_ewma_events.csv
table3_ici_cdt_events.csv
```

`table3_computed.csv` contains the measured FP, FN, mean RCI and CT values.
`table3_comparison.csv` keeps those measurements beside the published Table III
values and calculates their differences. CT is measured locally with a
monotonic high-resolution clock, so it is expected to differ across hardware.

Stage-I event time is the EWMA alarm time. Stage-II event time is
`validation_time`, after its future K-S window is available, so its recognition
capability index includes the validation delay.

## Implementation boundaries

- `src/simulation/jumping_mean.py`: generator and truth labels only.
- `src/detection/stage1/sd_ewma.py`: generic online EWMA warnings.
- `src/detection/stage2/ks.py`: generic K-S validation.
- `src/detection/two_stage.py`: reusable TSSD-EWMA pipeline.
- `src/detection/baselines/ici_cdt.py`: mean-feature ICI-CDT comparator.
- `src/reporting/metrics.py`: one-to-one repeated-event evaluation.
- `src/experiments/paper2015/d2.py`: D2 split, Table III scope and scoring.
- `scripts/run_2015_synthetic_reproduction.py`: command-line and output writing.
