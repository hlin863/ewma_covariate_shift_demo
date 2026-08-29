# Table IV multivariate reproduction

This benchmark integrates the 2015 paper's synthetic D3 and D4 streams with a
paper-specific multivariate EWMA path. It deliberately does not reuse the
repository's later prediction-error MSD implementation: Table IV requires the
EWMA-state statistic and time-varying covariance in Equations 7--9.

## Dataset definitions

- D3: 300 observations from a 10-dimensional Gaussian distribution.
- D4: the same geometry using a multivariate Student t distribution with 10
  degrees of freedom.
- Both: mean 0 for observations 1--100, mean 1 for 101--200, and mean 0 again
  from observation 201. The covariance has diagonal 0.45 and every
  off-diagonal entry 0.30.

For D4, the Gaussian scale matrix is multiplied by `(nu - 2) / nu`. This makes
the covariance of the generated t distribution equal the paper's stated
Sigma. Treating Sigma as the t scale instead would inflate covariance by 1.25.

## Detector variants

The runner computes the five Table-IV columns:

1. MSD-EWMA on all 10 variables (`H = 22.67`).
2. MSD-EWMA-PCA on the first two training-fitted components (`H = 10.58`).
3. MSD-EWMA-ICA on 10 training-fitted FastICA components (`H = 22.67`).
4. TSMSD-EWMA using equal before/after Hotelling windows.
5. TSMSD-EWMA-ICA using the same validation after FastICA.

The paper discusses FastICA, Infomax and FBSS but does not identify the generic
ICA algorithm in the synthetic Table-IV column. This implementation uses
scikit-learn FastICA and records that choice in output metadata. It must not be
reported as an exact Infomax reproduction.

The default `lambda = 0.10` is consistent with the published 10-variable
control limit, but the paper does not print one multivariate lambda for D3/D4.
It remains a command-line parameter for sensitivity analysis.

## Scoring decision

Figure 6 and the results discussion describe the detected departure after the
100th observation. Each Monte Carlo run therefore scores the first departure
at 101: alarms before 101 are FP, absence of a report during 101--200 is FN,
and RCI is the first report time minus 101. Sustained alarms within the shifted
regime are not counted as new false shift events. The return at 201 remains in
the dataset and exported traces.

The paper does not state its random seeds or Monte Carlo repetition count.
Consequently, published values are reference targets, not assertions that the
computed values must match bit-for-bit. Computation time is also hardware and
software dependent.

## Run

```bash
python scripts/run_2015_synthetic_reproduction.py --dataset table4
```

Run either dataset or shorten a development run:

```bash
python scripts/run_2015_synthetic_reproduction.py --dataset d3 --repetitions 10
python scripts/run_2015_synthetic_reproduction.py --dataset d4 --repetitions 10
```

Outputs are written beneath `outputs/metrics/paper2015/table4/`, including the
computed and published comparison CSVs, a Markdown table, summary metadata,
and representative streams/traces for audit and plotting.
