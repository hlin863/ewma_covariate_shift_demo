# NASA C-MAPSS data loading

The loader uses the existing `src/pdm/datasets/turbofan/` template and the
extracted files in `data/raw/turbofan_engine_degradation/`.

```python
from src.pdm.datasets.turbofan import TurbofanDataset, load_turbofan_dataset

data = load_turbofan_dataset(dataset="FD001")
# Equivalent class entry point, with an optional custom data directory:
data = TurbofanDataset.load("data/raw/turbofan_engine_degradation", "FD001")

print(data.training.shape, data.testing.shape)
training_targets = data.training_with_rul()
evaluation_targets = data.testing_with_rul()
```

`training` and `testing` retain the 26 source columns: `unit`, `cycle`,
three operating settings and 21 sensors. Rows are sorted by unit and cycle.
The default path resolves relative to the repository, independent of the
working directory. FD001–FD004 are supported, with case-insensitive names.

Counts verified from the committed files:

| Subset | Training rows | Test rows | Training engines | Test engines |
|---|---:|---:|---:|---:|
| FD001 | 20,631 | 13,096 | 100 | 100 |
| FD002 | 53,759 | 33,991 | 260 | 259 |
| FD003 | 24,720 | 16,596 | 100 | 100 |
| FD004 | 61,249 | 41,214 | 249 | 248 |

The supplied README reverses the FD004 engine counts. The loader uses the
actual unit IDs and validates the RUL vector against the test engine count.

The loader rejects missing or empty files, malformed tables, non-finite
measurements, invalid unit/cycle identifiers, missing or duplicate cycles,
and RUL files that do not contain exactly one non-negative integer value
per test engine. C-MAPSS unit IDs and cycles must start at 1 and be consecutive.

## Remaining useful life

Training trajectories reach failure, so the helper derives
`rul = final_cycle_for_unit - cycle`. Each training engine ends at zero.
Test trajectories stop before failure, so the helper derives
`rul = final_observed_cycle_for_unit - cycle + supplied_endpoint_rul`.
The entries in `test_rul` correspond to test units 1, 2, and so on.
Both helpers return new tables with an uncapped `rul` column.

Test RUL is offline evaluation/oracle information. Keep it out of feature
matrices and reveal only selected labels when implementing active learning.
Train and test IDs refer to separate fleets even when their numeric values match.

## Next experiment boundary

This layer loads trajectories and derives targets. The existing
`preprocessing.py` and `experiment.py` templates remain available for
engine-level splits, calibration-only scaling, configured classification
thresholds, and adaptive-learning evaluation. Use operating settings and
sensors as candidate features; do not include `rul` or the engine identifier.
Any binary degradation label must be documented as an experimental
transformation of RUL, since NASA's native task is RUL regression.

Run the loader checks from the repository root:

```bash
python -m pytest tests/pdm/test_turbofan.py -q
```

The tests cover small fixtures and all four committed NASA subsets. The
NASA `data/raw/turbofan_engine_degradation/readme.txt` defines the source
schema, train/test distinction, and endpoint RUL interpretation.
