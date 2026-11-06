# Online feature-level replay (first implementation)

This path processes **one scalar detector input at a time**, with the existing
FBCSP/PCA preprocessing reused rather than rewritten.

## Precisely what arrives at each step

- An EEG *trial* is first transformed into a precomputed FBCSP feature vector
  of shape `(D,)`, using the existing fitted FBCSP model.
- `OnlinePC1Monitor.update_one(feature_vector, time)` uses frozen PCA to
  transform the **one** vector into a PCA vector `(K,)`.
- It selects PC1, a **single scalar**, and calls
  `OnlineSDEWMA.update_one(pc1, time)` exactly once.
- The detector computes previous-state control limits, inclusive alarm
  decision, EWMA update and the selected error-variance update mode.
- The result includes the full projected vector for subsequent Stage-II
  multivariate buffering. There is no per-coordinate SD-EWMA update here.

If "one feature at a time" means every coordinate of one trial's feature vector
arrives separately, the input is *incomplete* until all `D` coordinates
have arrived; PCA cannot be applied to one coordinate independently. The future
acquisition adapter must assemble a complete timestamped feature vector first.

## Minimal usage

```python
import numpy as np
from src.online.replay import OnlinePC1Monitor

# Use the existing 2A/2B development FBCSP pipeline to obtain
# training.features, testing.features and testing.times.
monitor = OnlinePC1Monitor.from_training_features(
    training_features, pca_components=2, lambda_override=0.2
)
for result in monitor.replay(testing_features, testing_times):
    print(result.time, result.stage1_trace["x"], result.stage1_trace["stage_1_alarm"])
```

The `replay()` generator processes rows chronologically. For a real
source, call `monitor.update_one(feature_vector, time)` as each vector arrives.
The same fitted PCA and detector state persist between calls.

## Research safeguards

- `run_sd_ewma` retains the public API and 17-column trace schema; it now
  shares one-step mathematics with `OnlineSDEWMA`.
- The default online initial z is the *final calibration EWMA*, not the
  initial calibration mean, preserving the existing `training_final` protocol.
- This is **streaming feature replay**, not causal real-time EEG processing.
  The existing FBCSP pipeline uses zero-phase `sosfiltfilt` filtering;
  do not describe replay of its outputs as live causal EEG processing.
- No test/evaluation labels are required for monitoring.
- No Hotelling Stage-II validation or model retraining is performed yet.
  The next milestone is a bounded feature buffer and delayed two-window
  validation on Stage-I warnings, followed by the existing adaptation policies.
- Outputs are not represented as reproductions of published BCI results
  unless separate paper-aligned validation has been completed.
