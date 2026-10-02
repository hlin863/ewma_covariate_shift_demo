# Implementation lineage

This document preserves how the code structure evolved while keeping one
preferred import path for current development.

## Canonical package migrations

| Earlier path | Canonical path | Meaning of the transition |
| --- | --- | --- |
| `src.ewma` | `src.detection.stage1.sd_ewma` | Standalone EWMA became an explicit Stage-I warning method |
| `src.msd_ewma` | `src.detection.stage1.msd_ewma` | Multivariate EWMA grouped with Stage-I detection variants |
| `src.cse` | `src.detection.core` | CSE became dataset-agnostic detector orchestration |
| `src.cse_preprocessing` | `src.detection.preprocessing` | PCA preprocessing separated from orchestration |
| `src.fbcsp` | `src.bci.fbcsp` | Feature extraction placed under the BCI representation layer |
| `src.bci_data` | `src.bci.data` | BCI loading separated from dataset-specific protocols |
| `src.bci_features` | `src.bci.features` | Generic EEG features separated from FBCSP and dataset protocol |
| `src.table1_reproduction` | `src.reporting.table1` | Paper-style output separated from detector execution |
| `src.evaluation` | `src.reporting.metrics` | Evaluation metrics separated from algorithm code |
| `src.tssd_ewma` | `src.detection.two_stage` | Two-stage orchestration placed beside the detector stages |
| flat Dataset 2A modules | `src.bci.datasets.dataset2a` | Session/split assumptions isolated as dataset protocol |
| flat Dataset 2B modules | `src.bci.datasets.dataset2b` | Dataset 2B reproduction assumptions isolated as protocol |

The earlier modules remain as compatibility shims where practical. Their
presence documents the migration, while new code should use canonical paths.

## Stage-II interpretation lineage

Stage-II code intentionally preserves multiple interpretations rather than
collapsing them into one generic test:

- training-reference Hotelling / predictive interpretation;
- paper-style equal-window two-sample Hotelling;
- K-S validation for the univariate 2015 pathway.

Some Hotelling implementations still live in flat modules and are re-exported
through `src.detection.stage2`. This is an **incomplete structural migration**,
not evidence that one interpretation has replaced the others. If they are moved
later, preserve their names and methodological distinctions.

## Adaptation lineage

```text
supervised append-and-retrain
    true evaluation label available
        ↓
Diethe policy comparison
    detector evidence separated from update policy
        ↓
PWKNN transductive update
    labelled calibration + unlabelled evaluation
        ↓
future dynamic CSE-UAEL ensemble
    classifier growth + weighted voting
        ↓
future label-availability studies
    delayed / sparse / pseudo / no-label regimes
```

The current `src/adaptation/supervised.py` contains more than its historical
filename suggests because it records this transition. A future module split is
acceptable, but it must preserve the lineage explicitly: supervised and
transductive paths should remain separately identifiable rather than being
collapsed into an opaque generic learner.

## Result lineage

Saved outputs may be retained when they represent a methodological milestone.
A later result does not automatically invalidate an earlier one if the difference
documents a changed assumption. When keeping both, record the source commit,
configuration and reason for the change wherever practical.
