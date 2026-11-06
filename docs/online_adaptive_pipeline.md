# Full online-adaptive BCI feature-replay integration

This repository's existing BCI experiment is assembled **without duplicating its
statistical methods**:

```
2A/2B development split -> fitted FBCSP -> trial feature row (D,)
   |                                          |
   +-- calibration labels -> LinearSVM.fit   +-> LinearSVM.predict (before adaptation)
                                              |
                                         frozen PCA -> PC1 -> SD-EWMA
                                              |
                                       multivariate PCA buffer (2H+1)
                                              |
                               Stage-II two-window Hotelling at warning + H
                                              |
                                     AdaptationPolicy.should_update
                                              |
                       label availability -> append_and_retrain -> next prediction
```

Files: `src/online/replay.py` uses existing `src/detection/preprocessing.py`
and `src/detection/stage1/sd_ewma.py`. `src/online/validation.py` uses
`calculate_paper_two_sample_hotelling` from the existing paper-aligned
Stage-II module. `src/online/pipeline.py` reuses
`src/adaptation/classifier.py` and `src/adaptation/policies.py` rather than
reimplementing the classifier or policy.

## Reproduce online runs (with downloaded BCI raw files)

```bash
python scripts/run_online_bci.py --dataset 2a --subject 1 \
  --labels-2a data/raw/bci_competition_iv_2a_labels \
  --policy validated --label-delay 0

python scripts/run_online_bci.py --dataset 2b --subject 1 \
  --policy validated --label-delay 2
```

2A: Session I development training split and Session II evaluation.
2B: Sessions I–III development and Sessions IV–V evaluation, reusing the
existing `prepare_cse_uael_2019_protocol` builder. For 2B, evaluation
labels are generally unavailable in raw evaluation GDF; use
`--label-delay -1` to disable labelled adaptation, or supply official
labels to a prepared feature stream. Never invent GDF labels. The CLI should
fail explicitly if required evaluation labels are missing.

Control cases: `--policy never` and `--policy periodic`. Dataset labels
are used for post-run scoring, but are released for training only when their
configured availability time arrives; `--label-delay -1` withholds them
from the learner entirely.

The initial implementation is **synchronous supervised trial-level replay**.
It does not make zero-phase FBCSP filtering causal. It does not reproduce
published CSE-UAEL outcome numbers or model exoskeleton feedback. Existing
reproduction outputs and statistical functions are not modified.
