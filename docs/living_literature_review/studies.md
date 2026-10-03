# Study tracker

This table is the central literature-to-code map. Status describes what the
repository currently implements; it does not imply that an exact reproduction
has been achieved unless explicitly stated.

| Study / stage | Literature contribution tracked here | Repository representation | Evidence / notes | Current status |
| --- | --- | --- | --- | --- |
| Raza et al. (2015) | SD-EWMA, MSD-EWMA and two-stage covariate-shift detection in non-stationary environments | `src/detection/stage1/sd_ewma.py`, `src/detection/stage1/msd_ewma.py`, `src/detection/two_stage.py`, `src/experiments/paper2015/` | `docs/d2_reproduction.md`, `outputs/metrics/paper2015/`; source PDF in `papers/2015-raza-ewma-covariate-shift.pdf` | Reproduction and sensitivity work implemented; discrepancies remain part of the research record |
| Chowdhury et al. (2018) | Online adaptive BCI combining EEG feature processing, covariate-shift warning/validation and classifier adaptation | `src/bci/datasets/chowdhury/`, Chowdhury FBCSP variant in `src/bci/fbcsp.py`, supervised adaptation pathway in `src/adaptation/supervised.py` | source PDF in `papers/2018-chowdhury-online-adaptive-bci.pdf`; web/metadata views under `src/web/chowdhury.py` | Method components reconstructed and used for comparison; not presented as a complete clinical reproduction |
| Raza et al. (2019) | Filter-bank CSP/PCA CSE pipeline, PWKNN knowledge acquisition and unsupervised adaptive ensemble learning | `src/bci/fbcsp.py`, `src/detection/core.py`, Stage-II variants under `src/detection/stage2/`, PWKNN in `src/adaptation/pseudo_labelling.py`, and confidence-gated transductive adaptation in `src/adaptation/transductive.py` | Table-1 outputs, `docs/table1_algorithm_gap_analysis.md`, source PDF in `papers/2019-raza-cse-uael.pdf` | CSE and first PWKNN knowledge-base update stage implemented; dynamic LDA ensemble growth and weighted voting remain future work |
| Diethe et al. (2019) | Continual-learning reference architecture: monitoring, model-performance evidence, update policy, training horizon and adaptation cost | `src/adaptation/policies.py`, `src/adaptation/experiments/diethe.py`, `src/web/diethe.py` | `docs/diethe_experiments.md`; source is tracked externally rather than stored in `papers/` | Paper-inspired policy laboratory, explicitly not a reproduction of a Diethe algorithm |
| Current repository extension: complementary PCA monitoring | Separate evidence from change within the retained PCA score space and outside it in the reconstruction residual | `src/detection/stage1/complementary.py`, `src/simulation/pca_subspace.py` | `outputs/complementary*/` and BCI complementary result views | Repository extension used to expose PC1-only monitoring blind spots |
| Current repository extension: confidence-gated transductive adaptation | Remove evaluation ground-truth labels from the online learner and admit pseudo-labelled observations only above a confidence threshold | `run_transductive_adaptation` / historical `run_unsupervised_adaptation` in `src/adaptation/transductive.py`, `PWKNNPseudoLabeler` in `src/adaptation/pseudo_labelling.py`, and post-run scoring in `src/adaptation/evaluation.py` | `tests/adaptation/test_unsupervised.py`; evaluation truth is isolated to post-run scoring | Implemented and regression-tested; first full 2A/2B adaptive experiment remains to be run |

## How to read the tracker

Three labels should stay distinct:

- **Reproduction**: attempts to reconstruct a published algorithm or result.
- **Paper-inspired experiment**: applies a paper's architecture or question but
  adds repository-specific choices not claimed by the paper.
- **Repository extension**: a new experimental mechanism or comparison developed
  in this project.

When a new paper is added, record both what has been implemented and what has
*not* been implemented. The missing portion is part of the living review because
it defines the next research gap.
