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
| Hurtado et al. (2023) | Continual learning for predictive maintenance; active drift-triggered versus passive continual adaptation, single-model versus ensemble strategies, and the need for realistic non-stationary PdM benchmarks | `src/adaptation/policies.py`, `src/web/process.py`, `src/web/data_structures.py` | `/process/adaptive-learning` now exposes the active/passive taxonomy; C-MAPSS/Turbofan inspection is available under `/data-structures/turbofan` | Taxonomy and policy primitives are implemented and surfaced; a full active-versus-passive predictive-maintenance experiment remains future work |
| Current repository extension: complementary PCA monitoring | Separate evidence from change within the retained PCA score space and outside it in the reconstruction residual | `src/detection/stage1/complementary.py`, `src/simulation/pca_subspace.py` | `outputs/complementary*/` and BCI complementary result views | Repository extension used to expose PC1-only monitoring blind spots |
| Current repository extension: confidence-gated transductive adaptation | Remove evaluation ground-truth labels from the online learner and admit pseudo-labelled observations only above a confidence threshold | `run_transductive_adaptation` / historical `run_unsupervised_adaptation` in `src/adaptation/transductive.py`, `PWKNNPseudoLabeler` in `src/adaptation/pseudo_labelling.py`, and post-run scoring in `src/adaptation/evaluation.py` | `tests/adaptation/test_unsupervised.py`; evaluation truth is isolated to post-run scoring | Implemented and regression-tested; first full 2A/2B adaptive experiment remains to be run |

## November 2026 implementation extensions and assessment links

These entries supplement (not replace) the historical paper-lineage table above. Each status is deliberately bounded to what code and archived sources support.

| Study / stage | Research question and provenance | Repository implementation | Evidence / interpretation | Status |
| --- | --- | --- | --- | --- |
| Gupta et al. (2024) TWIN-ADAPT-inspired BCI | RQ2/RQ3: predict-before-update, accuracy warning, labelled feedback and cost | [BCI controller](../../src/adaptation/twin_adapt/controller.py), [fixed runner](../../scripts/run_twin_adapt_bci.py), [optimiser](../../scripts/run_twin_adapt_optimization.py) | [BCI protocol](../twin_adapt_bci.md), [cost objective and experiment output guide](../twin_adapt_optimization.md) | Linear SVM operational interpretation; **not** the paper's LightGBM/PSO reproduction; single-run claims remain tentative |
| Matched BCI adaptation policy comparison (17 Nov 2026) | RQ2: compare update policies under common replay assumptions | [Matched runner](../../scripts/run_matched_bci_comparison.py), [experiment module](../../src/adaptation/experiments/matched_bci_comparison.py) | Match preprocessing, training starts, chronological predictions, label release, subject, seed and batch size before making policy comparisons | Runner implemented; no general superiority claim without repeated, verified outputs |
| Opt-in marginal KL Stage-II evidence (17–18 Nov 2026) | RQ1 exploratory: describe pre/post-warning distributions without modifying validation | [KL diagnostics](../../src/detection/stage2/kl_diagnostics.py), [BCI figure export](../../scripts/export_kl_bci_figures.py), [saved results gallery](../../src/web/kl_results.py) | [KL guide](../kl_stage2_diagnostics.md) and `/results/kl-stage2`; bin calibration on development data, output manifest and ordinal 2B time semantics | Descriptive diagnostic and presentation exports; **not** a statistical confirmation test, real-EEG accuracy gain, or substitute for Hotelling |
| Multivariate structural-change exploration (16–17 Nov 2026) | Explore mean and second-order change, dynamic PCA/wavelet literature | [Multivariate inspection module](../../src/detection/multivariate_inspection.py) | [Park inspection guide](../park_multivariate_inspection.md) | Experimental work/stubs, not a validated reproduction of the published wavelet method |
| Initial PhD assessment claim provenance (19 Nov 2026) | RQ1–RQ3: connect four-page research narrative to exact experiments | [Table 1 runner](../../scripts/run_bci_table1_reproduction.py), [baseline audit](../../outputs/baseline_2026_10_12/README.md), [output directory guide](../../outputs/README.md) | Published B01 14 CSW / 10 CSV vs archived 17 CSW / 0 or 5 CSV is unresolved; archived aggregate subject means depend on output provenance | Literature-supported narrative and executable infrastructure; not a full paper-matched multi-subject replication |

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
