# Living computational literature review

This repository is maintained as more than a final implementation of one
covariate-shift detector. It is a computational record of how ideas in
non-stationary learning, EEG covariate-shift detection and adaptive BCI develop
across the literature and across the repository's own reproductions.

The preservation model has three parallel histories:

1. **Literature history** - what each study contributed, assumed, measured and
   left unresolved.
2. **Algorithmic history** - how the method changes from detector, to
   validation, to adaptation, to continual-learning policy.
3. **Implementation history** - how those ideas were encoded, revised,
   superseded, compared and extended in this repository.

The intended evidence chain is:

```text
paper / research question
        ↓
methodological interpretation
        ↓
implementation
        ↓
experiment
        ↓
saved evidence
        ↓
limitation / discrepancy
        ↓
next implementation or study
```

## Navigation

- [studies.md](studies.md) maps the major studies to code and evidence.
- [implementation_lineage.md](implementation_lineage.md) records how modules
  and functions evolved without erasing earlier research layouts.
- [provenance_policy.md](provenance_policy.md) defines canonical,
  historical, paper-inspired and extension statuses.
- [study_template.md](study_template.md) is the required structure for adding
  a new study to the tracker.

## Core chronological lineage

```text
Raza et al. (2015)
EWMA shift-detection foundations
        ↓
Chowdhury et al. (2018)
online adaptive BCI: EEG → CSP/PCA → warning → validation → classifier update
        ↓
Raza et al. (2019)
filter-bank features + CSE + PWKNN + adaptive ensemble learning
        ↓
Diethe et al. (2019)
continual-learning architecture: monitoring → policy → horizon → update cost
        ↓
repository extensions
complementary PCA score/residual evidence
policy comparison and adaptation auditing
confidence-gated PWKNN transductive updates
        ↓
future work
dynamic ensemble growth, weighted voting, label-availability experiments,
representation-aware and need-based adaptation
```

This diagram is a research lineage, not a claim that every later component is
a direct extension proposed by every earlier paper. Paper reproductions,
paper-inspired experiments and repository-originated extensions are labelled
separately in the study tracker.

## Current assessment evidence map (19 November 2026)

The initial PhD assessment asks three linked but non-equivalent questions. **RQ1** concerns reconstructing Raza's detector and explaining published-versus-computed counts; **RQ2** concerns the decision to update a classifier after predictive deterioration; **RQ3** concerns the accuracy-versus-update-cost trade-off. Detection of `P(X)` change alone does not prove classifier harm, nor does an executable retraining policy prove that updates are beneficial.

| Question | Literature | Implementation | Evidence and limitations |
| --- | --- | --- | --- |
| RQ1: detector reproducibility | Raza et al. (2015, 2019); Chowdhury et al. (2018) | [CSE core](../../src/detection/core.py), [Table 1 runner](../../scripts/run_bci_table1_reproduction.py), [Stage-II tests](../../src/detection/stage2/) | [Comparison CSV](../../outputs/metrics/bci_table1_comparison.csv) and [baseline audit](../../outputs/baseline_2026_10_12/README.md); **B01 14/10 published, computed 17/0 versus 17/5 unresolved** |
| RQ2: when to adapt | Diethe et al. (2019); Gupta et al. (2024) | [Policies](../../src/adaptation/policies.py), [accuracy controller](../../src/adaptation/twin_adapt/controller.py), [matched BCI runner](../../scripts/run_matched_bci_comparison.py) | [TWIN-ADAPT BCI protocol](../twin_adapt_bci.md); chronological predictions and feedback semantics must be held constant for comparative claims |
| RQ3: adaptive cost | Gupta et al. (2024); Hurtado et al. (2023) | [Optimisation runner](../../scripts/run_twin_adapt_optimization.py) | [Development-only optimisation guide](../twin_adapt_optimization.md); illustrative objective and single-run outputs are not proof of robust superiority |
| Exploratory change evidence | PCA score/residual and marginal KL approaches | [Complementary PCA](../../src/detection/stage1/complementary.py), [KL diagnostics](../../src/detection/stage2/kl_diagnostics.py) | [KL figure export and limitations](../kl_stage2_diagnostics.md); KL is descriptive, does not alter Hotelling `confirmed_shift`, and never triggers SVM updates |

**Publication and assessment guardrail:** include the Git SHA, command/configuration manifest, dataset/session roles, seed, run ID, output path and status (implemented, diagnostic, measured, tentative, unresolved, validated) when mapping claims to evidence. The 18 November KL figure exports are presentation artifacts derived from source data, not new classifier results. Newer work does not retrospectively resolve the B01 Stage-II conflict or prove numerical agreement with the original papers.

## Preservation principle

The repository favours **one canonical runnable implementation plus preserved
meaningful alternatives**. Earlier code should not remain active merely because
it is old, but it should not be deleted when it records an important change in
method interpretation. Compatibility shims, archived workflows, selected
milestone outputs and paper-specific variants can therefore be deliberate
research assets rather than technical debt.

A new refactor should answer two questions:

- What becomes the canonical path?
- What historical meaning would be lost if the previous path disappeared?

If the second answer is substantive, preserve the earlier path with provenance
rather than deleting it silently.
