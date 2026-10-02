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
