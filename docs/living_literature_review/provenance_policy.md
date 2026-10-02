# Provenance and preservation policy

The repository balances two goals: a clear canonical implementation for current
experiments and preservation of meaningful research history.

## Status vocabulary

Use these labels in documentation and commit messages where useful:

- **canonical** - preferred active implementation for new experiments;
- **compatibility shim** - older import path forwarding to canonical code;
- **historical** - superseded implementation retained for methodological provenance;
- **paper reproduction** - source-grounded attempt to reconstruct a published method/result;
- **paper-inspired** - uses a paper's question or architecture with repository-specific choices;
- **repository extension** - functionality introduced by this project rather than claimed as published;
- **planned** - literature-supported or research-motivated stage not yet implemented.

## Preserve when

Retain an earlier implementation, result or script when at least one is true:

1. it represents a materially different interpretation of a paper;
2. it uses a historically important dataset/feature protocol;
3. its result explains why the current implementation changed;
4. it is needed to reproduce a milestone reported in notes, slides or manuscript drafts;
5. it provides a stable compatibility path for older notebooks or experiments.

## Do not preserve as active code when

A duplicate contains no distinct methodological meaning, is generated clutter,
or would create two competing canonical implementations. In that case either
delete it through normal Git history or move it to `archive/` only if the
historical context remains useful.

## Adding a new study

A literature-driven implementation should add or update:

1. the study row in `studies.md`;
2. a paper-specific note when interpretation is non-trivial;
3. the source entry in `papers/README.md` if a PDF is legally/reliably stored
   in the repository, otherwise mark the source as external;
4. code links for the implemented algorithmic contribution;
5. saved evidence or an experiment command where available;
6. the remaining gap so future work is explicit.

Use [study_template.md](study_template.md) for the entry structure.

## Refactoring rule

Refactoring may change the canonical location without erasing history. Before
moving or replacing research code, record:

- old path;
- new path;
- whether behaviour is intended to be equivalent;
- methodological reason for the move/change;
- tests or outputs used to establish continuity.

A compatibility shim is preferred when older notebooks/scripts still matter.
An archive note is preferred when behaviour itself has been superseded.

## Output preservation

Commit milestone outputs when they support a documented research conclusion,
reproduction discrepancy or methodological comparison. Large regenerable sweeps
should normally remain outside Git unless they are necessary evidence. Every
retained result should, where practical, record configuration, seed/dataset,
source commit and command or runner.
