# Archive

This directory preserves superseded exploratory workflows when they remain
useful for understanding the development of the research. Archived code is
**not** part of the canonical execution path, but it is part of the repository's
implementation history.

The preservation rule is: archive a workflow when it records a meaningful
earlier methodological interpretation, dataset protocol, feature definition or
experiment that helps explain why the canonical implementation changed. Do not
archive accidental duplicates or generated clutter.

For every substantial archived workflow, preserve enough context to answer:

1. Which study or research question motivated it?
2. What assumptions did the implementation make?
3. What result or diagnostic did it produce?
4. Why was it superseded?
5. Which canonical code now replaces it?

## Current archived lineage

`archive/legacy_bci_2b/` preserves the earlier overlapping-window band-power
Dataset 2B workflow. It targeted a different interpretation and depended on
obsolete processed `.npz` files. It must not be used for the current
paper-pipeline reproduction, but it remains useful for tracing how the Dataset
2B interpretation evolved.

The current Dataset 2B path is represented by the structured
`src/bci/datasets/dataset2b/` package and the active experiment scripts.

See [docs/living_literature_review/provenance_policy.md](../docs/living_literature_review/provenance_policy.md)
for the repository-wide preservation policy.
