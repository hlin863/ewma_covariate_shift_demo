# Research process and results navigation

The home page is a compact route into the research process. Its three
process cards describe detection, confirmation and adaptation. Dataset
inspection and protocol preparation appear before these steps. Progress,
paper lineage, contributions and open questions remain available in
expandable sections.

| Process | Page | Purpose |
|---|---|---|
| Covariate shift | `/process/covariate-shift` | Feature preparation and Stage-I warning logic |
| Validation | `/process/validation` | Stage-II choices, reference data and eligibility |
| Adaptive learning | `/process/adaptive-learning` | Update policy, label boundaries and retention |

The shared **Data structures**, **Process** and **Results** menus are
available throughout the application. Existing URLs continue to work.
Process pages show methods and link to relevant tools; saved evidence
is indexed separately at `/results` by detection, validation/reproduction,
learning/classifiers and implementation checks.

Diagnostic counts and classifier accuracy previously embedded in
`/data-processing` now appear at `/results/data-processing`. The original
observed section, dataset switch and provenance labels are preserved.
The former `#observed` section links to that result page. No saved data,
experiment configuration or numerical algorithms are changed.

The interactive Diethe policy laboratory continues to show the outcome
of its current run. Automated tests remain a separate evidence category
from scientific reproduction and held-out model evaluation.
