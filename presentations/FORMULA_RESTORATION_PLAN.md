# Restoring mathematical notation in the progress-review deck

**Deck:** `presentations/EWMA_Covariate_Shift_Progress_Review_Haocheng_Lin.pptx`  
**Reference:** `main` at `69ae8d4` (15 October 2026)

## Verified defect and limits

The October 15 deck has 44 physical slides and **zero PowerPoint native equations** (Office Math / OMML) and zero legacy OLE equation objects. The 14 October (43-slide) and 11 October (34-slide) decks also have zero native equations. These inventories do **not** establish that full equations once existed and were deleted on 15 October. The relevant mathematical slides do contain prose and some Unicode mathematical characters, but the complete expressions described below are not present as regular extracted text.

**Repair the saved PPTX, not the statistical implementation.** Use real PowerPoint equation objects (Insert → Equation, Alt+=, Cambria Math) and preserve semantic source, or use labelled vector SVG equations if repeatable PowerPoint rendering is required. Do not conflate text written with Greek symbols with editable Office Math.

## Slide-by-slide restoration checklist

| Physical slide (15 Oct) | Slide topic | Candidate mathematical expression or source to verify | Boundary |
|---|---|---|---|
| 9 | Kronecker delta | `δ_ij = 1 if i=j; 0 otherwise` | General mathematical identity; do not label as a verbatim Park equation |
| 10 | Covariance and independence | `Cov(Z_i,Z_j)=E[(Z_i-EZ_i)(Z_j-EZ_j)]`; for centred, unit-variance, uncorrelated innovations, `Cov(Z_i,Z_j)=δ_ij` | The covariance identity does not itself imply independence |
| 11 | Coordinate derivative versus variance | `∂w_i/∂w_j = δ_ij` (independent coordinates), and `Var(X)=E[(X-E X)^2]` | Preserve distinction; confirm separate online update rule before insertion |
| 12 | Daubechies wavelets | `∫ t^k ψ(t) dt = 0` for `k=0,…,N−1` | Background vanishing-moment identity; verify specific filter assumptions |
| 13 | Transfer matrices and observed covariance | `Cov(AZ) = A Cov(Z) A^T` for deterministic `A` | Explanatory linear transform, not a substitute for Park's full wavelet model |
| 22 | Detector mathematics | Restore the actual EWMA state/error/control-limit formulae and paper variance update (Chowdhury §II.E.3, Eqs. 3, 5–9); separately identify the repository two-sample Stage-II variant | Do not silently replace the paper's separate-covariance expression with the repository pooled variant |
| 23 | Two statistics called T² | Multivariate EWMA `T²_t=(z_t−μ_0)^T Σ_{z_t}^{−1}(z_t−μ_0)`; pooled two-sample Hotelling `T²=(n_1 n_2/(n_1+n_2)) (x̄_1−x̄_2)^T S_p^{−1}(x̄_1−x̄_2)` | Different reference objects and calibration assumptions |
| 26 | Mean validation limitation | Separate-covariance mean contrast `Q=(x̄_1−x̄_2)^T(S_1/n_1+S_2/n_2)^{−1}(x̄_1−x̄_2)` | Not a direct equality-of-covariances test; pooled F calibration is not transferable automatically |

Other slides (notably the Park channel-aggregation/smoothing and CSE pipeline pages) should be checked against the cited research text before inserting new expressions. An explanatory formula is not necessarily the original paper's exact equation.

## Minimum acceptance checks

1. Keep the original deck untouched until the restored PPTX has been visually reviewed.
2. On the listed slides, insert at least the intended equations as native Office Math or SVG vector objects. Retain an editable LaTeX/UnicodeMath source manifest under `presentations/`.
3. Export a PDF from PowerPoint and visually check all affected slides for empty expressions, clipping, misplaced baselines, missing Greek glyphs, and small font sizes.
4. Run `python scripts/audit_presentation_equations.py presentations/EWMA_Covariate_Shift_Progress_Review_Haocheng_Lin.pptx --baseline /path/to/previous.pptx --json formula_audit.json`; review both counts and source text, not only the workflow pass/fail.
5. Update slide-number references when the deck structure changes. The historic directory `outputs/structural_changes/slide22/` now maps to slides 24–26, **not** to physical slide 22.
6. Preserve all original experimental results. Re-creating a missing equation in slides is a presentation repair, **not** a new verified reproduction of the referenced paper.

This plan records **candidate restoration notation**. Equation numbering, parameter choices, and paper-specific forms must be confirmed against their cited sources before being declared exact reproductions.
