# Initial PhD assessment presentation — editorial curation

**Evidence checkpoint: 24 October 2026**  
**Source:** `presentations/EWMA_Covariate_Shift_Progress_Review_Haocheng_Lin.pptx` (55 slides; unchanged).  
**Derived deck:** `presentations/EWMA_Covariate_Shift_Initial_Assessment_Curated.pptx` (21 slides when the curation workflow completes).

## Editorial decision

Preserve the 55-slide presentation as the **full research appendix / learning diary**. The initial assessment should tell one coherent story: **problem → research lineage → implemented detector → evidence → unresolved issues → WP1 readiness → next test**.

The shorter deck retains the original mathematics, image-based plots, author attributions, and numerical values on selected slides. It adds one explicitly *synthetic editorial summary* of WP1 readiness, which is not experimental evidence. It does not edit the original slide files or underlying experimental outputs.

## Curated slide sequence

| Initial-assessment order | Original slide | Reason to retain |
| --- | --- | --- |
| 1 | 1 | Research title and dated checkpoint |
| 2 | 2 | Presentation guide |
| 3 | 3 | Research architecture |
| 4 | 5 | High-level literature question |
| 5 | 8 | Four direct predecessors; literature selection |
| 6 | 9 | Distinct statistical change targets |
| 7 | 25 | BCI online evidence and scope |
| 8 | 11 | Broader PhD programme beyond CSE |
| 9 | 32 | Fixed features versus classifier updates |
| 10 | 33 | Stage-I and Stage-II mathematics |
| 11 | 39 | BCI 2A/2B data separation |
| 12 | 36 | Mean, variance and dependence conceptual stress tests |
| 13 | 13 | Park/CPSS proof-of-concept |
| 14 | 43 | Six adaptation-policy configurations |
| 15 | 46 | Archived single-seed policy comparison |
| 16 | 47 | Classifier/complementary-monitor summaries |
| 17 | 48 | B01 reproducibility discrepancy |
| 18 | 12 | Implemented versus proposed evidence ledger |
| 19 | NEW | Five WP1 goals: current evidence and open deliverables |
| 20 | 52 | WP1–WP3 planning timeline (clearly a plan) |
| 21 | 54 | Next evidence milestone |

## Repetition and material reserved for questions

| Original slides | Change | Rationale |
| --- | --- | --- |
| 5–15 | Select 5, 8, 9, 11, 12 and 13; move the others to the archival deck | Avoid five overlapping maps of literature lineage and proposed architecture |
| 16–24, 34–37, 55 | Keep one mathematics slide (33) and one structural-change graphic (36); retain detailed derivations in the 55-slide archive | Important learning progress, but excessive for an initial assessment |
| 25–30 | Keep 25 (BCI context); other source-specific numerical and limitation slides remain available for examination | Preserve clinical relevance without letting a single paper dominate the main story |
| 31–33 | Keep 32 and 33 | Mechanistic detail without repeating feature-extraction diagrams |
| 38–40 | Keep 39 | Dataset eligibility and chronology are more important than a second pipeline chart |
| 41–46 | Keep 43 and 46 | One policy-design slide and one evidenced outcome; other policy slides can support questions |
| 47–51 | Keep 47 and 48; earlier evidence ledger 12 covers overall status | Preserve numerical evaluation and unresolved reproduction, avoid multiple progress recaps |
| 52–54 | Keep 52 and 54 | Timeline plus one actionable milestone |
| 4 | Archive pending visual inspection | Image-only slide: automated text inventory cannot determine its purpose |

## Claims to protect in delivery

- Code and successful tests demonstrate **software behaviour**, not reproducibility of the paper's subject-level numbers or statistical guarantees.
- Park/CPSS: held-out synthetic pilot, small null calibration, **tau = 0**; not a proven accuracy improvement at matched false-positive rates.
- The Diethe policy archive contains a **five-policy single-seed** comparison; the code supports six policy types but no six-policy completed multi-seed benchmark is established.
- B01 archived Stage-II confirmations disagree (**0 versus 5**) with no frozen matching run manifest; do not resolve it rhetorically.
- Complementary PCA score/residual experiments are synthetic; do not infer EEG change-point ground truth from motor-imagery labels.
- ABCD is an **annotated equation-level teaching baseline**, not reproduction of Heyden Algorithm 2. Howard-style time-uniform calibration is a theoretical research avenue.
- Full CPD–CL–digital-twin and forgetting/energy-budget objectives remain future research.

## Speaker emphasis (approximate distribution)

**Research question and architecture:** 15%; **mathematical and literature foundation:** 20%; **implementation and methodology:** 25%; **evaluated results and unresolved discrepancies:** 25%; **WP1 deliverables and next milestone:** 15%.

## Future maintenance

Update the assessment *evidence ledger*, *WP1 checkpoint* and *next milestone* when:
- the archived B01 0-versus-5 discrepancy is resolved by an immutable run manifest,
- the CPSS or ABCD results are calibrated and verified on held-out streams,
- the adaptation-policy laboratory has archived multi-seed/delayed-label/retention/energy results,
- the CPD–CL–digital-twin interface moves from proposal to a reviewed specification.

For all numerical claims, record the exact data protocol, seed(s), detector and model configuration, repository SHA, and saved event output. Keep the full 55-slide research archive unchanged.
