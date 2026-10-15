# Slide 7–8 formula repair (Park et al., 2023)

## Verified cause

The original slide 7 contains one PNG/JPEG-backed freeform-group equation and
two purely decorative freeform groups. The original slide 8 contains *only two
decorative panel groups*, with **no embedded formula images**. The initial
automated repair skipped both slides and wrongly counted grouped shapes as
equations. The formulas on slide 8 cannot be recovered byte-for-byte from the
current saved slide because no equation assets exist there.

## Bounded repair

- Slide 7: preserve the existing author-supplied formula JPEG as a conventional
  picture, with byte-for-byte source image identity.
- Slide 8: reconstruct **only the displayed aggregation/smoothing equations**
  in its two existing panels using the project's current
  `src/detection/stage1/park_mean_cusum.py` and Park et al. (2023),
  Section 2.1 / Algorithm 1. These are source-grounded *reconstructions*,
  **not claimed as exact copies of the author's original slide images**.
- Preserve every slide's existing textual content and all other experimental
  and research objects.
- Save to `EWMA_Covariate_Shift_Progress_Review_FORMULAS_RESTORED.pptx`
  on the diagnostic branch only; the original file on main stays unchanged.

## Formula attribution

Park et al. (2023), Section 2.1:
\(\nu_t^{\max}=\max_d\nu_t^{(d)}\) and
\(\nu_t^{\mathrm{avg}}=p^{-1}\sum_d\nu_t^{(d)}\).
Algorithm 1 Step 2:
local-polynomial smoothing of each aggregation curve (span \(\alpha=0.1\))
followed by \(b_{\max}=\arg\max_t\tilde\nu_t^{\max}\) and
\(b_{\mathrm{avg}}=\arg\max_t\tilde\nu_t^{\mathrm{avg}}\).
\(\mathcal S_\alpha\) denotes the smoothing operation illustratively;
it is **not claimed as the paper's own operator notation**.

Pass automated image-integrity/layout-preservation checks before sharing
the candidate deck. A real PowerPoint visual review is still needed.
