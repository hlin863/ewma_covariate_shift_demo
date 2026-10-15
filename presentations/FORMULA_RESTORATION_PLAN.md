# Equation provenance and compatibility diagnosis

**Corrected 15 October 2026:** The prior conclusion that mathematical
expressions were missing from the PPTX was incorrect.

## Verified result

The 44-slide committed deck embeds the user-authored mathematics as PNG
assets inside **image-filled freeform shapes within PowerPoint groups**.
The groups do not appear as regular `p:pic` elements or `m:oMath`
equation nodes, so an inventory limited to those structures falsely
reported zero equations. There is no basis for asking the author to
retype mathematical formulas or replacing them with transcriptions
from papers. The preserved images in the deck are the source of truth.

The first compatibility repair accidentally omitted physical slides 7 and 8. These are now included in the source-asset conversion and asserted by CI; no formula re-entry is needed.

Relevant physical slides and linked media in the audited 15 October
presentation (commit `69ae8d4`):

| Slides | Subject | Existing formula PNGs |
|---|---|---|
| 9 | Kronecker delta | `image4.png` |
| 10 | Covariance, independence | `image5.png` |
| 11 | Coordinate derivative, variance, online update | `image6.png`–`image8.png` |
| 12 | Wavelet moments | `image9.png` |
| 13 | Transfer matrix, covariance | `image10.png`–`image11.png` |
| 20 | Feature extraction/CSP/PCA | `image12.png`–`image14.png` |
| 22 | Stage-I EWMA and Stage-II Hotelling | `image16.png`–`image20.png` |
| 23 | Two different T-squared statistics | `image21.png`–`image22.png` |
| 26 | Mean-test structural-change limitation | `image25.png` |

Those images were checked as embedded PNG bytes with valid image
relationships, not missing external references. Exact display on
every slide viewer remains a **rendering check** to complete.

## Non-destructive viewer-compatibility repair

`scripts/create_equation_compatible_deck.py` reads the original PPTX
and changes eligible one-image freeform groups into ordinary `p:pic`
picture shapes **in a new presentation copy**, using the original PNG
bytes, size, approximate bounding rectangle and slide drawing order.
It does not change, derive, or retype the mathematics; it refuses to
overwrite the source deck. The corresponding GitHub Actions workflow
produces the candidate copy as a downloadable artifact and audits its
equation-picture count and existing slide text.

The conversion must be visually reviewed, especially for clipping and
image cropping, in PowerPoint or a PDF export before replacing the
official deck. The existing PPTX is preserved unchanged on this PR.

## Correct audit rule

A comprehensive formula-media inventory must count:

1. native Office Math (`m:oMath`, `m:oMathPara`);
2. standard picture elements (`p:pic`);
3. **image-filled custom/freeform PowerPoint shapes** (`a:blipFill`
   inside `p:sp`), including shapes nested in `p:grpSp`);
4. relevant embedded media relationships, rather than relying on
   extracted text alone.

A zero native-math count does **not** imply zero formulas.
