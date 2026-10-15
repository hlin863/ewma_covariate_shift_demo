#!/usr/bin/env python3
"""Convert image-filled formula groups into ordinary PowerPoint pictures.

Some equation graphics in the October 2026 presentation are PNG images applied
as fills to FREEFORM elements inside GROUP shapes. Several viewers omit those
shapes. This repair REUSES THE EXACT EMBEDDED IMAGE BYTES instead of
reconstructing mathematical expressions.

The original PPTX is never overwritten. It creates a distinct repaired PPTX.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
FORMULA_SLIDES = {9, 10, 11, 12, 13, 20, 22, 23, 26}


def _slide_text(slide) -> str:
    return "\n".join(shape.text for shape in slide.shapes if shape.has_text_frame)


def _source_image(slide, group):
    """Return original embedded PNG/JPEG bytes for an image-fill-only freeform."""
    if group.shape_type != MSO_SHAPE_TYPE.GROUP or len(group.shapes) != 1:
        return None
    child = group.shapes[0]
    if child.shape_type != MSO_SHAPE_TYPE.FREEFORM:
        return None
    if child.has_text_frame and child.text.strip():
        return None
    refs = child._element.xpath(".//*[local-name()='blip']")
    if len(refs) != 1:
        return None
    rid = refs[0].get(EMBED)
    if not rid or rid not in slide.part.rels:
        return None
    relation = slide.part.rels[rid]
    if relation.is_external:
        return None
    part = relation.target_part
    if part.content_type not in ("image/png", "image/jpeg"):
        return None
    raw = part.blob
    if len(raw) < 100:
        return None
    return raw, str(part.partname)


def repair(source: Path, destination: Path, slides: set[int] | None = None) -> list[dict]:
    deck = Presentation(str(source))
    baseline_text = [_slide_text(s) for s in deck.slides]
    if slides is None:
        slides = FORMULA_SLIDES
    if max(slides) > len(deck.slides):
        raise ValueError(f"Deck has {len(deck.slides)} slides; expected at least {max(slides)}.")

    restored = []
    for slide_no in sorted(slides):
        slide = deck.slides[slide_no - 1]
        # Snapshot list: add_picture mutates the shape tree.
        for group in list(slide.shapes):
            payload = _source_image(slide, group)
            if payload is None:
                continue
            image, media_path = payload
            before = group._element
            parent = before.getparent()
            layer = parent.index(before)
            # Same geometry as the original GROUP, no mathematical retyping.
            picture = slide.shapes.add_picture(
                io.BytesIO(image), group.left, group.top,
                width=group.width, height=group.height,
            )
            parent.remove(picture._element)
            parent.insert(layer, picture._element)
            parent.remove(before)
            restored.append({
                "slide": slide_no,
                "media": media_path,
                "sha256": hashlib.sha256(image).hexdigest(),
                "x_emu": picture.left, "y_emu": picture.top,
                "width_emu": picture.width, "height_emu": picture.height,
            })

    if not restored:
        raise RuntimeError("No embedded formula images were found; refusing to create a misleading output.")
    assert len(deck.slides) == len(baseline_text)
    assert [_slide_text(s) for s in deck.slides] == baseline_text, "Slide text unexpectedly changed."
    destination.parent.mkdir(parents=True, exist_ok=True)
    deck.save(str(destination))

    # Reopen the resulting PowerPoint and verify that the new, conventional
    # picture shape points to identical image bytes at identical coordinates.
    checked = Presentation(str(destination))
    for row in restored:
        slide = checked.slides[row["slide"] - 1]
        matches = [
            sh for sh in slide.shapes
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE
            and sh.left == row["x_emu"] and sh.top == row["y_emu"]
            and sh.width == row["width_emu"] and sh.height == row["height_emu"]
            and hashlib.sha256(sh.image.blob).hexdigest() == row["sha256"]
        ]
        if not matches:
            raise AssertionError(f"Image verification failed for slide {row['slide']} / {row['media']}")
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--all-image-groups", action="store_true",
                        help="Repair compatible image-filled groups across all slides, not just math slides")
    args = parser.parse_args()
    from pptx import Presentation
    all_slides = set(range(1, len(Presentation(str(args.source)).slides) + 1))
    selected = all_slides if args.all_image_groups else FORMULA_SLIDES
    rows = repair(args.source, args.destination, selected)
    print(f"RESTORED {len(rows)} embedded PNG/JPEG equation graphics; source remains unchanged.")
    print("REPAIRS_PER_SLIDE", dict(sorted(Counter(r["slide"] for r in rows).items())))
    for r in rows:
        print(f"Slide {r['slide']:2} preserved {r['media']} sha256={r['sha256'][:16]}...")
    print("Output:", args.destination)


if __name__ == "__main__":
    main()
