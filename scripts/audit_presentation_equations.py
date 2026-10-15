#!/usr/bin/env python3
"""Read-only OOXML equation-inventory audit for the research slide deck."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

SLIDE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
MATH = "http://schemas.openxmlformats.org/officeDocument/2006/math"
DRAW = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRES = "http://schemas.openxmlformats.org/presentationml/2006/main"
M14 = "http://schemas.microsoft.com/office/drawing/2010/main"
TEX = re.compile(r"\\(?:frac|sqrt|sum|int|lambda|mu|sigma|beta|alpha|theta|begin|mathbb|mathbf|left|right)\b|\$[^$]{2,}\$")

def inventory(path: Path) -> list[dict]:
    with ZipFile(path) as archive:
        entries = sorted(
            (int(m.group(1)), key)
            for key in archive.namelist()
            if (m := SLIDE.fullmatch(key))
        )
        if not entries:
            raise ValueError(f"{path}: No slides")
        result = []
        for n, name in entries:
            root = ET.fromstring(archive.read(name))
            nodes = list(root.iter())
            strings = [e.text or "" for e in nodes if e.tag == f"{{{DRAW}}}t"]
            title = next((s.strip() for s in strings if len(s.strip()) > 5), "(empty)")
            result.append({
                "slide": n,
                "title": title[:100],
                "native_math": sum(e.tag == f"{{{MATH}}}oMath" for e in nodes),
                "math_paragraph": sum(e.tag == f"{{{MATH}}}oMathPara" for e in nodes),
                "office14_math": sum(e.tag == f"{{{M14}}}m" for e in nodes),
                "pictures": sum(e.tag == f"{{{PRES}}}pic" for e in nodes),
                "ole": sum(e.tag.rsplit("}",1)[-1] in ("oleObj","oleObject") for e in nodes),
                "text_runs": len(strings),
                "slide_text": " | ".join(s.strip().replace("\\n", " ") for s in strings if s.strip())[:4000],
                "raw_tex": [s[:140] for s in strings if TEX.search(s)],
            })
    return result

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("deck",type=Path)
    p.add_argument("--baseline",type=Path)
    p.add_argument("--json",type=Path)
    args=p.parse_args()
    current=inventory(args.deck)
    baseline=inventory(args.baseline) if args.baseline else None
    print(f"Current deck: {len(current)} slides")
    for row in current:
        print(f"{row['slide']:>2} {row['title'][:44]:44} nativeMath={row['native_math']:>2} mathP={row['math_paragraph']:>2} a14Math={row['office14_math']:>2} ole={row['ole']:>2} pictures={row['pictures']:>2} rawTeX={len(row['raw_tex']):>2}")
        if row["slide"] in {8,9,10,11,12,13,22,23,24,25,26,43}:
            print(f"      SLIDE_TEXT: {row['slide_text']}")
        for s in row["raw_tex"][:3]:
            print(f"      CHECK UNRENDERED SOURCE: {s!r}")
    risks=[]
    if baseline is not None:
        old_by_num={r["slide"]:r for r in baseline}
        unique_old_titles={}
        for r in baseline:
            unique_old_titles.setdefault(r["title"],[]).append(r)
        print(f"\nBaseline: {len(baseline)} slides")
        for old in baseline:
            if old["slide"] in {8,9,10,11,12,13,22,23,24,25,26,43}:
                print(f"BASELINE_TEXT {old['slide']}: {old['slide_text']}")
        for r in current:
            match=unique_old_titles.get(r["title"],[])
            old=match[0] if len(match)==1 else old_by_num.get(r["slide"])
            if old is None: continue
            decreases={k:(old[k],r[k]) for k in ("native_math","math_paragraph","office14_math","ole","pictures") if r[k]<old[k]}
            if decreases:
                risks.append({"slide":r["slide"],"title":r["title"],"baseline_slide":old["slide"],"decreased":decreases})
                print(f"REVIEW POTENTIAL LOSS slide {r['slide']} (baseline {old['slide']}): {r['title']} | {decreases}")
        print(f"Potential loss candidates: {len(risks)}")
    if args.json:
        args.json.parent.mkdir(exist_ok=True,parents=True)
        args.json.write_text(json.dumps({"current":current,"baseline":baseline,"potential_losses":risks},indent=2,ensure_ascii=False),encoding="utf-8")
    # Deliberately never auto-fail on reductions: rewritten slides may intentionally change objects.
    # This is structural auditing, not visual equation-rendering verification.

if __name__ == "__main__":
    main()
