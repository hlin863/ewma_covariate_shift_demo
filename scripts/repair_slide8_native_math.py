"""Native editable OfficeMath for Park et al. (2023) Slide 8.

Two uses:
* add_equation(): future slide-8 regeneration (no equation pictures).
* repair(): replace seven image equations in the existing 44-slide PowerPoint.

Only slide8.xml is changed in the repair output. No data, other slides,
drawings, relationships, media or OfficeMath on other slides is modified.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import sys
import zipfile

from lxml import etree
from pptx import Presentation
from pptx.enum.text import PP_ALIGN

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A14 = "http://schemas.microsoft.com/office/drawing/2010/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
SLIDE_XML = "ppt/slides/slide8.xml"


def sub(base, value): return ("sub", base, value)
def sup(base, value): return ("sup", base, value)
def both(base, low, high): return ("both", base, low, high)
def frac(num, den): return ("frac", num, den)
def radical(body): return ("radical", body)
def nary(symbol, low, high, body): return ("nary", symbol, low, high, body)
def accent(base, character): return ("accent", base, character)
def seq(*parts): return list(parts)


# Each original picture slot has one transcribed and semantically structured
# OfficeMath expression. The scientific notation is from Park Algorithm 1.
EQUATIONS = {
    "Slide 8 / step1 / equation 1": seq(both("ν","t","max")," = ",sub("max","1≤d≤p")," ",both("ν","t","(d)")),
    "Slide 8 / step1 / equation 2": seq(both("ν","t","avg")," = ",frac("1","p")," ",nary("∑","d=1","p",both("ν","t","(d)"))),
    "Slide 8 / step2 / equation 1": seq(accent(both("ν","t","g"),"̃")," = ",sub("LP","0.1"),"(",sup("ν","g"),")(t)"),
    "Slide 8 / step2 / equation 2": seq(sub("b","g")," = ",sub("arg max","t")," ",accent(both("ν","t","g"),"̃")),
    "Slide 8 / step3 / equation 1": seq(sup("A","max"),sup("A","avg")," = 0"),
    "Slide 8 / step4 / equation 1": seq("|",sup("b","max")," − ",sup("b","avg"),"| < ",frac(radical("T"),"2")),
    "Slide 8 / step4 / equation 2": seq(accent("b","̂")," = ",frac(seq(sup("b","max")," + ",sup("b","avg")),"2")),
}


def element(parent, local_name):
    return etree.SubElement(parent, f"{{{M}}}{local_name}")


def emit(parent, value):
    if isinstance(value, str):
        element(element(parent,"r"),"t").text = value
    elif isinstance(value, list):
        for part in value: emit(parent, part)
    elif isinstance(value, tuple):
        kind = value[0]
        if kind in ("sub", "sup", "both"):
            node = element(parent, {"sub":"sSub","sup":"sSup","both":"sSubSup"}[kind])
            emit(element(node,"e"),value[1])
            if kind in ("sub","both"): emit(element(node,"sub"),value[2])
            if kind in ("sup","both"): emit(element(node,"sup"),value[2 if kind=="sup" else 3])
        elif kind == "frac":
            node=element(parent,"f")
            emit(element(node,"num"),value[1]); emit(element(node,"den"),value[2])
        elif kind == "radical":
            node=element(parent,"rad")
            element(element(node,"radPr"),"degHide").set(f"{{{M}}}val","1")
            element(node,"deg")
            emit(element(node,"e"),value[1])
        elif kind == "nary":
            node=element(parent,"nary")
            element(element(node,"naryPr"),"chr").set(f"{{{M}}}val",value[1])
            emit(element(node,"sub"),value[2])
            emit(element(node,"sup"),value[3])
            emit(element(node,"e"),value[4])
        elif kind == "accent":
            node=element(parent,"acc")
            element(element(node,"accPr"),"chr").set(f"{{{M}}}val",value[2])
            emit(element(node,"e"),value[1])
        else:
            raise ValueError(f"Unsupported equation tree: {value}")
    else:
        raise TypeError(value)


def add_equation(slide, name, left, top, width, height, *, font_pt=None):
    """Insert a real editable OfficeMath shape; never add an image."""
    if name not in EQUATIONS: raise ValueError(f"Unregistered equation slot: {name}")
    box=slide.shapes.add_textbox(left,top,width,height)
    box.name=name.replace(" / equation "," / EDITABLE OfficeMath equation ")
    frame=box.text_frame
    frame.clear()
    frame.word_wrap=False
    frame.margin_left=frame.margin_right=0
    frame.margin_top=frame.margin_bottom=0
    paragraph=frame.paragraphs[0]
    paragraph.alignment=PP_ALIGN.CENTER
    end=paragraph._p.get_or_add_endParaRPr()
    size=round(min(28,max(16,height/914400*35))) if font_pt is None else font_pt
    end.set("sz",str(int(size*100)))
    end.set("lang","en-GB")
    ink=etree.SubElement(end,f"{{{A}}}solidFill")
    etree.SubElement(ink,f"{{{A}}}srgbClr").set("val","15263E")
    wrapper=etree.Element(f"{{{A14}}}m",nsmap={"a14":A14,"m":M})
    emit(element(element(wrapper,"oMathPara"),"oMath"),EQUATIONS[name])
    end.addprevious(wrapper)
    return box


def count_math(slide):
    return len(slide._element.xpath('.//*[local-name()="oMath"]'))


def repair(source:Path, destination:Path):
    if source.resolve()==destination.resolve():raise ValueError("Refuse to overwrite original")
    prs=Presentation(str(source))
    if len(prs.slides)!=44:raise ValueError(f"Expected 44-slide deck, got {len(prs.slides)}")
    slide=prs.slides[7]
    old={shape.name:shape for shape in slide.shapes if shape.name in EQUATIONS}
    if set(old)!=set(EQUATIONS) or any(shape.shape_type!=13 for shape in old.values()):
        raise ValueError("Seven recognized image-backed equations not found; refusing unsafe update")
    if count_math(slide)!=0:raise ValueError("Slide 8 contains pre-existing math; investigate before replacing")
    orig_text=Counter(shape.text for shape in slide.shapes if shape.has_text_frame)
    for name,old_picture in old.items():
        new=add_equation(slide,name,old_picture.left,old_picture.top,
                         old_picture.width,old_picture.height)
        parent=old_picture._element.getparent()
        parent.replace(old_picture._element,new._element)
    after_text=Counter(shape.text for shape in slide.shapes if shape.has_text_frame)
    if after_text!=orig_text+Counter({"":7}):
        raise AssertionError("Slide 8 editorial text unexpectedly changed")
    if count_math(slide)!=7:raise AssertionError("Seven OMML equations not present")
    for kind in ("f","nary","sSub","sSup","sSubSup","rad","acc"):
        if not slide._element.xpath('.//*[local-name()="'+kind+'"]'):
            raise AssertionError("Unstructured or missing OfficeMath "+kind)
    updated=etree.tostring(slide._element,encoding="UTF-8",
                            xml_declaration=True,standalone=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as zin,zipfile.ZipFile(destination,"w") as zout:
        for info in zin.infolist():
            zout.writestr(info,updated if info.filename==SLIDE_XML else zin.read(info.filename))
    with zipfile.ZipFile(source) as zin,zipfile.ZipFile(destination) as zout:
        assert zin.namelist()==zout.namelist()
        changed=[p for p in zin.namelist() if zin.read(p)!=zout.read(p)]
        if changed!=[SLIDE_XML]:raise AssertionError("Unexpected OOXML changes: "+str(changed))
    verified=Presentation(str(destination))
    assert len(verified.slides)==44
    assert count_math(verified.slides[7])==7
    assert not [s.name for s in verified.slides[7].shapes
                if s.name in EQUATIONS and s.shape_type==13]
    print("VERIFIED: seven editable OfficeMath equations, no equation pictures on slide 8")
    print("VERIFIED: only slide8.xml changed; all other PPTX ZIP members byte-identical")
    print("VERIFIED: preserved 44 slides, current theme, panels and explanatory text")
    return destination


if __name__=="__main__":
    if len(sys.argv)!=3:
        raise SystemExit("Usage: python -m scripts.repair_slide8_native_math INPUT.pptx OUTPUT.pptx")
    repair(Path(sys.argv[1]),Path(sys.argv[2]))
