#!/usr/bin/env python3
"""Read-only image/media inventory of EVERY slide in the published main PPTX."""
import io, json, hashlib, zipfile, sys
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE as T
from pptx.oxml.ns import qn
from PIL import Image,ImageDraw,ImageFont

OUTPUT=Path("presentations/full_deck_formula_audit")
OUTPUT.mkdir(parents=True,exist_ok=True)
def kind(sh):
    try: return sh.shape_type
    except NotImplementedError:return None
def rec_shapes(shapes):
    for s in shapes:
        if kind(s)==T.GROUP:yield from rec_shapes(s.shapes)
        else:yield s
def title(slide):
    return next((x.text[:130] for x in slide.shapes if x.has_text_frame and x.text.strip()),"")
def extract(path):
    prs=Presentation(str(path))
    output=[]
    thumbs=[]
    for n,slide in enumerate(prs.slides,1):
        rec={"slide":n,"title":title(slide),"native_omml":len(slide._element.xpath('.//*[local-name()="oMath"]')),"assets":[]}
        for k,sh in enumerate(rec_shapes(slide.shapes)):
            refs=sh._element.xpath('.//*[local-name()="blip"]')
            for j,ref in enumerate(refs):
                rid=ref.get(qn("r:embed"))
                if not rid:continue
                blob=slide.part.related_part(rid).blob
                try:
                    img=Image.open(io.BytesIO(blob)).convert("RGBA")
                except Exception as exc:
                    print("BAD_IMAGE",n,k,exc);continue
                digest=hashlib.sha256(blob).hexdigest()
                image_name=f"s{n:02d}_object{k:02d}_{j}_{digest[:9]}.png"
                img.save(OUTPUT/image_name)
                rect=[round(v/914400,2) for v in (sh.left,sh.top,sh.width,sh.height)]
                rec["assets"].append({"file":image_name,"sha256":digest,"shape_name":sh.name,"kind":str(kind(sh)),"bounds_inches":rect,"size_pixels":img.size})
                thumbs.append((n,image_name,rec["title"],img))
        output.append(rec)
    (OUTPUT/"manifest.json").write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
    # Overview contact sheets sized to allow reliable human transcription of embedded math
    W=1600; rowh=275; cellw=800
    for start in range(0,len(thumbs),12):
        subset=thumbs[start:start+12]
        canvas=Image.new("RGB",(W,rowh*((len(subset)+1)//2)),(242,246,250))
        draw=ImageDraw.Draw(canvas)
        for i,(n,name,t,img) in enumerate(subset):
            x=(i%2)*cellw;y=(i//2)*rowh
            draw.rectangle((x+3,y+3,x+cellw-5,y+rowh-3),outline="#708090",width=2)
            draw.text((x+10,y+11),f"Slide {n} | {name}",fill="#121824")
            draw.text((x+10,y+31),t[:82],fill="#40546a")
            # Composite transparent image over white; avoid wrong alpha interpretation
            white=Image.new("RGBA",img.size,"white")
            white.alpha_composite(img)
            preview=white.convert("RGB")
            preview.thumbnail((cellw-30,rowh-72))
            canvas.paste(preview,(x+int((cellw-preview.width)/2),y+62))
        canvas.save(OUTPUT/f"overview_{start//12+1:02d}.png")
    for r in output:
        print(f"SLIDE {r['slide']:2} | native={r['native_omml']:2} | visual_assets={len(r['assets'])} | {r['title'][:65]}")
        for m in r["assets"]:
            print("   ",m["file"],m["bounds_inches"],m["size_pixels"])
    print("AUDITED ALL",len(prs.slides),"SLIDES; MEDIA",len(thumbs))

if __name__=="__main__":extract(sys.argv[1])
