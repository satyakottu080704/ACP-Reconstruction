"""
vsdx_export.py -- Generate Acorn-style digital floor plan VSDX files.

Matches reference "AI Draft.vsdx" format:
  A4 portrait, inches, no masters, raw geometry shapes.
  Rooms: blue polygons, walls: dark lines, ACM: red annotations, legend at bottom.
"""
from __future__ import annotations
import math, zipfile
from datetime import datetime
from pathlib import Path

PAGE_W = 8.26771653543307
PAGE_H = 11.69291338582677
_MX, _MTB, _MBT = 0.80, 1.00, 1.80
_DW = PAGE_W - 2 * _MX
_DH = PAGE_H - _MTB - _MBT
_Y0 = PAGE_H - _MTB
_C_ROOM, _C_ACM, _C_WALL = "#3264c8", "#dc3232", "#202020"
_C_RED, _C_LGBDR, _C_WHITE, _C_BLACK = "#cc0000", "#b4b4b4", "#ffffff", "#000000"
_SID = 0

def _next_id():
    global _SID; _SID += 1; return _SID

def _reset_ids():
    global _SID; _SID = 0

def _n2v(nx, ny):
    return _MX + nx * _DW, _Y0 - ny * _DH

def _poly_n2v(poly):
    return [_n2v(float(p[0]), float(p[1])) for p in poly]

def _bbox(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def _f(v):
    s = "{:.6f}".format(v).rstrip("0").rstrip(".")
    return s or "0"

def _room_shape(pts_vis, label, room_num, room_type, is_acm):
    x0,y0,x1,y1 = _bbox(pts_vis)
    pw,ph = max(x1-x0,0.01), max(y1-y0,0.01)
    pcx,pcy = (x0+x1)/2, (y0+y1)/2
    local = [(px-x0, py-y0) for px,py in pts_vis]
    rows = ""
    for ix,(lx,ly) in enumerate(local):
        t = "MoveTo" if ix==0 else "LineTo"
        rows += f"<Row T='{t}' IX='{ix+1}'><Cell N='X' V='{_f(lx)}'/><Cell N='Y' V='{_f(ly)}'/></Row>"
    rows += f"<Row T='LineTo' IX='{len(local)+1}'><Cell N='X' V='{_f(local[0][0])}'/><Cell N='Y' V='{_f(local[0][1])}'/></Row>"
    if is_acm:
        type_str = "ACM Positive"
    elif room_type == "no_access":
        type_str = "No Access"
    elif room_type == "loft":
        type_str = "Loft"
    else:
        type_str = "NAD"

    name_str = label.strip() or room_type.replace("_"," ").title()
    text = f"{room_num}&#xA;{type_str}&#xA;{name_str}"
    fill = _C_ACM if is_acm else _C_ROOM
    sid = _next_id()
    return (
        f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
        f"<Cell N='PinX' V='{_f(pcx)}'/><Cell N='PinY' V='{_f(pcy)}'/>"
        f"<Cell N='Width' V='{_f(pw)}'/><Cell N='Height' V='{_f(ph)}'/>"
        f"<Cell N='LocPinX' V='{_f(pw/2)}' F='Width*0.5'/><Cell N='LocPinY' V='{_f(ph/2)}' F='Height*0.5'/>"
        f"<Cell N='Angle' V='0'/>"
        f"<Cell N='FillForegnd' V='{fill}'/><Cell N='FillBkgnd' V='{_C_WHITE}'/><Cell N='FillPattern' V='1'/>"
        f"<Cell N='LineColor' V='{_C_WALL}'/><Cell N='LineWeight' V='0.015'/>"
        f"<Section N='Character'><Row IX='0'><Cell N='Color' V='{_C_WHITE}'/><Cell N='Style' V='1'/><Cell N='Size' V='0.09' U='PT'/></Row></Section>"
        f"<Section N='Paragraph'><Row IX='0'><Cell N='HorzAlign' V='1'/></Row></Section>"
        f"<Section N='Geometry' IX='0'><Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/>{rows}</Section>"
        f"<Text><cp IX='0'/><pp IX='0'/>{text}</Text>"
        f"</Shape>"
    )

def _line_shape(bx,by,ex,ey,color=_C_WALL,weight=0.02):
    pcx,pcy = (bx+ex)/2,(by+ey)/2
    length = math.sqrt((ex-bx)**2+(ey-by)**2)
    if length < 1e-9: return ""
    angle = math.atan2(ey-by,ex-bx)
    sid = _next_id()
    return (
        f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
        f"<Cell N='PinX' V='{_f(pcx)}' F='(BeginX+EndX)/2'/>"
        f"<Cell N='PinY' V='{_f(pcy)}' F='(BeginY+EndY)/2'/>"
        f"<Cell N='Width' V='{_f(length)}' F='SQRT((EndX-BeginX)^2+(EndY-BeginY)^2)'/>"
        f"<Cell N='Height' V='0'/>"
        f"<Cell N='LocPinX' V='{_f(length/2)}' F='Width*0.5'/><Cell N='LocPinY' V='0' F='Height*0.5'/>"
        f"<Cell N='Angle' V='{_f(angle)}' F='ATAN2(EndY-BeginY,EndX-BeginX)'/>"
        f"<Cell N='BeginX' V='{_f(bx)}'/><Cell N='BeginY' V='{_f(by)}'/>"
        f"<Cell N='EndX' V='{_f(ex)}'/><Cell N='EndY' V='{_f(ey)}'/>"
        f"<Cell N='LineColor' V='{color}'/><Cell N='LineWeight' V='{_f(weight)}'/>"
        f"<Cell N='FillPattern' V='0'/>"
        f"<Section N='Geometry' IX='0'><Cell N='NoFill' V='1'/><Cell N='NoLine' V='0'/>"
        f"<Row T='MoveTo' IX='1'><Cell N='X' V='0' F='Width*0'/><Cell N='Y' V='0'/></Row>"
        f"<Row T='LineTo' IX='2'><Cell N='X' V='{_f(length)}' F='Width*1'/><Cell N='Y' V='0'/></Row>"
        f"</Section></Shape>"
    )

def _rect_shape(px,py,w,h,fill="",border=_C_LGBDR,text="",text_color=_C_BLACK,bold=False,font_size=0.10):
    sid = _next_id()
    fill_xml = f"<Cell N='FillForegnd' V='{fill}'/><Cell N='FillBkgnd' V='{_C_WHITE}'/><Cell N='FillPattern' V='1'/>" if fill else "<Cell N='FillPattern' V='0'/>"
    border_xml = f"<Cell N='LineColor' V='{border}'/><Cell N='LineWeight' V='0.01'/>" if border else "<Cell N='LinePattern' V='0'/>"
    style = "1" if bold else "0"
    text_xml = (
        f"<Section N='Character'><Row IX='0'><Cell N='Color' V='{text_color}'/><Cell N='Style' V='{style}'/><Cell N='Size' V='{_f(font_size)}' U='PT'/></Row></Section>"
        f"<Section N='Paragraph'><Row IX='0'><Cell N='HorzAlign' V='1'/></Row></Section>"
        f"<Text><cp IX='0'/><pp IX='0'/>{text}</Text>"
    ) if text else ""
    return (
        f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
        f"<Cell N='PinX' V='{_f(px)}'/><Cell N='PinY' V='{_f(py)}'/>"
        f"<Cell N='Width' V='{_f(w)}'/><Cell N='Height' V='{_f(h)}'/>"
        f"<Cell N='LocPinX' V='{_f(w/2)}' F='Width*0.5'/><Cell N='LocPinY' V='{_f(h/2)}' F='Height*0.5'/>"
        f"<Cell N='Angle' V='0'/>{fill_xml}{border_xml}"
        f"<Section N='Geometry' IX='0'><Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/>"
        f"<Row T='MoveTo' IX='1'><Cell N='X' V='0' F='Width*0'/><Cell N='Y' V='0' F='Height*0'/></Row>"
        f"<Row T='LineTo' IX='2'><Cell N='X' V='{_f(w)}' F='Width*1'/><Cell N='Y' V='0' F='Height*0'/></Row>"
        f"<Row T='LineTo' IX='3'><Cell N='X' V='{_f(w)}' F='Width*1'/><Cell N='Y' V='{_f(h)}' F='Height*1'/></Row>"
        f"<Row T='LineTo' IX='4'><Cell N='X' V='0' F='Width*0'/><Cell N='Y' V='{_f(h)}' F='Height*1'/></Row>"
        f"<Row T='LineTo' IX='5'><Cell N='X' V='0' F='Width*0'/><Cell N='Y' V='0' F='Height*0'/></Row>"
        f"</Section>{text_xml}</Shape>"
    )

def _acm_annotation(vx, vy, acm_num, label):
    lw, lh = 1.50, 0.28
    label_cx = min(vx+1.30+lw/2, PAGE_W-lw/2-0.20)
    shapes = _rect_shape(label_cx, vy, lw, lh, border="",
        text=f"{acm_num}&#xA;{label}", text_color=_C_BLACK, font_size=0.09)
    shapes += _line_shape(vx+0.15, vy, label_cx-lw/2, vy, color=_C_RED, weight=0.015)
    return shapes

def _stair_shape(pts_vis, direction="UP"):
    x0,y0,x1,y1 = _bbox(pts_vis)
    w,h = max(x1-x0,0.01), max(y1-y0,0.01)
    cx,cy = (x0+x1)/2,(y0+y1)/2
    shapes = _rect_shape(cx,cy,w,h,border=_C_WALL,fill="")
    n_treads = max(3,int(h/0.15))
    for i in range(1,n_treads):
        ty = y0+i*h/n_treads
        shapes += _line_shape(x0,ty,x1,ty,color=_C_WALL,weight=0.005)
    lw2 = min(w*0.6,0.50)
    shapes += _rect_shape(cx,cy+h*0.35,lw2,0.14,border="",text=direction,font_size=0.09)
    return shapes

def _door_shape(pts_vis):
    x0,y0,x1,y1 = _bbox(pts_vis)
    shapes  = _line_shape(x0,y0,x0,y1,color=_C_WALL,weight=0.015)
    shapes += _line_shape(x1,y0,x1,y1,color=_C_WALL,weight=0.015)
    shapes += _line_shape(x0,y0,x1,y1,color=_C_WALL,weight=0.005)
    return shapes

def _wall_outlines(rooms):
    edges = set(); shapes = ""
    for room in rooms:
        poly = room.polygon; n = len(poly)
        for i in range(n):
            p1 = _n2v(float(poly[i][0]),float(poly[i][1]))
            p2 = _n2v(float(poly[(i+1)%n][0]),float(poly[(i+1)%n][1]))
            key = (round(min(p1[0],p2[0]),4),round(min(p1[1],p2[1]),4),
                   round(max(p1[0],p2[0]),4),round(max(p1[1],p2[1]),4))
            if key not in edges:
                edges.add(key)
                s = _line_shape(p1[0],p1[1],p2[0],p2[1],color=_C_WALL)
                if s: shapes += s
    return shapes

def _legend():
    lx,ly = PAGE_W/2, 0.85
    bw,bh = 2.20, 1.45
    shapes = _rect_shape(lx,ly,bw,bh,border=_C_LGBDR,fill=_C_WHITE)
    shapes += _rect_shape(lx,ly+bh/2-0.14,bw-0.20,0.18,border="",text="Legend",bold=True,font_size=0.11)
    sw,sh = 0.22,0.16
    for i,(scolor,slabel) in enumerate([(_C_ACM,"ACM Positive"),(_C_ROOM,"No Access"),(_C_WHITE,"Clear / NAD")]):
        row_y = ly+bh/2-0.40-i*0.30
        sx = lx-bw/2+0.18+sw/2
        shapes += _rect_shape(sx,row_y,sw,sh,fill=scolor,border=_C_BLACK)
        shapes += _rect_shape(sx+sw/2+0.70,row_y,1.20,sh,border="",text=slabel,font_size=0.09)
    return shapes

def _title_bar(floor_label):
    py = PAGE_H-_MTB/2
    return _rect_shape(PAGE_W/2,py,PAGE_W-0.80,0.38,border=_C_LGBDR,fill="",
        text=f"Floor Plans: {floor_label}",bold=True,font_size=0.13)

def _build_page(rooms, doors, stairs, floor_label, acm_counter):
    s = _title_bar(floor_label)
    for i,room in enumerate(rooms):
        poly = getattr(room,"polygon",[])
        if len(poly) < 3: continue
        s += _room_shape(_poly_n2v(poly),room.label,
                         getattr(room,"number","") or f"{i+1:03d}",
                         room.room_type,room.is_acm)
    s += _wall_outlines(rooms)
    for door in doors:
        poly = getattr(door,"polygon",[])
        if len(poly) >= 3: s += _door_shape(_poly_n2v(poly))
    for stair in stairs:
        poly = getattr(stair,"polygon",[])
        if len(poly) >= 3: s += _stair_shape(_poly_n2v(poly))
    for room in rooms:
        if room.is_acm:
            cx,cy = room.centroid()
            vx,vy = _n2v(cx,cy)
            acm_counter[0] += 1
            s += _acm_annotation(vx,vy,f"S{acm_counter[0]:03d}",room.label.strip() or "ACM")
    s += _legend()
    return (
        "<?xml version='1.0' encoding='utf-8' ?>"
        "<PageContents xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
        "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' "
        "xml:space='preserve'>"
        f"<Shapes>{s}</Shapes></PageContents>"
    )

_CT = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
    "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
    "<Default Extension='xml' ContentType='application/xml'/>"
    "<Override PartName='/visio/document.xml' ContentType='application/vnd.ms-visio.drawing.main+xml'/>"
    "<Override PartName='/visio/pages/pages.xml' ContentType='application/vnd.ms-visio.pages+xml'/>"
    "<Override PartName='/visio/pages/page1.xml' ContentType='application/vnd.ms-visio.page+xml'/>"
    "<Override PartName='/visio/windows.xml' ContentType='application/vnd.ms-visio.windows+xml'/>"
    "<Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>"
    "<Override PartName='/docProps/app.xml' ContentType='application/vnd.openxmlformats-officedocument.extended-properties+xml'/>"
    "</Types>"
)
_RELS_ROOT = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
    "<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/document' Target='visio/document.xml'/>"
    "<Relationship Id='rId3' Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties' Target='docProps/core.xml'/>"
    "<Relationship Id='rId4' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties' Target='docProps/app.xml'/>"
    "</Relationships>"
)
_DOC_RELS = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
    "<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/pages' Target='pages/pages.xml'/>"
    "<Relationship Id='rId2' Type='http://schemas.microsoft.com/visio/2010/relationships/windows' Target='windows.xml'/>"
    "</Relationships>"
)
_PAGES_RELS = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
    "<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/page' Target='page1.xml'/>"
    "</Relationships>"
)
_WINDOWS = (
    f"<?xml version='1.0' encoding='utf-8' ?>"
    f"<Windows ClientWidth='1920' ClientHeight='828' "
    f"xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
    f"xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' "
    f"xml:space='preserve'>"
    f"<Window ID='0' WindowType='Drawing' WindowState='1073741824' "
    f"WindowLeft='0' WindowTop='0' WindowWidth='1920' WindowHeight='900' "
    f"ContainerType='Page' Page='0' ViewScale='-1' "
    f"ViewCenterX='{PAGE_W/2:.4f}' ViewCenterY='{PAGE_H/2:.4f}'>"
    f"<ShowRulers>1</ShowRulers><ShowGrid>1</ShowGrid>"
    f"<ShowPageBreaks>1</ShowPageBreaks><ShowGuides>1</ShowGuides>"
    f"<ShowConnectionPoints>1</ShowConnectionPoints>"
    f"</Window></Windows>"
)

def _pages_xml(page_name):
    return (
        "<?xml version='1.0' encoding='utf-8' ?>"
        "<Pages xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
        "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' "
        "xml:space='preserve'>"
        f"<Page ID='0' NameU='{page_name}' IsCustomNameU='1' "
        f"Name='{page_name}' IsCustomName='1' "
        f"ViewScale='-1' ViewCenterX='{PAGE_W/2:.6f}' ViewCenterY='{PAGE_H/2:.6f}'>"
        "<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
        f"<Cell N='PageWidth' V='{PAGE_W}'/>"
        f"<Cell N='PageHeight' V='{PAGE_H}'/>"
        "<Cell N='ShdwOffsetX' V='0.1181102362204724'/>"
        "<Cell N='ShdwOffsetY' V='-0.1181102362204724'/>"
        "<Cell N='PageScale' V='0.03937007874015748' U='MM'/>"
        "<Cell N='DrawingScale' V='0.03937007874015748' U='MM'/>"
        "<Cell N='DrawingSizeType' V='0'/>"
        "<Cell N='DrawingScaleType' V='0'/>"
        "<Cell N='DrawingResizeType' V='1'/>"
        "<Cell N='PrintPageOrientation' V='1'/>"
        "</PageSheet><Rel r:id='rId1'/></Page></Pages>"
    )

def _document_xml():
    return (
        "<?xml version='1.0' encoding='utf-8' ?>"
        "<VisioDocument xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
        "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' "
        "xml:space='preserve'>"
        "<DocumentSettings TopPage='0' DefaultTextStyle='3' DefaultLineStyle='3' "
        "DefaultFillStyle='3' DefaultGuideStyle='4'>"
        "<GlueSettings>9</GlueSettings><SnapSettings>65847</SnapSettings>"
        "<SnapExtensions>34</SnapExtensions><SnapAngles/>"
        "<DynamicGridEnabled>1</DynamicGridEnabled></DocumentSettings>"
        "<Colors>"
        "<ColorEntry IX='27' RGB='#3264C8'/>"
        "<ColorEntry IX='28' RGB='#202020'/>"
        "<ColorEntry IX='29' RGB='#CC0000'/>"
        "<ColorEntry IX='30' RGB='#DC3232'/>"
        "<ColorEntry IX='31' RGB='#B4B4B4'/>"
        "</Colors>"
        "<StyleSheets>"
        "<StyleSheet ID='0' NameU='No Style' Name='No Style'>"
        "<Cell N='LineWeight' V='0.01041666666666667'/>"
        "<Cell N='LineColor' V='0'/><Cell N='LinePattern' V='1'/>"
        "<Cell N='FillForegnd' V='1'/><Cell N='FillBkgnd' V='0'/><Cell N='FillPattern' V='1'/>"
        "</StyleSheet>"
        "<StyleSheet ID='3' NameU='Normal' Name='Normal'>"
        "<Cell N='LineWeight' V='0.01041666666666667'/>"
        "<Cell N='LineColor' V='0'/><Cell N='LinePattern' V='1'/>"
        "<Cell N='FillForegnd' V='1'/><Cell N='FillBkgnd' V='0'/><Cell N='FillPattern' V='1'/>"
        "</StyleSheet></StyleSheets></VisioDocument>"
    )

def _app_xml():
    return (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Properties xmlns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'>"
        "<Application>Microsoft Visio</Application><AppVersion>16.0000</AppVersion>"
        "</Properties>"
    )

def _core_xml(title="Floor Plan"):
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<cp:coreProperties "
        "xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' "
        "xmlns:dc='http://purl.org/dc/elements/1.1/' "
        "xmlns:dcterms='http://purl.org/dc/terms/' "
        "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>"
        f"<dc:title>{title}</dc:title>"
        "<dc:creator>Acorn Survey Platform</dc:creator>"
        f"<dcterms:created xsi:type='dcterms:W3CDTF'>{now}</dcterms:created>"
        f"<dcterms:modified xsi:type='dcterms:W3CDTF'>{now}</dcterms:modified>"
        "</cp:coreProperties>"
    )

def _write_vsdx(rooms, doors, stairs, out_path, page_name, floor_label):
    _reset_ids()
    acm_counter = [0]
    page1 = _build_page(rooms, doors, stairs, floor_label, acm_counter)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(out_path), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS_ROOT)
        z.writestr("visio/document.xml", _document_xml())
        z.writestr("visio/_rels/document.xml.rels", _DOC_RELS)
        z.writestr("visio/pages/pages.xml", _pages_xml(page_name))
        z.writestr("visio/pages/_rels/pages.xml.rels", _PAGES_RELS)
        z.writestr("visio/pages/page1.xml", page1)
        z.writestr("visio/windows.xml", _WINDOWS)
        z.writestr("docProps/app.xml", _app_xml())
        z.writestr("docProps/core.xml", _core_xml(floor_label))

def export_vsdx(plan, out_path, stem=None):
    """
    Export PlanModel to VSDX matching Acorn AI Draft reference format.

    If plan.has_loft:
        writes <stem>_ground.vsdx  and  <stem>_loft.vsdx
    else:
        writes <stem>_ground.vsdx

    Returns: Path to the primary (ground) .vsdx file.  When a loft exists a
    second <stem>_loft.vsdx is written alongside it; both are mirrored to
    OUTPUT_FOLDER/visio.
    """
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = out_path.stem
        for sfx in ("_ground","_loft"):
            stem = stem.replace(sfx,"")

    ground_rooms = [r for r in plan.rooms if not r.is_loft]
    loft_rooms   = [r for r in plan.rooms if r.is_loft]
    all_doors    = list(plan.doors)
    all_stairs   = list(plan.stairs)
    floor_lbl    = (plan.floor_labels[0] if plan.floor_labels else "Ground Floor")

    produced = {}

    gp = out_dir / f"{stem}_ground.vsdx"
    _write_vsdx(ground_rooms, all_doors, all_stairs, gp, "Floor Plans", floor_lbl)
    produced["ground"] = gp
    print(f"[vsdx] ground -> {gp}")

    if plan.has_loft and loft_rooms:
        lp = out_dir / f"{stem}_loft.vsdx"
        _write_vsdx(loft_rooms, [], [], lp, "Loft", "Loft")
        produced["loft"] = lp
        print(f"[vsdx] loft   -> {lp}")

    try:
        import sys
        _root = Path(__file__).resolve().parents[1]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        import config as _cfg, shutil
        mirror_dir = Path(_cfg.OUTPUT_FOLDER) / "visio"
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for p in produced.values():
            try: shutil.copy2(str(p), str(mirror_dir / p.name))
            except Exception: pass
    except Exception:
        pass

    # Exporter contract: return a single Path (export_all() and callers
    # treat the return value as the written file)
    return produced["ground"]
