"""
pdf_export.py — export PlanModel to PDF.

Strategy:
  1. Preferred: render the Acorn-styled PNG (png_export) and embed it via
     Pillow, so PDF output is identical to the PNG/SVG deliverables.
  2. Fallback (no Pillow): draw vector paths directly with reportlab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from reconstruction.plan_model import PlanModel

_PAGE_W = 595.0   # A4 width  in points (72 dpi)
_PAGE_H = 842.0   # A4 height in points

try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


def export_pdf(plan: PlanModel, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer the Acorn-styled raster path so PDF matches PNG/SVG exactly
    try:
        return _export_via_png(plan, out_path)
    except Exception:
        if _REPORTLAB:
            return _export_reportlab(plan, out_path)
        raise


# ─── reportlab path ───────────────────────────────────────────────────────────

def _export_reportlab(plan: PlanModel, out_path: Path) -> Path:
    pad = 40.0
    draw_w = _PAGE_W - 2 * pad
    draw_h = _PAGE_H - 2 * pad

    def to_pt(x: float, y: float):
        return (pad + x * draw_w, _PAGE_H - pad - y * draw_h)

    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle("Acorn Floor Plan")

    # Background
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, _PAGE_W, _PAGE_H, fill=1, stroke=0)

    for room in plan.rooms:
        if len(room.polygon) < 3:
            continue
        pts = [to_pt(x, y) for x, y in room.polygon]

        if room.is_acm or room.room_type == "acm":
            c.setFillColorRGB(0.86, 0.31, 0.31, alpha=0.7)
        elif room.is_loft:
            c.setFillColorRGB(0.94, 0.90, 0.78)
        else:
            c.setFillColorRGB(0.92, 0.96, 1.0)

        c.setStrokeColorRGB(0.24, 0.24, 0.24)
        c.setLineWidth(1.0)

        p = c.beginPath()
        p.moveTo(*pts[0])
        for px, py in pts[1:]:
            p.lineTo(px, py)
        p.close()
        c.drawPath(p, fill=1, stroke=1)

        # ACM hatching
        if room.is_acm or room.room_type == "acm":
            c.saveState()
            c.clipPath(p, stroke=0)
            c.setStrokeColorRGB(0.55, 0.05, 0.05)
            c.setLineWidth(0.5)
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            spacing = 10
            for offset in range(int(x_min - (y_max - y_min)), int(x_max + spacing), spacing):
                c.line(offset, y_min, offset + (y_max - y_min), y_max)
            c.restoreState()

        # Label
        label = room.label or room.ocr_text or ""
        if label:
            cx, cy = room.centroid()
            lx, ly = to_pt(cx, cy)
            c.setFillColorRGB(0.08, 0.08, 0.08)
            c.setFont("Helvetica", 8)
            c.drawCentredString(lx, ly, label[:40])

    # Walls
    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    c.setLineWidth(2.0)
    for wall in plan.walls:
        if len(wall.points) < 2:
            continue
        pts = [to_pt(x, y) for x, y in wall.points]
        p = c.beginPath()
        p.moveTo(*pts[0])
        for px, py in pts[1:]:
            p.lineTo(px, py)
        c.drawPath(p, fill=0, stroke=1)

    # Stairs
    c.setFillColorRGB(0.94, 0.94, 0.94)
    c.setStrokeColorRGB(0.31, 0.31, 0.31)
    c.setLineWidth(1.0)
    for stair in plan.stairs:
        if len(stair.polygon) < 3:
            continue
        pts = [to_pt(x, y) for x, y in stair.polygon]
        p = c.beginPath()
        p.moveTo(*pts[0])
        for px, py in pts[1:]:
            p.lineTo(px, py)
        p.close()
        c.drawPath(p, fill=1, stroke=1)

    c.save()
    return out_path


# ─── PNG fallback path ────────────────────────────────────────────────────────

def _export_via_png(plan: PlanModel, out_path: Path) -> Path:
    try:
        from PIL import Image
        from exporters.png_export import export_png
    except ImportError:
        raise RuntimeError(
            "Neither reportlab nor Pillow is available. "
            "Install reportlab for PDF export: pip install reportlab"
        )

    png_path = out_path.with_suffix(".tmp.png")
    export_png(plan, png_path)
    img = Image.open(str(png_path))
    img.convert("RGB").save(str(out_path), "PDF", resolution=150)
    png_path.unlink(missing_ok=True)
    return out_path
