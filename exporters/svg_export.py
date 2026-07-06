"""
svg_export.py — export PlanModel to SVG in the Acorn survey-plan style.

Style is derived from real Acorn deliverables (see exporters/acorn_style.py):
  - White background; clear rooms unfilled.
  - Double-line grey walls (light band edged with darker grey).
  - ACM rooms: pale pink fill + light red diagonal <pattern> hatching.
  - No-access rooms: solid dark slate fill.
  - Bold two-line room labels ("004" / "BEDROOM").
  - Bold underlined floor headers; floors laid out side by side.
  - Sample annotations outside the plan with leader arrows
    (red when ACM-positive, black otherwise).
  - Doors: wall gap + grey quarter-circle swing arc.
  - Stairs: treads + direction arrow.  Loft hatches: box-with-X symbol.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Union

from reconstruction.plan_model import PlanModel, DoorSegment, StairPolygon
from . import acorn_style as st

_VIEW_W = 1400  # SVG user units for world width


def export_svg(plan: PlanModel, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = st.layout_plan(plan)
    s = _VIEW_W / layout.world_w                # world -> svg scale
    W, H = _VIEW_W, layout.world_h * s
    pad = 0.03 * W

    def wp(pt: Tuple[float, float]) -> Tuple[float, float]:
        return (pad + pt[0] * s, pad + pt[1] * s)

    wall_px = max(2.5, st.WALL_W * s)
    out: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W + 2*pad:.0f} {H + 2*pad:.0f}" '
        f'width="{W + 2*pad:.0f}" height="{H + 2*pad:.0f}" '
        f'font-family="Arial, Helvetica, sans-serif">',
        "<defs>",
        f'<pattern id="acm_hatch" patternUnits="userSpaceOnUse" width="12" height="12" '
        f'patternTransform="rotate(45)">'
        f'<rect width="12" height="12" fill="{st.ROOM_ACM_FILL}"/>'
        f'<line x1="0" y1="0" x2="0" y2="12" stroke="{st.ACM_HATCH}" stroke-width="1.2"/>'
        f"</pattern>",
        _arrow_marker("arrow_black", st.ANN_BLACK),
        _arrow_marker("arrow_red", st.ANN_RED),
        _arrow_marker("arrow_grey", st.STAIR_COLOUR),
        "</defs>",
        f'<rect x="0" y="0" width="{W + 2*pad:.0f}" height="{H + 2*pad:.0f}" fill="{st.BG}"/>',
    ]

    for fi, block in layout.blocks.items():
        out.append(_floor_header(block, wp, s))
        rooms = [(i, r) for i, r in enumerate(plan.rooms) if r.floor_idx == fi]

        # 1. fills
        for _, room in rooms:
            if len(room.polygon) < 3:
                continue
            pts = _pts([wp(p) for p in block.tf_poly(room.polygon)])
            if room.no_access or room.room_type == "no_access":
                out.append(f'<polygon points="{pts}" fill="{st.ROOM_NO_ACCESS_FILL}"/>')
            elif room.is_acm or room.room_type == "acm":
                out.append(f'<polygon points="{pts}" fill="url(#acm_hatch)"/>')

        # 2. double-line walls (outer dark stroke + lighter inner stroke)
        for _, room in rooms:
            if len(room.polygon) < 3:
                continue
            pts = _pts([wp(p) for p in block.tf_poly(room.polygon)])
            out.append(
                f'<polygon points="{pts}" fill="none" stroke="{st.WALL_DARK}" '
                f'stroke-width="{wall_px:.2f}" stroke-linejoin="miter"/>'
                f'<polygon points="{pts}" fill="none" stroke="{st.WALL_LIGHT}" '
                f'stroke-width="{max(wall_px - 2, 0.8):.2f}" stroke-linejoin="miter"/>'
            )

        # 3. explicit wall segments, drawn only on the containing floor block
        for wall in plan.walls:
            if len(wall.points) < 2:
                continue
            mx = sum(p[0] for p in wall.points) / len(wall.points)
            my = sum(p[1] for p in wall.points) / len(wall.points)
            bx0, by0, bx1, by1 = block.src_bbox
            if not (bx0 - 1e-6 <= mx <= bx1 + 1e-6 and by0 - 1e-6 <= my <= by1 + 1e-6):
                continue
            pw = [wp(p) for p in block.tf_poly(wall.points)]
            coords = " L".join(f"{x:.1f} {y:.1f}" for x, y in pw)
            close = " Z" if len(pw) > 2 else ""
            out.append(
                f'<path d="M{coords}{close}" fill="none" stroke="{st.WALL_DARK}" '
                f'stroke-width="{wall_px:.2f}"/>'
                f'<path d="M{coords}{close}" fill="none" stroke="{st.WALL_LIGHT}" '
                f'stroke-width="{max(wall_px - 2, 0.8):.2f}"/>'
            )

        # 4. doors (only those inside this floor block)
        for door in plan.doors:
            bx0, by0, bx1, by1 = block.src_bbox
            if (bx0 - 1e-6 <= door.center[0] <= bx1 + 1e-6
                    and by0 - 1e-6 <= door.center[1] <= by1 + 1e-6):
                out.append(_svg_door(door, block, wp, s))

        # 5. stairs
        for stair in plan.stairs:
            if getattr(stair, "floor_idx", 0) == fi:
                out.append(_svg_stairs(stair, block, wp))

        # 6. loft hatch symbols (only on the floor block that contains them)
        for hx, hy in plan.hatch_symbols:
            bx0, by0, bx1, by1 = block.src_bbox
            if bx0 - 1e-6 <= hx <= bx1 + 1e-6 and by0 - 1e-6 <= hy <= by1 + 1e-6:
                out.append(_svg_hatch_symbol(wp(block.tf(hx, hy)), max(7.0, 0.012 * s)))

        # 7. room labels
        for _, room in rooms:
            lines = st.room_label_lines(room)
            if not lines:
                continue
            cx, cy = wp(block.tf(*st.room_label_pos(plan, room)))
            colour = "#FFFFFF" if (room.no_access or room.room_type == "no_access") \
                else st.LABEL_COLOUR
            out.append(_text_block(lines, cx, cy, st.LABEL_FONT * s, colour))

    # 8. sample annotations
    for ann in layout.annotations:
        fs = st.ANN_FONT * s
        line_h = st.ANN_LINE_H * s
        ax, ay = wp((ann.x, ann.y))
        parts = []
        for i, line in enumerate(ann.lines):
            parts.append(
                f'<text x="{ax:.1f}" y="{ay + (i + 0.8) * line_h:.1f}" '
                f'text-anchor="middle" font-size="{fs:.1f}" font-weight="bold" '
                f'fill="{ann.colour}">{_esc(line)}</text>'
            )
        (sx, sy), (ex, ey) = wp(ann.arrow_start), wp(ann.arrow_end)
        marker = "arrow_red" if ann.colour == st.ANN_RED else "arrow_black"
        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{ann.colour}" stroke-width="2" marker-end="url(#{marker})"/>'
        )
        out.append("".join(parts))

    out.append("</svg>")
    out_path.write_text("\n".join(out), encoding="utf-8")
    return out_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _pts(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _arrow_marker(mid: str, colour: str) -> str:
    return (
        f'<marker id="{mid}" markerWidth="10" markerHeight="8" refX="9" refY="4" '
        f'orient="auto" markerUnits="userSpaceOnUse">'
        f'<path d="M0 0 L10 4 L0 8 Z" fill="{colour}"/></marker>'
    )


def _floor_header(block, wp, s) -> str:
    x, y = wp(block.header_pos)
    fs = st.HEADER_FONT * s
    return (
        f'<text x="{x:.1f}" y="{y + fs:.1f}" font-size="{fs:.1f}" font-weight="bold" '
        f'text-decoration="underline" fill="{st.LABEL_COLOUR}">{_esc(block.label)}:</text>'
    )


def _text_block(lines: List[str], cx: float, cy: float, fs: float, colour: str) -> str:
    line_h = fs * 1.35
    y0 = cy - (len(lines) - 1) * line_h / 2 + fs * 0.35
    return "".join(
        f'<text x="{cx:.1f}" y="{y0 + i * line_h:.1f}" text-anchor="middle" '
        f'font-size="{fs:.1f}" font-weight="bold" fill="{colour}">{_esc(line)}</text>'
        for i, line in enumerate(lines)
    )


def _svg_door(door: DoorSegment, block, wp, s) -> str:
    cx, cy = wp(block.tf(*door.center))
    r = max(8.0, door.width * block.scale * s / 2)
    a0, a1 = door.angle_deg, door.angle_deg + 90
    sx = cx + r * math.cos(math.radians(a0))
    sy = cy + r * math.sin(math.radians(a0))
    ex = cx + r * math.cos(math.radians(a1))
    ey = cy + r * math.sin(math.radians(a1))
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{max(2.0, r/4):.1f}" fill="{st.BG}"/>'
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}" '
        f'stroke="{st.DOOR_COLOUR}" stroke-width="1.5"/>'
        f'<path d="M{sx:.1f} {sy:.1f} A{r:.1f} {r:.1f} 0 0 1 {ex:.1f} {ey:.1f}" '
        f'fill="none" stroke="{st.DOOR_COLOUR}" stroke-width="1.5"/>'
    )


def _svg_stairs(stair: StairPolygon, block, wp) -> str:
    if len(stair.polygon) < 3:
        return ""
    pw = [wp(p) for p in block.tf_poly(stair.polygon)]
    xs, ys = [p[0] for p in pw], [p[1] for p in pw]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    parts = [
        f'<polygon points="{_pts(pw)}" fill="none" stroke="{st.STAIR_COLOUR}" stroke-width="1.5"/>'
    ]
    # treads are perpendicular to the flight direction
    if abs(math.cos(math.radians(stair.direction_deg))) > 0.7:
        tread = max(6.0, (x1 - x0) / 7)
        tx = x0 + tread
        while tx < x1:
            parts.append(
                f'<line x1="{tx:.1f}" y1="{y0:.1f}" x2="{tx:.1f}" y2="{y1:.1f}" '
                f'stroke="{st.STAIR_COLOUR}" stroke-width="1"/>'
            )
            tx += tread
    else:
        tread = max(6.0, (y1 - y0) / 7)
        ty = y0 + tread
        while ty < y1:
            parts.append(
                f'<line x1="{x0:.1f}" y1="{ty:.1f}" x2="{x1:.1f}" y2="{ty:.1f}" '
                f'stroke="{st.STAIR_COLOUR}" stroke-width="1"/>'
            )
            ty += tread
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    alen = max(14.0, max(y1 - y0, x1 - x0) / 3)
    ang = math.radians(stair.direction_deg)
    parts.append(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" '
        f'x2="{cx + alen * math.cos(ang):.1f}" y2="{cy + alen * math.sin(ang):.1f}" '
        f'stroke="{st.STAIR_COLOUR}" stroke-width="2" marker-end="url(#arrow_grey)"/>'
    )
    return "".join(parts)


def _svg_hatch_symbol(centre, half: float) -> str:
    cx, cy = centre
    c = st.STAIR_COLOUR
    return (
        f'<rect x="{cx - half:.1f}" y="{cy - half:.1f}" width="{2*half:.1f}" '
        f'height="{2*half:.1f}" fill="none" stroke="{c}" stroke-width="1.5"/>'
        f'<line x1="{cx - half:.1f}" y1="{cy - half:.1f}" x2="{cx + half:.1f}" '
        f'y2="{cy + half:.1f}" stroke="{c}" stroke-width="1"/>'
        f'<line x1="{cx - half:.1f}" y1="{cy + half:.1f}" x2="{cx + half:.1f}" '
        f'y2="{cy - half:.1f}" stroke="{c}" stroke-width="1"/>'
    )
