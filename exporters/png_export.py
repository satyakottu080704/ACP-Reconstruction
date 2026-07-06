"""
png_export.py — render PlanModel to a PNG in the Acorn survey-plan style.

Style is derived from real Acorn deliverables (see exporters/acorn_style.py):
  - White background; clear rooms unfilled.
  - Double-line grey walls (light band edged with darker grey).
  - ACM rooms: pale pink fill + light red diagonal hatching.
  - No-access rooms: solid dark slate fill.
  - Bold two-line room labels ("004" / "BEDROOM").
  - Bold underlined floor headers; floors laid out side by side.
  - Sample annotations outside the plan with leader arrows
    (red when ACM-positive, black otherwise).
  - Doors: wall gap + grey quarter-circle swing arc.
  - Stairs: treads + direction arrow.  Loft hatches: box-with-X symbol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple

import numpy as np
import cv2

from reconstruction.plan_model import PlanModel, DoorSegment, StairPolygon
from . import acorn_style as st
from .acorn_style import hex_to_bgr

_FONT = cv2.FONT_HERSHEY_DUPLEX


def export_png(
    plan: PlanModel,
    out_path: Union[str, Path],
    canvas_px: int = 2000,
    padding: float = 0.04,
) -> Path:
    """
    Render the plan to a PNG (Acorn survey style).

    Args:
        plan: PlanModel with normalised [0,1] polygons.
        out_path: output file path.
        canvas_px: target canvas WIDTH in pixels (height follows layout).
        padding: outer margin as a fraction of the canvas width.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = st.layout_plan(plan)
    pad = int(canvas_px * padding)
    scale = (canvas_px - 2 * pad) / layout.world_w
    height_px = int(layout.world_h * scale) + 2 * pad

    img = np.full((height_px, canvas_px, 3), 255, dtype=np.uint8)

    def wp(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (pad + int(pt[0] * scale), pad + int(pt[1] * scale))

    wall_px = max(3, int(st.WALL_W * scale))

    for fi, block in layout.blocks.items():
        _draw_floor_header(img, block, wp, scale)
        rooms = [(i, r) for i, r in enumerate(plan.rooms) if r.floor_idx == fi]

        # 1. fills
        for _, room in rooms:
            if len(room.polygon) < 3:
                continue
            pts = np.array([wp(p) for p in block.tf_poly(room.polygon)], np.int32)
            if room.no_access or room.room_type == "no_access":
                cv2.fillPoly(img, [pts], hex_to_bgr(st.ROOM_NO_ACCESS_FILL))
            elif room.is_acm or room.room_type == "acm":
                cv2.fillPoly(img, [pts], hex_to_bgr(st.ROOM_ACM_FILL))
                _hatch(img, pts, hex_to_bgr(st.ACM_HATCH), spacing=max(10, int(0.012 * scale)))

        # 2. double-line walls (room outlines)
        for _, room in rooms:
            if len(room.polygon) < 3:
                continue
            pts = np.array([wp(p) for p in block.tf_poly(room.polygon)], np.int32)
            cv2.polylines(img, [pts], True, hex_to_bgr(st.WALL_DARK), wall_px, cv2.LINE_AA)
            inner = max(1, wall_px - 2)
            cv2.polylines(img, [pts], True, hex_to_bgr(st.WALL_LIGHT), inner, cv2.LINE_AA)

        # 3. explicit wall segments (from wall-mask reconstruction),
        # drawn only on the floor block that contains them
        for wall in plan.walls:
            if len(wall.points) < 2:
                continue
            mx = sum(p[0] for p in wall.points) / len(wall.points)
            my = sum(p[1] for p in wall.points) / len(wall.points)
            bx0, by0, bx1, by1 = block.src_bbox
            if not (bx0 - 1e-6 <= mx <= bx1 + 1e-6 and by0 - 1e-6 <= my <= by1 + 1e-6):
                continue
            pts = np.array([wp(p) for p in block.tf_poly(wall.points)], np.int32)
            closed = len(pts) > 2
            cv2.polylines(img, [pts], closed, hex_to_bgr(st.WALL_DARK), wall_px, cv2.LINE_AA)
            cv2.polylines(img, [pts], closed, hex_to_bgr(st.WALL_LIGHT),
                          max(1, wall_px - 2), cv2.LINE_AA)

        # 4. doors (only those inside this floor block)
        for door in plan.doors:
            bx0, by0, bx1, by1 = block.src_bbox
            if (bx0 - 1e-6 <= door.center[0] <= bx1 + 1e-6
                    and by0 - 1e-6 <= door.center[1] <= by1 + 1e-6):
                _draw_door(img, door, block, wp, scale)

        # 5. stairs
        for stair in plan.stairs:
            if getattr(stair, "floor_idx", 0) == fi:
                _draw_stairs(img, stair, block, wp)

        # 6. loft hatch symbols
        for hx, hy in plan.hatch_symbols:
            bx0, by0, bx1, by1 = block.src_bbox
            if bx0 - 1e-6 <= hx <= bx1 + 1e-6 and by0 - 1e-6 <= hy <= by1 + 1e-6:
                _draw_hatch_symbol(img, wp(block.tf(hx, hy)), max(8, int(0.014 * scale)))

        # 7. room labels (bold, number over name)
        for _, room in rooms:
            lines = st.room_label_lines(room)
            if not lines:
                continue
            cx, cy = wp(block.tf(*st.room_label_pos(plan, room)))
            colour = (255, 255, 255) if (room.no_access or room.room_type == "no_access") \
                else hex_to_bgr(st.LABEL_COLOUR)
            _draw_text_block(img, lines, cx, cy, st.LABEL_FONT * scale, colour, bold=True)

    # 8. sample annotations + leader arrows
    for ann in layout.annotations:
        colour = hex_to_bgr(ann.colour)
        line_h = st.ANN_LINE_H * scale
        cx = pad + int(ann.x * scale)
        ty = pad + ann.y * scale
        for i, line in enumerate(ann.lines):
            _draw_centred_text(img, line, cx, int(ty + (i + 0.8) * line_h),
                               st.ANN_FONT * scale, colour, bold=True)
        cv2.arrowedLine(img, wp(ann.arrow_start), wp(ann.arrow_end), colour,
                        max(2, int(0.0022 * scale)), cv2.LINE_AA, tipLength=0.10)

    cv2.imwrite(str(out_path), img)
    return out_path


# ── drawing helpers ───────────────────────────────────────────────────────────

def _font_scale(px_height: float) -> float:
    """cv2 font scale for a target text height in pixels."""
    return max(px_height / 26.0, 0.35)


def _draw_floor_header(img, block, wp, scale):
    x, y = wp(block.header_pos)
    text = f"{block.label}:"
    fs = _font_scale(st.HEADER_FONT * scale)
    th = max(2, int(fs * 2))
    (tw, txh), _ = cv2.getTextSize(text, _FONT, fs, th)
    cv2.putText(img, text, (x, y + txh), _FONT, fs, hex_to_bgr(st.LABEL_COLOUR), th, cv2.LINE_AA)
    cv2.line(img, (x, y + txh + 6), (x + tw, y + txh + 6), hex_to_bgr(st.LABEL_COLOUR), th, cv2.LINE_AA)


def _draw_centred_text(img, text, cx, cy, px_h, colour, bold=False):
    fs = _font_scale(px_h)
    th = max(2 if bold else 1, int(fs * (2 if bold else 1)))
    (tw, txh), _ = cv2.getTextSize(text, _FONT, fs, th)
    x = int(cx - tw / 2)
    x = max(2, min(img.shape[1] - tw - 2, x))
    cy = max(txh + 2, min(img.shape[0] - 2, cy))
    cv2.putText(img, text, (x, cy), _FONT, fs, colour, th, cv2.LINE_AA)


def _draw_text_block(img, lines, cx, cy, px_h, colour, bold=False):
    line_h = px_h * 1.35
    y0 = cy - (len(lines) - 1) * line_h / 2
    for i, line in enumerate(lines):
        _draw_centred_text(img, line, cx, int(y0 + i * line_h), px_h, colour, bold)


def _hatch(img, pts, colour, spacing=14):
    """Light diagonal (45 deg) hatching clipped to a polygon."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    overlay = img.copy()
    length = int(np.hypot(x1 - x0, y1 - y0)) + spacing
    for off in range(-length, length, spacing):
        cv2.line(overlay, (x0 + off, y0), (x0 + off + length, y0 + length), colour, 1, cv2.LINE_AA)
    sel = mask == 255
    img[sel] = overlay[sel]


def _draw_door(img, door: DoorSegment, block, wp, scale):
    cx, cy = wp(block.tf(*door.center))
    r = max(10, int(door.width * block.scale * scale / 2))
    a0 = int(door.angle_deg)
    colour = hex_to_bgr(st.DOOR_COLOUR)
    # white gap in the wall under the swing
    cv2.circle(img, (cx, cy), max(3, r // 4), (255, 255, 255), -1)
    cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, colour, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy),
             (cx + int(r * np.cos(np.radians(a0))), cy + int(r * np.sin(np.radians(a0)))),
             colour, 2, cv2.LINE_AA)


def _draw_stairs(img, stair: StairPolygon, block, wp):
    if len(stair.polygon) < 3:
        return
    pts = np.array([wp(p) for p in block.tf_poly(stair.polygon)], np.int32)
    colour = hex_to_bgr(st.STAIR_COLOUR)
    cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)

    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    overlay = img.copy()
    # treads are perpendicular to the flight direction
    horizontal_flight = abs(np.cos(np.radians(stair.direction_deg))) > 0.7
    if horizontal_flight:
        tread = max(6, (x1 - x0) // 7)
        for tx in range(x0 + tread, x1, tread):
            cv2.line(overlay, (tx, y0), (tx, y1), colour, 1, cv2.LINE_AA)
    else:
        tread = max(6, (y1 - y0) // 7)
        for ty in range(y0 + tread, y1, tread):
            cv2.line(overlay, (x0, ty), (x1, ty), colour, 1, cv2.LINE_AA)
    sel = mask == 255
    img[sel] = overlay[sel]

    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    alen = max(14, (max(y1 - y0, x1 - x0)) // 3)
    ang = np.radians(stair.direction_deg)
    cv2.arrowedLine(img, (cx, cy),
                    (int(cx + alen * np.cos(ang)), int(cy + alen * np.sin(ang))),
                    colour, 2, cv2.LINE_AA, tipLength=0.3)


def _draw_hatch_symbol(img, centre, half):
    """Loft hatch / skylight: square with an X."""
    cx, cy = centre
    colour = hex_to_bgr(st.STAIR_COLOUR)
    p0, p1 = (cx - half, cy - half), (cx + half, cy + half)
    cv2.rectangle(img, p0, p1, colour, 2, cv2.LINE_AA)
    cv2.line(img, p0, p1, colour, 1, cv2.LINE_AA)
    cv2.line(img, (cx - half, cy + half), (cx + half, cy - half), colour, 1, cv2.LINE_AA)
