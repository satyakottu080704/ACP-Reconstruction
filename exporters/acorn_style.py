"""
acorn_style.py — shared style constants + layout engine for Acorn survey-plan
rendering (matches the hand-finished digital plans produced by Acorn
Analytical Services, e.g. "N-70078 plan.jpg").

Conventions extracted from real deliverables:
  - White background; clear rooms have NO fill (white).
  - Walls: double-line grey (light-grey band edged by darker grey lines).
  - ACM rooms: pale pink/red fill + light red diagonal hatching.
  - No-access rooms: solid dark slate fill.
  - Room labels: bold, two centred lines — "004" / "BEDROOM".
  - Floor headers: bold underlined "Ground Floor:" at the top-left of each
    floor block; multiple floors laid out side by side.
  - Sample annotations: bold text OUTSIDE the plan with a straight leader
    arrow pointing at the room wall.  "Ref S001 ..." for cross-references.
    RED text/arrow when ACM-positive, black otherwise.
  - Doors: wall gap + thin grey quarter-circle swing arc.
  - Loft hatches / skylights: small square with an X.

Pure geometry/text module — no OpenCV / SVG specifics — consumed by both
png_export and svg_export so the two stay pixel-consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from reconstruction.plan_model import PlanModel, RoomPolygon, SampleAnnotation

# ── palette (hex) ─────────────────────────────────────────────────────────────
BG = "#FFFFFF"
WALL_DARK = "#8C8C8C"        # outer edges of the double wall line
WALL_LIGHT = "#D2D2D2"       # infill band of the wall
ROOM_CLEAR_FILL = "#FFFFFF"
ROOM_ACM_FILL = "#F2DADA"    # pale pink ACM wash
ACM_HATCH = "#D89696"        # light red diagonal hatch
ROOM_NO_ACCESS_FILL = "#5A7086"  # solid dark slate
DOOR_COLOUR = "#8C8C8C"
STAIR_COLOUR = "#787878"
LABEL_COLOUR = "#1A1A1A"
ANN_BLACK = "#1A1A1A"
ANN_RED = "#C00000"

# ── text metrics (world units; world height of one floor block ~= 1.0) ───────
LABEL_FONT = 0.022           # room label font size
ANN_FONT = 0.020             # annotation font size
HEADER_FONT = 0.028          # floor header font size
ANN_LINE_H = 0.026
ANN_WRAP_CHARS = 18
CHAR_W_FACTOR = 0.55         # approx glyph width / font size for Arial-bold

# ── layout constants (world units) ────────────────────────────────────────────
PLAN_H = 0.62                # height of the plan drawing area per floor
SIDE_COL_W = 0.30            # width of a left/right annotation column
BAND_H = 0.17                # height of a top/bottom annotation band
HEADER_H = 0.06
FLOOR_GAP = 0.10             # gap between floor blocks
WALL_W = 0.0075              # double-wall total thickness

# Surveyor shorthand → full description (extend per client rules as needed)
MATERIAL_EXPANSIONS: Dict[str, str] = {
    "TC": "Textured coating to plasterboard ceiling",
    "TBA": "Floor tiles with bitumen adhesive below carpet",
    "FTBA": "Floor tiles with bitumen adhesive",
    "VBA": "Vinyl with bitumen adhesive",
    "VB": "Vinyl with bitumen to the floor",
    "FT": "Floor tiles",
    "VT": "Vinyl tiles",
    "AIB": "Asbestos insulating board",
    "IB": "Insulating board",
    "BITUMEN": "Bitumen",
    "MASTIC": "Mastic",
    "PUTTY": "Window putty",
    "ROPE": "Rope seal",
    "GASKET": "Gasket",
    "CEMENT": "Cement",
}


def expand_material(material: str) -> str:
    """Expand surveyor shorthand ("TC") to the full description."""
    key = (material or "").strip().upper().replace(".", "")
    return MATERIAL_EXPANSIONS.get(key, material or "")


def sample_display_lines(s: SampleAnnotation, wrap_chars: int = ANN_WRAP_CHARS) -> List[str]:
    """["Ref S001", "Textured coating", "to plasterboard", "ceiling"]"""
    head = f"Ref {s.sample_id}" if s.is_ref else (s.sample_id or "S?")
    body = s.text or expand_material(s.material)
    lines = [head.strip()]
    if body:
        lines += wrap_text(body, wrap_chars)
    return lines


def wrap_text(text: str, max_chars: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def room_label_pos(plan, room: RoomPolygon) -> Tuple[float, float]:
    """
    Normalised label anchor.  Default: area centroid.  If a stair polygon
    or a loft-hatch symbol sits near the centroid, anchor the label toward
    the room's top edge instead (labels must never overlap symbols).
    """
    cx, cy = room.centroid()
    x0, y0, x1, y1 = room.bbox
    if x1 <= x0 or y1 <= y0:
        return (cx, cy)

    def near(px, py):
        return abs(px - cx) < (x1 - x0) * 0.35 and abs(py - cy) < (y1 - y0) * 0.3

    blocked = False
    for st_ in getattr(plan, "stairs", []):
        if getattr(st_, "floor_idx", 0) != room.floor_idx or len(st_.polygon) < 3:
            continue
        sx = sum(p[0] for p in st_.polygon) / len(st_.polygon)
        sy = sum(p[1] for p in st_.polygon) / len(st_.polygon)
        if near(sx, sy):
            blocked = True
    for hx, hy in getattr(plan, "hatch_symbols", []):
        if x0 <= hx <= x1 and y0 <= hy <= y1 and near(hx, hy):
            blocked = True
    if blocked:
        return (cx, y0 + (y1 - y0) * 0.18)
    return (cx, cy)


def room_label_lines(room: RoomPolygon) -> List[str]:
    """Two-line Acorn label: number on top, name below."""
    lines = []
    if room.number:
        lines.append(str(room.number))
    name = room.label or room.ocr_text or ""
    if name:
        lines += wrap_text(name, 14)
    return lines


# ── layout dataclasses ────────────────────────────────────────────────────────

@dataclass
class FloorBlock:
    """One floor laid out into world coordinates."""
    floor_idx: int
    label: str
    # plan drawing rect in world coords
    plan_x: float = 0.0
    plan_y: float = 0.0
    plan_w: float = 0.0
    plan_h: float = 0.0
    header_pos: Tuple[float, float] = (0.0, 0.0)
    # source content bbox in normalised [0,1] image space
    src_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    # image_w / image_h — normalised space is ANISOTROPIC for non-square
    # images; all transforms must re-stretch x by this factor so drawn
    # proportions match the sketch (a tall block stays tall).
    img_aspect: float = 1.0

    def tf(self, nx: float, ny: float) -> Tuple[float, float]:
        """normalised image point → world point (true-aspect preserving)."""
        x0, y0, x1, y1 = self.src_bbox
        ar = self.img_aspect
        sw = max((x1 - x0) * ar, 1e-6)   # true (pixel-proportional) width
        sh = max(y1 - y0, 1e-6)
        s = min(self.plan_w / sw, self.plan_h / sh)
        ox = self.plan_x + (self.plan_w - sw * s) / 2
        oy = self.plan_y + (self.plan_h - sh * s) / 2
        return (ox + (nx - x0) * ar * s, oy + (ny - y0) * s)

    def tf_poly(self, poly) -> List[Tuple[float, float]]:
        return [self.tf(x, y) for x, y in poly]

    @property
    def scale(self) -> float:
        """world units per normalised-x unit (use for widths/radii)."""
        x0, y0, x1, y1 = self.src_bbox
        ar = self.img_aspect
        sw = max((x1 - x0) * ar, 1e-6)
        sh = max(y1 - y0, 1e-6)
        return min(self.plan_w / sw, self.plan_h / sh) * ar


@dataclass
class AnnotationPlacement:
    """A placed sample annotation: text block + leader arrow (world coords)."""
    lines: List[str]
    x: float                    # text block centre x
    y: float                    # text block top y
    colour: str = ANN_BLACK
    arrow_start: Tuple[float, float] = (0.0, 0.0)
    arrow_end: Tuple[float, float] = (0.0, 0.0)

    @property
    def height(self) -> float:
        return len(self.lines) * ANN_LINE_H


@dataclass
class PlanLayout:
    """Full world-space layout: floor blocks + placed annotations."""
    blocks: Dict[int, FloorBlock] = field(default_factory=dict)
    annotations: List[AnnotationPlacement] = field(default_factory=list)
    world_w: float = 1.0
    world_h: float = 1.0


# ── layout engine ─────────────────────────────────────────────────────────────

def layout_plan(plan: PlanModel) -> PlanLayout:
    """
    Arrange floors side by side (Acorn convention) and place all sample
    annotations outside each floor's plan rect with leader arrows.
    """
    floors = sorted({r.floor_idx for r in plan.rooms}) or [0]
    layout = PlanLayout()

    # Which sides of each floor need annotation space?
    ann_by_floor: Dict[int, List[SampleAnnotation]] = {f: [] for f in floors}
    for s in plan.samples:
        f = s.floor_idx if s.floor_idx in ann_by_floor else floors[0]
        ann_by_floor[f].append(s)

    cursor_x = 0.0
    world_h = HEADER_H + BAND_H + PLAN_H + BAND_H
    img_aspect = (plan.image_width / plan.image_height
                  if plan.image_width and plan.image_height else 1.0)
    for fi in floors:
        src = _floor_content_bbox(plan, fi)
        # true (pixel-proportional) aspect, not normalised-space aspect
        aspect = ((src[2] - src[0]) * img_aspect) / max(src[3] - src[1], 1e-6)
        plan_w = max(PLAN_H * aspect, 0.2)

        sides = _annotation_sides(plan, ann_by_floor[fi], src)
        left_w = SIDE_COL_W if sides["left"] else 0.04
        right_w = SIDE_COL_W if sides["right"] else 0.04

        block = FloorBlock(
            floor_idx=fi,
            label=_floor_label(plan, fi),
            plan_x=cursor_x + left_w,
            plan_y=HEADER_H + BAND_H,
            plan_w=plan_w,
            plan_h=PLAN_H,
            src_bbox=src,
            img_aspect=img_aspect,
        )
        block.header_pos = (block.plan_x, HEADER_H * 0.8)
        layout.blocks[fi] = block

        _place_floor_annotations(plan, block, sides, layout.annotations)

        cursor_x = block.plan_x + plan_w + right_w + FLOOR_GAP

    layout.world_w = max(cursor_x - FLOOR_GAP, 0.5)
    layout.world_h = world_h
    return layout


def _floor_label(plan: PlanModel, fi: int) -> str:
    for r in plan.rooms:
        if r.floor_idx == fi and r.floor_label:
            return r.floor_label
    if 0 <= fi < len(plan.floor_labels):
        return plan.floor_labels[fi]
    return {0: "Ground Floor", 1: "First Floor", 2: "Loft"}.get(fi, f"Floor {fi}")


def _floor_content_bbox(plan: PlanModel, fi: int) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for r in plan.rooms:
        if r.floor_idx == fi:
            xs += [p[0] for p in r.polygon]
            ys += [p[1] for p in r.polygon]
    for st in plan.stairs:
        if getattr(st, "floor_idx", 0) == fi:
            xs += [p[0] for p in st.polygon]
            ys += [p[1] for p in st.polygon]
    if not xs:
        return (0.0, 0.0, 1.0, 1.0)
    # include floor-boundary outline(s) belonging to this floor (a boundary
    # belongs here if it contains a room centroid of this floor) so the
    # block covers the full drawn footprint, not just labeled rooms
    for bpoly in getattr(plan, "floor_boundary", []) or []:
        if len(bpoly) < 3:
            continue
        owns = False
        for r in plan.rooms:
            if r.floor_idx == fi and _pt_in_poly(*r.centroid(), bpoly):
                owns = True
                break
        if owns:
            xs += [p[0] for p in bpoly]
            ys += [p[1] for p in bpoly]
    return (min(xs), min(ys), max(xs), max(ys))


def _pt_in_poly(x, y, poly) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _annotation_sides(plan, samples, src) -> Dict[str, List]:
    """Assign each sample to the nearest side of the floor's content bbox."""
    sides: Dict[str, List] = {"left": [], "right": [], "top": [], "bottom": []}
    x0, y0, x1, y1 = src
    for s in samples:
        tx, ty = s.target
        u = (tx - x0) / max(x1 - x0, 1e-6)
        v = (ty - y0) / max(y1 - y0, 1e-6)
        d = {"left": u, "right": 1 - u, "top": v, "bottom": 1 - v}
        side = min(d, key=d.get)
        # Rebalance: keep vertical bands from over-filling
        if side in ("top", "bottom") and len(sides[side]) >= 3:
            side = "left" if u < 0.5 else "right"
        sides[side].append(s)
    return sides


def _place_floor_annotations(plan, block: FloorBlock, sides, out: List[AnnotationPlacement]):
    bx, by, bw, bh = block.plan_x, block.plan_y, block.plan_w, block.plan_h

    def target_world(s: SampleAnnotation) -> Tuple[float, float]:
        return block.tf(*s.target)

    def room_poly_world(s: SampleAnnotation) -> Optional[List[Tuple[float, float]]]:
        if 0 <= s.room_idx < len(plan.rooms):
            return block.tf_poly(plan.rooms[s.room_idx].polygon)
        return None

    def make(s, tx_c, ty_top, from_side):
        lines = sample_display_lines(s)
        h = len(lines) * ANN_LINE_H
        tw = max(len(l) for l in lines) * ANN_FONT * CHAR_W_FACTOR
        tgt = target_world(s)
        if from_side == "left":
            start = (tx_c + tw / 2 + 0.012, ty_top + h / 2)
        elif from_side == "right":
            start = (tx_c - tw / 2 - 0.012, ty_top + h / 2)
        elif from_side == "top":
            start = (tx_c, ty_top + h + 0.008)
        else:
            start = (tx_c, ty_top - 0.008)
        end = _clip_arrow(start, tgt, room_poly_world(s))
        out.append(AnnotationPlacement(
            lines=lines, x=tx_c, y=ty_top,
            colour=ANN_RED if s.acm_positive else ANN_BLACK,
            arrow_start=start, arrow_end=end,
        ))

    # left / right columns — stack by target y
    for side in ("left", "right"):
        items = sorted(sides[side], key=lambda s: s.target[1])
        if not items:
            continue
        total_h = sum(len(sample_display_lines(s)) * ANN_LINE_H + 0.03 for s in items)
        y = max(by + (bh - total_h) / 2, 0.02)
        xc = bx - SIDE_COL_W / 2 if side == "left" else bx + bw + SIDE_COL_W / 2
        for s in items:
            make(s, xc, y, side)
            y += len(sample_display_lines(s)) * ANN_LINE_H + 0.03

    # top / bottom bands — spread by target x using real text widths so
    # neighbouring annotation blocks never overlap
    for side in ("top", "bottom"):
        items = sorted(sides[side], key=lambda s: s.target[0])
        if not items:
            continue
        gap = 0.03
        widths = [
            max(len(l) for l in sample_display_lines(s)) * ANN_FONT * CHAR_W_FACTOR
            for s in items
        ]
        total = sum(widths) + gap * (len(items) - 1)
        x = bx + bw / 2 - total / 2
        for s, w in zip(items, widths):
            xc = x + w / 2
            h = len(sample_display_lines(s)) * ANN_LINE_H
            ty = (by - 0.02 - h) if side == "top" else (by + bh + 0.03)
            make(s, xc, ty, side)
            x += w + gap


def _clip_arrow(start, end, poly) -> Tuple[float, float]:
    """Stop the leader arrow at the room boundary instead of the centroid."""
    if not poly or len(poly) < 3:
        return end
    best_t = None
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        t = _seg_intersect_t(start, end, p1, p2)
        if t is not None and (best_t is None or t < best_t):
            best_t = t
    if best_t is None:
        return end
    t = min(best_t + 0.04, 1.0)  # nudge slightly inside the wall
    return (start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t)


def _seg_intersect_t(a, b, c, d) -> Optional[float]:
    """Param t along a→b of intersection with segment c→d (None if none)."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def hex_to_bgr(hx: str) -> Tuple[int, int, int]:
    hx = hx.lstrip("#")
    return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
