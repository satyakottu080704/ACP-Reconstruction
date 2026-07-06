"""
dxf_export.py — export PlanModel to DXF (AutoCAD/Visio compatible).

Uses ezdxf if available; falls back to a hand-written minimal DXF (R12 ASCII)
that contains just LWPOLYLINE entities — enough for most CAD tools and
cross-platform without any compiled extension.

Layers:
  ROOMS    — room boundary polylines
  WALLS    — wall polylines
  DOORS    — door arcs
  STAIRS   — stair polygons
  LABELS   — TEXT entities for room labels
  ACM      — ACM room fills (HATCH entities, ezdxf only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from reconstruction.plan_model import PlanModel

_SCALE = 1000.0  # normalised [0,1] → mm (1m in DXF units)

try:
    import ezdxf
    _EZDXF = True
except ImportError:
    _EZDXF = False


def export_dxf(plan: PlanModel, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _EZDXF:
        return _export_ezdxf(plan, out_path)
    else:
        return _export_minimal_dxf(plan, out_path)


# ─── ezdxf path ───────────────────────────────────────────────────────────────

def _export_ezdxf(plan: PlanModel, out_path: Path) -> Path:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for layer in ("ROOMS", "WALLS", "DOORS", "STAIRS", "LABELS", "ACM"):
        doc.layers.add(layer)

    def pts3d(poly):
        return [(x * _SCALE, (1 - y) * _SCALE, 0) for x, y in poly]

    # Rooms
    for room in plan.rooms:
        if len(room.polygon) < 3:
            continue
        pts = pts3d(room.polygon + [room.polygon[0]])
        layer = "ACM" if room.is_acm else "ROOMS"
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})

        # ACM hatch
        if room.is_acm and hasattr(ezdxf.entities, "Hatch"):
            try:
                hatch = msp.add_hatch(color=1, dxfattribs={"layer": "ACM"})
                hatch.paths.add_polyline_path(
                    [(x * _SCALE, (1 - y) * _SCALE) for x, y in room.polygon],
                    is_closed=True,
                )
                hatch.set_pattern_fill("ANSI31", scale=5.0)
            except Exception:
                pass

        # Label
        label = room.label or room.ocr_text or ""
        if label:
            cx, cy = room.centroid()
            msp.add_text(
                label,
                height=15,
                dxfattribs={
                    "layer": "LABELS",
                    "insert": (cx * _SCALE, (1 - cy) * _SCALE),
                    "halign": 1, "valign": 2,
                },
            )

    # Walls
    for wall in plan.walls:
        if len(wall.points) < 2:
            continue
        if len(wall.points) == 2:
            p1, p2 = wall.points
            msp.add_line(
                (p1[0] * _SCALE, (1 - p1[1]) * _SCALE),
                (p2[0] * _SCALE, (1 - p2[1]) * _SCALE),
                dxfattribs={"layer": "WALLS"},
            )
        else:
            pts = pts3d(wall.points)
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "WALLS"})

    # Doors
    for door in plan.doors:
        cx = door.center[0] * _SCALE
        cy = (1 - door.center[1]) * _SCALE
        r = max(10.0, door.width * _SCALE / 2)
        msp.add_arc(
            center=(cx, cy),
            radius=r,
            start_angle=door.angle_deg,
            end_angle=door.angle_deg + 90,
            dxfattribs={"layer": "DOORS"},
        )

    # Stairs
    for stair in plan.stairs:
        if len(stair.polygon) < 3:
            continue
        pts = pts3d(stair.polygon + [stair.polygon[0]])
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "STAIRS"})

    doc.saveas(str(out_path))
    return out_path


# ─── minimal hand-written DXF (R12, no dependencies) ─────────────────────────

def _export_minimal_dxf(plan: PlanModel, out_path: Path) -> Path:
    """
    Write a minimal R12 ASCII DXF with LWPOLYLINE-equivalent POLYLINE entities.
    Compatible with AutoCAD, LibreCAD, and Visio import.
    """
    lines = []

    def emit(*args):
        for a in args:
            lines.append(str(a))

    def _header():
        emit("0", "SECTION", "2", "HEADER",
             "9", "$ACADVER", "1", "AC1009",
             "0", "ENDSEC")

    def _entities_start():
        emit("0", "SECTION", "2", "ENTITIES")

    def _entities_end():
        emit("0", "ENDSEC")

    def _polyline(pts_norm, layer: str = "ROOMS", closed: bool = True):
        emit("0", "POLYLINE",
             "8", layer,
             "66", "1",
             "70", "1" if closed else "0")
        for x, y in pts_norm:
            px = x * _SCALE
            py = (1 - y) * _SCALE
            emit("0", "VERTEX",
                 "8", layer,
                 "10", f"{px:.3f}",
                 "20", f"{py:.3f}",
                 "30", "0.0")
        emit("0", "SEQEND")

    def _text(label: str, x: float, y: float, layer: str = "LABELS"):
        px = x * _SCALE
        py = (1 - y) * _SCALE
        emit("0", "TEXT",
             "8", layer,
             "10", f"{px:.3f}",
             "20", f"{py:.3f}",
             "30", "0.0",
             "40", "15",
             "1", label[:255])

    emit("0", "SECTION", "2", "HEADER",
         "9", "$ACADVER", "1", "AC1009",
         "0", "ENDSEC",
         "0", "SECTION", "2", "ENTITIES")

    for room in plan.rooms:
        if len(room.polygon) < 3:
            continue
        layer = "ACM" if room.is_acm else "ROOMS"
        _polyline(room.polygon, layer=layer, closed=True)
        label = room.label or room.ocr_text or ""
        if label:
            cx, cy = room.centroid()
            _text(label, cx, cy)

    for wall in plan.walls:
        if len(wall.points) < 2:
            continue
        _polyline(wall.points, layer="WALLS", closed=len(wall.points) > 2)

    for stair in plan.stairs:
        if len(stair.polygon) < 3:
            continue
        _polyline(stair.polygon, layer="STAIRS", closed=True)

    emit("0", "ENDSEC", "0", "EOF")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
