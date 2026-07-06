"""
plan_model.py — canonical data model for the v2 reconstruction engine.

Single source of truth consumed by all exporters (JSON, PNG, SVG, DXF, PDF, VSDX).
Coordinates are always in normalised [0, 1] space (x/image_w, y/image_h)
so exporters can scale to any output resolution independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class RoomPolygon:
    """One detected room, represented as a closed polygon."""
    polygon: List[Tuple[float, float]]  # [(x, y), ...] normalised [0,1]
    label: str = ""                      # "Kitchen", "Living Room", ...
    number: str = ""                     # "001", "004" - surveyor room number
    room_type: str = "clear"             # "clear" | "acm" | "no_access" | "loft"
    no_access: bool = False              # solid dark fill in Acorn style
    floor_idx: int = 0                   # 0=Ground, 1=First, 2=Loft, ...
    floor_label: str = "Ground Floor"
    confidence: float = 1.0
    mask_quality: float = 1.0            # 0-1 from masks.mask_quality_score()
    ocr_text: str = ""                   # raw OCR hit inside the room
    ocr_confidence: float = 0.0
    is_acm: bool = False
    is_loft: bool = False
    area: float = 0.0                    # normalised area (polygon area in [0,1]^2)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x1,y1,x2,y2
    detection_id: int = -1               # index in original YOLO detections list

    def centroid(self) -> Tuple[float, float]:
        """Area (shoelace) centroid — robust to uneven vertex density.
        Falls back to the vertex mean for degenerate polygons."""
        if not self.polygon:
            return (0.0, 0.0)
        n = len(self.polygon)
        if n >= 3:
            a = cx = cy = 0.0
            for i in range(n):
                x1, y1 = self.polygon[i]
                x2, y2 = self.polygon[(i + 1) % n]
                cross = x1 * y2 - x2 * y1
                a += cross
                cx += (x1 + x2) * cross
                cy += (y1 + y2) * cross
            if abs(a) > 1e-12:
                return (cx / (3.0 * a), cy / (3.0 * a))
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class WallSegment:
    """A wall represented as a line segment (centre-line) or polygon."""
    points: List[Tuple[float, float]]   # 2 points for line, N for polygon
    thickness: float = 0.01             # normalised wall thickness
    is_exterior: bool = False
    connected_rooms: List[int] = field(default_factory=list)  # RoomPolygon indices


@dataclass
class DoorSegment:
    """A detected door - wall gap position + swing arc."""
    center: Tuple[float, float]         # normalised (x, y)
    width: float = 0.05                 # normalised opening width
    angle_deg: float = 0.0              # hinge direction for swing arc
    polygon: List[Tuple[float, float]] = field(default_factory=list)  # raw mask polygon
    room_a: int = -1                    # index of room on one side
    room_b: int = -1                    # index of room on other side
    confidence: float = 1.0


@dataclass
class StairPolygon:
    """Detected stairs - polygon + direction arrow."""
    polygon: List[Tuple[float, float]]
    direction_deg: float = 270.0        # angle of the UP arrow (270=upward in image)
    floor_idx: int = 0
    label: str = "Stairs"
    confidence: float = 1.0
    detection_id: int = -1


@dataclass
class SampleAnnotation:
    """
    A surveyor sample annotation ("S001 TC", "Ref S002 TBA", ...).

    Rendered in Acorn style as bold text OUTSIDE the plan with a straight
    leader arrow pointing at the target room.  Red text/arrow when the
    sample is ACM-positive, black otherwise.
    """
    sample_id: str = ""                 # "S001", "P002"
    material: str = ""                  # "TC", "TBA", "Mastic", ...
    text: str = ""                      # expanded description (fallback: material)
    is_ref: bool = False                # "Ref S001 ..." cross-reference
    acm_positive: bool = False          # red text + red arrow when True
    target: Tuple[float, float] = (0.0, 0.0)  # normalised point the arrow hits
    room_idx: int = -1                  # index into PlanModel.rooms (-1 = none)
    floor_idx: int = 0


@dataclass
class QualityReport:
    """Per-image quality metrics produced by quality.py."""
    detection_conf_avg: float = 0.0
    mask_quality_avg: float = 0.0
    geometry_valid: bool = True
    geometry_issues: List[str] = field(default_factory=list)
    ocr_conf_avg: float = 0.0
    topology_ok: bool = True
    topology_issues: List[str] = field(default_factory=list)
    export_valid: bool = True
    overall_score: float = 0.0         # 0-1
    needs_review: bool = False
    review_flags: List[str] = field(default_factory=list)


@dataclass
class PlanModel:
    """
    Canonical floor-plan model. Single source of truth for all exporters.
    All geometry is in normalised [0,1] coordinate space.
    """
    rooms: List[RoomPolygon] = field(default_factory=list)
    walls: List[WallSegment] = field(default_factory=list)
    doors: List[DoorSegment] = field(default_factory=list)
    stairs: List[StairPolygon] = field(default_factory=list)
    samples: List[SampleAnnotation] = field(default_factory=list)
    # Loft-hatch / skylight symbols (box with X) - normalised points
    hatch_symbols: List[Tuple[float, float]] = field(default_factory=list)
    # Outer boundary polygon(s) from the 'floor' class (normalised)
    floor_boundary: List[List[Tuple[float, float]]] = field(default_factory=list)

    # Room adjacency graph: {room_idx: [room_idx, ...]}
    adjacency: Dict[int, List[int]] = field(default_factory=dict)

    # Human-readable connectivity: {"Living Room": {"connected_to": ["Kitchen", ...]}}
    connectivity: Dict[str, Any] = field(default_factory=dict)

    quality: QualityReport = field(default_factory=QualityReport)

    # Metadata
    image_width: int = 0
    image_height: int = 0
    source_image: str = ""
    engine_version: str = "v2"
    has_loft: bool = False
    floor_labels: List[str] = field(default_factory=list)

    def rooms_on_floor(self, floor_idx: int) -> List[RoomPolygon]:
        return [r for r in self.rooms if r.floor_idx == floor_idx]

    def acm_rooms(self) -> List[RoomPolygon]:
        return [r for r in self.rooms if r.is_acm or r.room_type == "acm"]

    def loft_rooms(self) -> List[RoomPolygon]:
        return [r for r in self.rooms if r.is_loft or r.room_type == "loft"]

    def to_pixel_polygon(self, room: RoomPolygon) -> List[Tuple[int, int]]:
        """Convert normalised polygon to pixel coordinates."""
        w, h = self.image_width or 1, self.image_height or 1
        return [(int(x * w), int(y * h)) for x, y in room.polygon]

    def samples_on_floor(self, floor_idx: int) -> List[SampleAnnotation]:
        return [s for s in self.samples if s.floor_idx == floor_idx]


def from_legacy_floorplan(fp) -> "PlanModel":
    """
    Adapter: legacy ``pipeline.FloorPlan`` -> v2 ``PlanModel``.

    Duck-typed (no import of pipeline.py) so it works with any object exposing
    the legacy attributes: rooms (bbox px, label, number, room_type, no_access,
    floor, floor_idx, contour), samples (id, material, x_pct, y_pct,
    acm_positive, is_ref, target_room_number, target_floor_idx),
    doors/stairs dicts, image_size, floor_names.
    """
    w, h = (fp.image_size or (1, 1))
    w, h = max(int(w), 1), max(int(h), 1)
    plan = PlanModel(image_width=w, image_height=h,
                     source_image=getattr(fp, "project_number", "") or "")

    floor_names = dict(getattr(fp, "floor_names", {}) or {})
    for room in getattr(fp, "rooms", []) or []:
        contour = getattr(room, "contour", None)
        if contour is not None and len(contour) >= 3:
            pts = contour.reshape(-1, 2)
            poly = [(float(x) / w, float(y) / h) for x, y in pts]
        else:
            x, y, bw, bh = room.bbox
            poly = [(x / w, y / h), ((x + bw) / w, y / h),
                    ((x + bw) / w, (y + bh) / h), (x / w, (y + bh) / h)]
        fi = int(getattr(room, "floor_idx", 0) or 0)
        r = RoomPolygon(
            polygon=poly,
            label=getattr(room, "label", "") or "",
            number=str(getattr(room, "number", "") or ""),
            room_type=getattr(room, "room_type", "clear") or "clear",
            no_access=bool(getattr(room, "no_access", False)),
            is_acm=bool(getattr(room, "has_acm", False))
                   or getattr(room, "room_type", "") == "acm",
            floor_idx=fi,
            floor_label=floor_names.get(fi) or getattr(room, "floor", "Ground Floor"),
            confidence=float(getattr(room, "detection_confidence", None) or 1.0),
        )
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        r.bbox = (min(xs), min(ys), max(xs), max(ys))
        r.area = abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                         - poly[(i + 1) % len(poly)][0] * poly[i][1]
                         for i in range(len(poly)))) / 2.0
        if r.is_acm:
            r.room_type = "acm"
        plan.rooms.append(r)

    # Samples -> annotations; resolve target room by number when available
    num_to_idx = {}
    for i, r in enumerate(plan.rooms):
        if r.number:
            num_to_idx.setdefault((r.number.lstrip("0") or "0", r.floor_idx), i)
    for s in getattr(fp, "samples", []) or []:
        tgt_num = getattr(s, "target_room_number", None)
        tfi = int(getattr(s, "target_floor_idx", 0) or 0)
        room_idx = -1
        if tgt_num:
            room_idx = num_to_idx.get((str(tgt_num).lstrip("0") or "0", tfi), -1)
        if room_idx >= 0:
            cx, cy = plan.rooms[room_idx].centroid()
        else:
            cx = float(getattr(s, "x_pct", 0)) / 100.0
            cy = float(getattr(s, "y_pct", 0)) / 100.0
        plan.samples.append(SampleAnnotation(
            sample_id=getattr(s, "id", "") or "",
            material=getattr(s, "material", "") or "",
            is_ref=bool(getattr(s, "is_ref", False)),
            acm_positive=bool(getattr(s, "acm_positive", False)),
            target=(cx, cy),
            room_idx=room_idx,
            floor_idx=tfi,
        ))

    for d in getattr(fp, "doors", []) or []:
        if isinstance(d, dict) and "x" in d and "y" in d:
            plan.doors.append(DoorSegment(
                center=(float(d["x"]) / w, float(d["y"]) / h),
                width=float(d.get("width", 0.05 * w)) / w,
                angle_deg=float(d.get("angle", 0.0)),
            ))

    floors = sorted({r.floor_idx for r in plan.rooms}) or [0]
    plan.floor_labels = [
        floor_names.get(fi) or next(
            (r.floor_label for r in plan.rooms if r.floor_idx == fi),
            f"Floor {fi}",
        )
        for fi in floors
    ]
    plan.has_loft = any(r.is_loft or r.room_type == "loft" for r in plan.rooms)
    return plan
