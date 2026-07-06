"""
engine.py — v2 reconstruction orchestrator.

Pipeline:
  Image → YOLO → masks → morph cleanup → contour → simplify →
  wall merge → orthogonalize/grid-snap → room graph →
  door-on-wall attach → geometry validation → exports

Public API:
  reconstruct_from_masks(detections, image_size, source_image, ...)
  from_yolo_result(result, source_image, ...)        # ultralytics adapter
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .plan_model import (
    PlanModel, RoomPolygon, WallSegment, DoorSegment, StairPolygon
)
from .masks import refine_mask, mask_quality_score
from .polygons import mask_to_polygon, reconstruct_walls
from .cleanup import snap_plan, resolve_overlaps, partition_loft
from .postprocess import (
    orthogonalize_plan, attach_doors_to_walls, orient_stairs, share_walls,
)
from .room_graph import build_adjacency, assign_doors_to_rooms, to_connected_to_json
from .quality import compute_quality


# Class indices from config (must match training order)
_CLS_NAMES = {
    0: "acm",
    1: "door",
    2: "floor",
    3: "room",
    4: "stairs",
    5: "walls",
}
_CLS_IDX = {v: k for k, v in _CLS_NAMES.items()}


def reconstruct_from_masks(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    source_image: Optional[np.ndarray] = None,
    run_ocr: bool = True,
    snap_distance: float = 0.015,
    sliver_threshold: float = 0.005,
    epsilon_factor: float = 0.005,
    min_mask_area: int = 100,
) -> PlanModel:
    """
    Build a PlanModel from a list of raw detections.

    Each detection is a dict:
    {
        "class_name": str,        # "room", "door", "walls", "stairs", "acm", "floor"
        "class_id":   int,
        "confidence": float,
        "mask":       np.ndarray, # uint8 binary mask (H x W), values 0 or 255
        "polygon":    list,       # optional: [(x,y)...] in pixel coords
        "bbox":       (x1,y1,x2,y2),  # optional pixel bbox
    }

    Args:
        detections:   list of detection dicts (see above).
        image_size:   (width, height) of the source image.
        source_image: BGR numpy array for OCR (optional).
        run_ocr:      whether to run OCR on room regions.
        snap_distance: vertex snap threshold (normalised).
        sliver_threshold: overlap sliver threshold (normalised area).
        epsilon_factor:  Douglas-Peucker epsilon as fraction of perimeter.
        min_mask_area:   minimum mask pixel area to process.

    Returns:
        PlanModel with populated rooms, walls, doors, stairs, adjacency, quality.
    """
    w, h = image_size
    plan = PlanModel(image_width=w, image_height=h)
    plan.floor_labels = ["Ground Floor"]

    rooms: List[RoomPolygon] = []
    walls_raw: List[np.ndarray] = []
    doors_raw: List[Dict] = []
    stairs_raw: List[Dict] = []

    for idx, det in enumerate(detections):
        cls = det.get("class_name", "").lower()
        conf = float(det.get("confidence", 1.0))
        mask = det.get("mask")

        if mask is not None and mask.size > 0 and cls != "walls":
            # NOTE: walls are exempt — refine_mask hole-fills, which would
            # flood the area ENCLOSED by wall lines and destroy the thin
            # wall structure. extract_wall_centerlines does its own
            # close/open morphology instead.
            mask = refine_mask(mask, open_k=3, close_k=7, min_area_px=min_mask_area)

        if cls in ("room", "acm"):
            poly = _extract_poly(det, mask, w, h, epsilon_factor)
            if not poly:
                continue
            number, name = _split_label(det.get("label", ""))
            room = RoomPolygon(
                polygon=poly,
                label=name,
                number=det.get("number", "") or number,
                room_type="acm" if cls == "acm" else "clear",
                is_acm=(cls == "acm"),
                confidence=conf,
                mask_quality=mask_quality_score(mask) if mask is not None else 0.5,
                detection_id=idx,
            )
            room.area = _polygon_area(poly)
            x1, y1, x2, y2 = _poly_bbox(poly)
            room.bbox = (x1, y1, x2, y2)
            rooms.append(room)

        elif cls == "walls":
            if mask is not None and mask.size > 0:
                walls_raw.append(mask)

        elif cls == "door":
            poly = _extract_poly(det, mask, w, h, epsilon_factor)
            if poly:
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                door = DoorSegment(
                    center=(cx, cy),
                    polygon=poly,
                    confidence=conf,
                )
                doors_raw.append(door)

        elif cls == "stairs":
            poly = _extract_poly(det, mask, w, h, epsilon_factor)
            if poly:
                stair = StairPolygon(
                    polygon=poly,
                    confidence=conf,
                    detection_id=idx,
                )
                stairs_raw.append(stair)

        elif cls == "floor":
            # Floor mask -> outer boundary polygon (coverage constraint)
            poly = _extract_poly(det, mask, w, h, epsilon_factor)
            if poly and len(poly) >= 3:
                plan.floor_boundary.append(poly)

    plan.rooms = rooms
    plan.doors = doors_raw
    plan.stairs = stairs_raw

    # Reconstruct walls: mask -> skeleton -> centerline segments (preferred);
    # legacy contour-based reconstruction as fallback
    if walls_raw:
        combined_wall = walls_raw[0].copy()
        for m in walls_raw[1:]:
            if m.shape == combined_wall.shape:
                combined_wall = np.maximum(combined_wall, m)
        try:
            from .walls import extract_wall_centerlines
            plan.walls = extract_wall_centerlines(combined_wall, (w, h))
        except Exception:
            plan.walls = []
        if not plan.walls:
            wall_polys = reconstruct_walls(combined_wall)
            plan.walls = [WallSegment(points=p) for p in wall_polys]

    # Topology (polygonization): closed regions between walls become the
    # room geometry; detections are matched to regions by IoU. Mask-based
    # room polygons remain the fallback when walls/floor are missing.
    try:
        from .topology import rooms_from_walls
        rooms_from_walls(plan)
    except Exception:
        pass

    # Geometry cleanup
    snap_plan(plan, snap_distance=snap_distance)
    resolve_overlaps(plan, sliver_threshold=sliver_threshold)
    partition_loft(plan)

    # Post-processing: 90-degree enforcement + grid snap (clean CAD output)
    orthogonalize_plan(plan)

    # Adjacent rooms must share EXACT wall coordinates (no slivers)
    share_walls(plan)

    # Gap detection & filling: rooms must completely fill the floor outline
    try:
        from .coverage import fill_gaps, carve_nested_rooms
        carve_nested_rooms(plan)   # cut nested rooms (CPDs) out of containers
        fill_gaps(plan)
        share_walls(plan)   # re-lock shared coordinates after growth
    except Exception:
        pass

    # Room graph
    build_adjacency(plan)
    assign_doors_to_rooms(plan)
    to_connected_to_json(plan)

    # Doors must lie ON walls (drop floating doors, orient the swing);
    # stair arrows follow the flight's long axis
    attach_doors_to_walls(plan)
    orient_stairs(plan)

    # OCR
    if run_ocr and source_image is not None and source_image.size > 0:
        try:
            from .ocr_rooms import run_ocr_on_rooms
            run_ocr_on_rooms(plan, source_image)
        except Exception:
            pass

    # Quality report + constraint checklist (no gaps/overlaps, inside floor,
    # doors on walls, orthogonality, connectivity)
    compute_quality(plan)
    try:
        from .coverage import validate_constraints
        validate_constraints(plan)
    except Exception:
        pass

    return plan


def from_yolo_result(
    result,
    source_image: Optional[np.ndarray] = None,
    run_ocr: bool = True,
    **kwargs,
) -> PlanModel:
    """
    Adapter for ultralytics YOLO result objects.

    Converts result.masks.xy / result.boxes into the `detections` format
    expected by reconstruct_from_masks().

    Usage:
        from ultralytics import YOLO
        model = YOLO("models/weights/best.pt")
        results = model(image, imgsz=1280, conf=0.15, iou=0.5)
        plan = from_yolo_result(results[0], source_image=image)
    """
    if source_image is not None:
        h, w = source_image.shape[:2]
    elif result.orig_shape:
        h, w = result.orig_shape
    else:
        h, w = 1280, 1280

    detections = []

    # Get masks (preferred: results.masks.xy are polygon contours in pixel space)
    masks_xy = None
    masks_data = None
    if result.masks is not None:
        try:
            masks_xy = result.masks.xy   # list of (N,2) arrays in pixel coords
        except Exception:
            pass
        try:
            masks_data = result.masks.data  # tensor (N, H, W)
        except Exception:
            pass

    boxes = result.boxes
    n = len(boxes) if boxes is not None else 0

    for i in range(n):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        cls_name = _CLS_NAMES.get(cls_id, f"class_{cls_id}")

        # Get pixel-space mask
        mask_np = None
        if masks_data is not None and i < len(masks_data):
            m = masks_data[i].cpu().numpy()
            import cv2
            mask_np = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
        elif masks_xy is not None and i < len(masks_xy):
            pts = masks_xy[i].astype(np.int32)
            mask_np = np.zeros((h, w), dtype=np.uint8)
            import cv2
            cv2.fillPoly(mask_np, [pts.reshape(-1, 1, 2)], 255)

        # Get pixel-space polygon from masks_xy if available
        poly_px = None
        if masks_xy is not None and i < len(masks_xy):
            poly_px = [(float(x), float(y)) for x, y in masks_xy[i]]

        det = {
            "class_name": cls_name,
            "class_id": cls_id,
            "confidence": conf,
            "mask": mask_np,
            "polygon_px": poly_px,
        }

        # Bounding box
        try:
            b = boxes.xyxy[i].cpu().numpy()
            det["bbox"] = tuple(b.tolist())
        except Exception:
            pass

        detections.append(det)

    return reconstruct_from_masks(
        detections, (w, h),
        source_image=source_image,
        run_ocr=run_ocr,
        **kwargs,
    )


# ─── helpers ──────────────────────────────────────────────────────────────────

def _split_label(label: str) -> Tuple[str, str]:
    """'005 Kitchen' → ('005', 'Kitchen'); 'Kitchen' → ('', 'Kitchen')."""
    label = (label or "").strip()
    parts = label.split(None, 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[0], parts[1]
    return "", label


def _extract_poly(det, mask, w, h, epsilon_factor):
    """Get normalised polygon from detection (mask preferred, else bbox fallback)."""
    # Prefer mask
    if mask is not None and mask.size > 0:
        poly = mask_to_polygon(
            mask,
            epsilon_factor=epsilon_factor,
            normalise=True,
        )
        if poly and len(poly) >= 3:
            return poly

    # Try pre-computed pixel polygon
    poly_px = det.get("polygon_px")
    if poly_px and len(poly_px) >= 3:
        return [(float(x) / w, float(y) / h) for x, y in poly_px]

    # Fallback: bbox as rectangle
    bbox = det.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        return [
            (x1 / w, y1 / h), (x2 / w, y1 / h),
            (x2 / w, y2 / h), (x1 / w, y2 / h),
        ]
    return []


def _polygon_area(poly) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _poly_bbox(poly) -> Tuple[float, float, float, float]:
    if not poly:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))
