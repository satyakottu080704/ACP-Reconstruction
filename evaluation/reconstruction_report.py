"""
reconstruction_report.py — combined reconstruction quality report.

Produces a JSON-serialisable report matching the Acorn target format:

{
  "Room Accuracy": 94.2,
  "Wall Accuracy": 90.1,
  "Polygon Accuracy": 91.5,
  "OCR Accuracy": 97.8,
  "Overall Reconstruction": 93.4
}

Plus detailed per-metric sub-scores.

Metrics:
  Room IoU         — matched predicted vs GT room polygons
  Wall IoU         — matched predicted vs GT wall polygons
  Polygon IoU      — combined room+wall IoU
  Hausdorff        — boundary distance (lower = better; normalised to 0-100)
  Wall Continuity  — fraction of predicted wall pixels within tolerance of GT wall
  Door Alignment   — fraction of predicted doors within tolerance of GT doors
  OCR Token Acc    — token-level accuracy of room labels vs GT labels
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .geometry_benchmark import polygon_iou, match_polygons, hausdorff_distance


def generate_report(
    pred_plan,
    gt_path: str,
    iou_threshold: float = 0.3,
    wall_tolerance: float = 0.02,
    door_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Generate the full reconstruction report for one predicted plan vs ground truth.

    Args:
        pred_plan: PlanModel
        gt_path:   path to ground-truth JSON (same format as geometry_benchmark)
        iou_threshold: minimum IoU to count as a match
        wall_tolerance: normalised distance tolerance for wall continuity
        door_tolerance: normalised distance tolerance for door alignment

    Returns:
        dict with "Room Accuracy", "Wall Accuracy", "Polygon Accuracy",
        "OCR Accuracy", "Overall Reconstruction", and detailed sub-scores.
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        return {"error": f"GT not found: {gt_path}"}

    with gt_path.open(encoding="utf-8") as f:
        gt = json.load(f)

    gt_rooms = gt.get("rooms", [])
    gt_walls = gt.get("walls", [])
    gt_doors = gt.get("doors", [])

    # ── Room IoU ──────────────────────────────────────────────────────────────
    pred_room_polys = [r.polygon for r in pred_plan.rooms if len(r.polygon) >= 3]
    gt_room_polys = [r.get("polygon", []) for r in gt_rooms if len(r.get("polygon", [])) >= 3]

    room_matched, _, _ = match_polygons(pred_room_polys, gt_room_polys, iou_threshold)
    room_ious = [polygon_iou(pred_room_polys[pi], gt_room_polys[gi])
                 for pi, gi in room_matched]
    room_iou_mean = _mean(room_ious)

    # ── Wall IoU ──────────────────────────────────────────────────────────────
    pred_wall_polys = [w.points for w in pred_plan.walls if len(w.points) >= 2]
    gt_wall_polys = [w.get("points", []) for w in gt_walls if len(w.get("points", [])) >= 2]

    wall_iou_mean = 0.0
    if pred_wall_polys and gt_wall_polys:
        wall_matched, _, _ = match_polygons(pred_wall_polys, gt_wall_polys, iou_threshold)
        wall_ious = [polygon_iou(pred_wall_polys[pi], gt_wall_polys[gi])
                     for pi, gi in wall_matched]
        wall_iou_mean = _mean(wall_ious)

    # ── Polygon IoU (rooms + walls combined) ─────────────────────────────────
    poly_iou_mean = _mean(room_ious + (wall_ious if gt_wall_polys else []))

    # ── Hausdorff ─────────────────────────────────────────────────────────────
    hd_values = [hausdorff_distance(pred_room_polys[pi], gt_room_polys[gi])
                 for pi, gi in room_matched]
    hd_mean = _mean(hd_values)
    # Normalise: hd=0 → 100, hd=0.1 → 0 (anything ≥ 0.1 normalised is bad)
    hd_score = max(0.0, 1.0 - hd_mean / 0.1) * 100

    # ── Wall continuity ───────────────────────────────────────────────────────
    wall_cont = _wall_continuity(pred_plan, gt_walls, wall_tolerance)

    # ── Door alignment ────────────────────────────────────────────────────────
    door_align = _door_alignment(pred_plan, gt_doors, door_tolerance)

    # ── OCR accuracy ──────────────────────────────────────────────────────────
    ocr_acc = _ocr_accuracy(pred_plan, gt_rooms, room_matched, pred_room_polys, gt_room_polys)

    # ── Overall ───────────────────────────────────────────────────────────────
    # Weighted: rooms (35%), walls (20%), polygon (15%), door (10%), ocr (20%)
    overall = (
        0.35 * room_iou_mean * 100
        + 0.20 * wall_iou_mean * 100
        + 0.15 * poly_iou_mean * 100
        + 0.10 * door_align
        + 0.20 * ocr_acc
    )

    return {
        # Top-level Acorn format
        "Room Accuracy": round(room_iou_mean * 100, 1),
        "Wall Accuracy": round(wall_iou_mean * 100, 1),
        "Polygon Accuracy": round(poly_iou_mean * 100, 1),
        "OCR Accuracy": round(ocr_acc, 1),
        "Overall Reconstruction": round(overall, 1),
        # Detailed sub-scores
        "detail": {
            "room_iou_mean": round(room_iou_mean, 4),
            "wall_iou_mean": round(wall_iou_mean, 4),
            "polygon_iou_mean": round(poly_iou_mean, 4),
            "hausdorff_mean": round(hd_mean, 4),
            "hausdorff_score_0_100": round(hd_score, 1),
            "wall_continuity_pct": round(wall_cont, 1),
            "door_alignment_pct": round(door_align, 1),
            "ocr_token_accuracy_pct": round(ocr_acc, 1),
            "matched_rooms": len(room_matched),
            "pred_rooms": len(pred_room_polys),
            "gt_rooms": len(gt_room_polys),
        },
    }


def save_report(report: Dict[str, Any], out_path) -> Path:
    """Write report JSON to file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


# ─── sub-metric helpers ───────────────────────────────────────────────────────

def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _wall_continuity(pred_plan, gt_walls: list, tolerance: float) -> float:
    """
    Fraction of predicted wall endpoint pairs within `tolerance` of a GT wall.
    """
    if not gt_walls or not pred_plan.walls:
        return 100.0 if not gt_walls else 0.0

    gt_segs = []
    for w in gt_walls:
        pts = w.get("points", [])
        for k in range(len(pts) - 1):
            gt_segs.append((pts[k], pts[k + 1]))

    total, hit = 0, 0
    for w in pred_plan.walls:
        pts = w.points
        for k in range(len(pts) - 1 if len(pts) > 2 else 1):
            p = pts[k]
            total += 1
            if any(_point_near_seg(p, seg[0], seg[1], tolerance) for seg in gt_segs):
                hit += 1

    return (hit / total * 100) if total > 0 else 100.0


def _door_alignment(pred_plan, gt_doors: list, tolerance: float) -> float:
    """Fraction of predicted doors within `tolerance` of a GT door centre."""
    if not gt_doors:
        return 100.0
    if not pred_plan.doors:
        return 0.0

    gt_centres = [d.get("center", [0, 0]) for d in gt_doors]
    hit = 0
    for d in pred_plan.doors:
        cx, cy = d.center
        if any(_dist2d((cx, cy), gc) <= tolerance for gc in gt_centres):
            hit += 1
    return hit / len(pred_plan.doors) * 100


def _ocr_accuracy(pred_plan, gt_rooms: list, matched, pred_polys, gt_polys) -> float:
    """Token-level OCR accuracy for matched rooms."""
    if not matched:
        return 0.0

    gt_labels = {i: r.get("label", "").lower().split() for i, r in enumerate(gt_rooms)}
    pred_labels = {i: pred_plan.rooms[i].label.lower().split()
                   for i in range(len(pred_plan.rooms))}

    correct, total = 0, 0
    for pi, gi in matched:
        pred_toks = pred_labels.get(pi, [])
        gt_toks = gt_labels.get(gi, [])
        total += max(len(gt_toks), 1)
        correct += sum(1 for t in pred_toks if t in gt_toks)

    return (correct / total * 100) if total > 0 else 0.0


def _dist2d(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _point_near_seg(p, a, b, tol: float) -> bool:
    """Is point p within tol of segment a-b?"""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return _dist2d(p, a) <= tol
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nx_, ny_ = ax + t * dx, ay + t * dy
    return _dist2d((px, py), (nx_, ny_)) <= tol
