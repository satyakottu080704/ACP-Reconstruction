"""
quality.py — per-plan quality scoring and review flag generation.

Produces a QualityReport that tells the pipeline whether the reconstruction
result needs human review before it can be published.

Gate criteria (any flag → needs_review = True):
  - Average detection confidence < 0.3
  - Average mask quality < 0.5
  - Any invalid polygon (< 3 points or zero area)
  - No rooms detected
  - Topology issues: isolated rooms (no adjacency), or doors with no room assignment
  - Export validation failure (caller must set)
"""
from __future__ import annotations

from typing import List

from .plan_model import PlanModel, QualityReport

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


def compute_quality(plan: PlanModel) -> QualityReport:
    """
    Analyse `plan` and return a populated QualityReport.
    Also assigns plan.quality = report.
    """
    report = QualityReport()
    issues_geom: List[str] = []
    issues_topo: List[str] = []
    flags: List[str] = []

    # ── Detection confidence ──────────────────────────────────────────────────
    confs = [r.confidence for r in plan.rooms if r.confidence > 0]
    confs += [s.confidence for s in plan.stairs if s.confidence > 0]
    confs += [d.confidence for d in plan.doors if d.confidence > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    report.detection_conf_avg = round(avg_conf, 4)
    if avg_conf < 0.3:
        flags.append("low_detection_confidence")

    # ── Mask quality ─────────────────────────────────────────────────────────
    mquals = [r.mask_quality for r in plan.rooms if r.mask_quality >= 0]
    avg_mq = sum(mquals) / len(mquals) if mquals else 0.0
    report.mask_quality_avg = round(avg_mq, 4)
    if avg_mq < 0.5:
        flags.append("low_mask_quality")

    # ── Geometry validity ────────────────────────────────────────────────────
    no_rooms = len(plan.rooms) == 0
    if no_rooms:
        issues_geom.append("no_rooms_detected")
        flags.append("no_rooms_detected")

    for i, room in enumerate(plan.rooms):
        if len(room.polygon) < 3:
            issues_geom.append(f"room_{i}_too_few_vertices ({len(room.polygon)})")
            flags.append("invalid_geometry")
        else:
            area = _polygon_area(room.polygon)
            if area < 1e-6:
                issues_geom.append(f"room_{i}_zero_area")
                flags.append("invalid_geometry")
            if _SHAPELY:
                try:
                    p = ShapelyPolygon(room.polygon)
                    if not p.is_valid:
                        issues_geom.append(f"room_{i}_invalid_shapely")
                except Exception:
                    pass

    report.geometry_valid = len(issues_geom) == 0
    report.geometry_issues = issues_geom

    # ── OCR confidence ────────────────────────────────────────────────────────
    ocr_confs = [r.ocr_confidence for r in plan.rooms if r.ocr_confidence > 0]
    avg_ocr = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0
    report.ocr_conf_avg = round(avg_ocr, 4)

    # ── Topology ─────────────────────────────────────────────────────────────
    if plan.adjacency:
        isolated = [i for i, nbrs in plan.adjacency.items() if len(nbrs) == 0]
        if isolated and len(plan.rooms) > 1:
            issues_topo.append(f"isolated_rooms: {isolated}")
            flags.append("isolated_rooms")

    unassigned_doors = [
        i for i, d in enumerate(plan.doors)
        if d.room_a < 0
    ]
    if unassigned_doors:
        issues_topo.append(f"unassigned_doors: {unassigned_doors}")

    report.topology_ok = len(issues_topo) == 0
    report.topology_issues = issues_topo

    # ── Overall score ─────────────────────────────────────────────────────────
    # Weighted average of sub-scores (all [0,1])
    conf_score = min(1.0, avg_conf / 0.7)     # conf of 0.7+ = full score
    geom_score = 1.0 if report.geometry_valid else 0.5
    topo_score = 1.0 if report.topology_ok else 0.7
    mq_score = min(1.0, avg_mq)
    overall = 0.35 * conf_score + 0.25 * geom_score + 0.2 * topo_score + 0.2 * mq_score
    report.overall_score = round(overall, 4)

    # ── Review decision ───────────────────────────────────────────────────────
    report.needs_review = len(flags) > 0 or overall < 0.6
    # merge with flags set earlier in the pipeline (topology, gap filling);
    # do not overwrite them
    existing = list(getattr(plan.quality, "review_flags", []) or [])
    report.review_flags = existing + [f for f in flags if f not in existing]
    report.needs_review = report.needs_review or bool(existing)

    plan.quality = report
    return report


def quality_report_dict(plan: PlanModel) -> dict:
    """Serialise the quality report to a plain dict suitable for JSON export."""
    qr = plan.quality
    return {
        "detection_conf_avg": qr.detection_conf_avg,
        "mask_quality_avg": qr.mask_quality_avg,
        "geometry_valid": qr.geometry_valid,
        "geometry_issues": qr.geometry_issues,
        "ocr_conf_avg": qr.ocr_conf_avg,
        "topology_ok": qr.topology_ok,
        "topology_issues": qr.topology_issues,
        "export_valid": qr.export_valid,
        "overall_score": qr.overall_score,
        "needs_review": qr.needs_review,
        "review_flags": qr.review_flags,
        "room_count": len(plan.rooms),
        "door_count": len(plan.doors),
        "stair_count": len(plan.stairs),
        "has_loft": plan.has_loft,
        "engine_version": plan.engine_version,
    }


def _polygon_area(poly) -> float:
    """Shoelace formula for polygon area."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
