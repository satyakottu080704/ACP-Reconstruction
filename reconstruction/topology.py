"""
topology.py — room polygons FROM wall topology (diagram stages 5-6:
"Topology (polygonization): build wall graph, find closed regions, each
region = room candidate" and "Room label assignment: IoU box <-> polygon").

Given wall centerlines + the floor boundary, the enclosed regions between
walls ARE the rooms — closed by construction, sharing walls by construction.
Detected room masks/boxes are then matched to these cells by IoU:

  - cell matched to a detection (IoU >= threshold): the detection's room
    takes the CELL polygon (crisper than its mask, topology-correct);
  - detection with no matching cell: keeps its mask polygon (wall graph
    locally incomplete) — flagged;
  - large cell with no detection: added as an unlabeled room candidate —
    flagged for review (a room YOLO missed).

Raster-based (OpenCV + NumPy only), consistent with coverage.py.
"""
from __future__ import annotations

import numpy as np
import cv2

from .plan_model import PlanModel, RoomPolygon

_RASTER = 1024


def _to_px(v: float) -> int:
    return int(v * (_RASTER - 1))


def _rasterize_poly(poly) -> np.ndarray:
    m = np.zeros((_RASTER, _RASTER), np.uint8)
    if len(poly) >= 3:
        pts = np.array([(_to_px(x), _to_px(y)) for x, y in poly], np.int32)
        cv2.fillPoly(m, [pts], 255)
    return m


def rooms_from_walls(
    plan: PlanModel,
    min_cell_frac: float = 0.005,
    iou_match: float = 0.40,
) -> PlanModel:
    """
    Build closed regions (cells) from wall centerlines inside each floor
    boundary and reconcile them with detected rooms by IoU.

    Requires plan.walls (centerline segments) and plan.floor_boundary;
    no-op otherwise, so mask-based rooms remain the fallback path.
    """
    if not plan.walls or not plan.floor_boundary:
        return plan

    from .polygons import mask_to_polygon
    from .postprocess import orthogonalize_polygon, snap_to_grid

    for boundary in plan.floor_boundary:
        if len(boundary) < 3:
            continue
        floor = _rasterize_poly(boundary)
        floor_area = int(np.count_nonzero(floor))
        if floor_area == 0:
            continue

        # rooms belonging to this boundary (centroid containment)
        member_idx = [i for i, r in enumerate(plan.rooms)
                      if len(r.polygon) >= 3 and
                      cv2.pointPolygonTest(
                          np.array([(_to_px(x), _to_px(y)) for x, y in boundary],
                                   np.int32),
                          (_to_px(r.centroid()[0]), _to_px(r.centroid()[1])),
                          False) >= 0]

        # wall raster: draw centerlines with their thickness
        wall_ras = np.zeros((_RASTER, _RASTER), np.uint8)
        drew = False
        for w in plan.walls:
            if len(w.points) < 2:
                continue
            mx = sum(p[0] for p in w.points) / len(w.points)
            my = sum(p[1] for p in w.points) / len(w.points)
            if cv2.pointPolygonTest(
                    np.array([(_to_px(x), _to_px(y)) for x, y in boundary],
                             np.int32),
                    (_to_px(mx), _to_px(my)), False) < 0:
                continue
            t = max(3, int(w.thickness * _RASTER * 1.5))
            # extend endpoints by ~2x thickness so T-junctions always seal
            # (diagram stage 4: "extend lines to intersections")
            ext = 2.0 * t
            for i in range(len(w.points) - 1):
                x1, y1 = _to_px(w.points[i][0]), _to_px(w.points[i][1])
                x2, y2 = _to_px(w.points[i + 1][0]), _to_px(w.points[i + 1][1])
                # extend the outermost sub-segments beyond their endpoints
                import math as _m
                dxp, dyp = x2 - x1, y2 - y1
                lp = _m.hypot(dxp, dyp)
                if lp > 1e-6:
                    ux, uy = dxp / lp, dyp / lp
                    if i == 0:
                        x1 = int(x1 - ux * ext); y1 = int(y1 - uy * ext)
                    if i == len(w.points) - 2:
                        x2 = int(x2 + ux * ext); y2 = int(y2 + uy * ext)
                cv2.line(wall_ras, (x1, y1), (x2, y2), 255, t)
            drew = True
        if not drew:
            continue
        # Seal door openings at the VECTOR level: bridge near-collinear
        # wall segments across gaps up to ~door width (a door gap is a
        # break in a collinear wall run).  Raster closing with a fat
        # kernel would also swallow narrow rooms (cupboards), so it is
        # only used with a small kernel for pen noise afterwards.
        from .walls import _try_merge
        segs2 = [(w.points[0], w.points[-1]) for w in plan.walls
                 if len(w.points) >= 2]
        t_draw = max(3, int(plan.walls[0].thickness * _RASTER * 1.5))             if plan.walls else 3
        for i in range(len(segs2)):
            for j in range(i + 1, len(segs2)):
                bridged = _try_merge(segs2[i], segs2[j],
                                     angle_tol_deg=7.0, gap_tol=0.10,
                                     offset_tol=0.006)
                if bridged is not None:
                    (bx1, by1), (bx2, by2) = bridged
                    cv2.line(wall_ras,
                             (_to_px(bx1), _to_px(by1)),
                             (_to_px(bx2), _to_px(by2)), 255, t_draw)
        wall_ras = cv2.morphologyEx(
            wall_ras, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

        # enclosed cells = floor minus walls
        open_space = cv2.bitwise_and(
            cv2.erode(floor, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))),
            cv2.bitwise_not(wall_ras))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(open_space, 4)

        cells = []
        for ci in range(1, n):
            if stats[ci, cv2.CC_STAT_AREA] / floor_area < min_cell_frac:
                continue
            cells.append((labels == ci).astype(np.uint8) * 255)
        if not cells:
            continue

        # IoU match: detected rooms <-> cells
        room_masks = {i: _rasterize_poly(plan.rooms[i].polygon) for i in member_idx}
        used_cells = set()
        for i in member_idx:
            best_ci, best_iou = -1, 0.0
            for ci, cell in enumerate(cells):
                if ci in used_cells:
                    continue
                inter = int(np.count_nonzero(cv2.bitwise_and(room_masks[i], cell)))
                union = int(np.count_nonzero(cv2.bitwise_or(room_masks[i], cell)))
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_ci, best_iou = ci, iou
            if best_ci >= 0 and best_iou >= iou_match:
                poly = mask_to_polygon(cells[best_ci], epsilon_factor=0.004,
                                       normalise=True)
                if poly and len(poly) >= 3:
                    plan.rooms[i].polygon = snap_to_grid(
                        orthogonalize_polygon(poly, 15.0), 0.004)
                    xs = [p[0] for p in plan.rooms[i].polygon]
                    ys = [p[1] for p in plan.rooms[i].polygon]
                    plan.rooms[i].bbox = (min(xs), min(ys), max(xs), max(ys))
                    used_cells.add(best_ci)
            else:
                plan.quality.review_flags.append(
                    f"room_{i}_topology_incomplete_mask_fallback")

        # large unmatched cells = rooms YOLO missed
        ref_floor = plan.rooms[member_idx[0]].floor_idx if member_idx else 0
        ref_label = plan.rooms[member_idx[0]].floor_label if member_idx else "Ground Floor"
        stair_ras = np.zeros((_RASTER, _RASTER), np.uint8)
        for st_ in plan.stairs:
            stair_ras |= _rasterize_poly(st_.polygon)
        for ci, cell in enumerate(cells):
            if ci in used_cells:
                continue
            cell_area = int(np.count_nonzero(cell))
            frac = cell_area / floor_area
            if frac < 0.02:
                continue  # small leftover: handled later by fill_gaps
            # cells occupied by stairs are the staircase, not a missed room
            st_overlap = int(np.count_nonzero(cv2.bitwise_and(cell, stair_ras)))
            if st_overlap / max(cell_area, 1) > 0.5:
                continue
            poly = mask_to_polygon(cell, epsilon_factor=0.004, normalise=True)
            if poly and len(poly) >= 3:
                room = RoomPolygon(
                    polygon=snap_to_grid(orthogonalize_polygon(poly, 15.0), 0.004),
                    label="", confidence=0.0,
                    floor_idx=ref_floor, floor_label=ref_label,
                )
                xs = [p[0] for p in room.polygon]
                ys = [p[1] for p in room.polygon]
                room.bbox = (min(xs), min(ys), max(xs), max(ys))
                plan.rooms.append(room)
                plan.quality.review_flags.append(
                    f"unlabeled_room_candidate_{frac:.1%}")
                plan.quality.needs_review = True
    return plan
