"""
coverage.py — gap detection & filling + constraint validation
(diagram stages 9-10: "Gap Detection & Filling" and "Constraint Solving").

fill_gaps(plan):
    Union(all rooms) vs floor boundary -> detect gap polygons -> assign each
    gap to the adjacent room sharing the longest border -> merge, so the
    rooms completely fill the floor outline (no gaps, full coverage).
    Raster-based (OpenCV) so it needs no Shapely.

validate_constraints(plan):
    Machine checks from the diagram checklist:
      - no gaps / no overlaps
      - all rooms inside the floor boundary
      - doors on walls
      - all angles ~90 degrees
      - connectivity (isolated rooms)
    Results are merged into plan.quality (issues + review flags).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import cv2

from .plan_model import PlanModel

_RASTER = 1024  # working raster size for coverage ops


def _rasterize(polys, size=_RASTER) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    for poly in polys:
        if len(poly) >= 3:
            pts = np.array([(int(x * (size - 1)), int(y * (size - 1)))
                            for x, y in poly], np.int32)
            cv2.fillPoly(m, [pts], 255)
    return m


def fill_gaps(
    plan: PlanModel,
    min_gap_frac: float = 2e-4,
    max_gap_frac: float = 0.04,
) -> PlanModel:
    """
    Fill uncovered space between rooms and the floor boundary by growing
    the adjacent room with the longest shared border into each gap.

    Gaps larger than max_gap_frac of the floor area are NOT auto-filled
    (they are probably an undetected room) — they get a review flag instead.
    """
    if not plan.floor_boundary or not plan.rooms:
        return plan

    floor = _rasterize(plan.floor_boundary)
    floor_area = int(np.count_nonzero(floor))
    if floor_area == 0:
        return plan

    room_masks = [_rasterize([r.polygon]) for r in plan.rooms]
    union = np.zeros_like(floor)
    for rm in room_masks:
        union |= rm
    # stairs legitimately occupy floor area — count them as covered
    for stair in plan.stairs:
        union |= _rasterize([stair.polygon])

    gaps = cv2.bitwise_and(floor, cv2.bitwise_not(union))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)

    changed = set()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for gi in range(1, n):
        area = stats[gi, cv2.CC_STAT_AREA]
        frac = area / floor_area
        if frac < min_gap_frac:
            continue
        comp = (labels == gi).astype(np.uint8) * 255
        if frac > max_gap_frac:
            plan.quality.review_flags.append(
                f"large_uncovered_area_{frac:.1%}_possible_missed_room")
            plan.quality.needs_review = True
            continue
        ring = cv2.dilate(comp, kernel)
        candidates = [ri for ri, rm in enumerate(room_masks)
                      if np.count_nonzero(cv2.bitwise_and(ring, rm)) > 0]
        if not candidates:
            continue
        if len(candidates) == 1:
            # Only one neighbour touches this gap: give it the whole thing.
            # (A small gap that touches 2+ rooms must NOT take this shortcut
            # -- even a low-area component can be a perimeter ring wrapping
            # around several rooms; assigning it wholly to whichever room
            # has the most overlap balloons that room around the others.
            # See the partition branch below, which is always safe.)
            best, best_overlap = candidates[0], 0
            for ri in candidates:
                overlap = int(np.count_nonzero(
                    cv2.bitwise_and(ring, room_masks[ri])))
                if overlap > best_overlap:
                    best, best_overlap = ri, overlap
            room_masks[best] |= comp
            changed.add(best)
        else:
            # A gap touching several rooms (e.g. a perimeter ring) must be
            # PARTITIONED: each gap pixel goes to its nearest adjacent room.
            # Assigning the whole component to one room would balloon that
            # room around the others and create massive overlaps.
            dists = np.stack([
                cv2.distanceTransform(
                    cv2.bitwise_not(room_masks[ri]), cv2.DIST_L2, 3)
                for ri in candidates
            ])
            nearest = np.argmin(dists, axis=0)
            comp_sel = comp > 0
            for ci, ri in enumerate(candidates):
                part = np.zeros_like(comp)
                part[comp_sel & (nearest == ci)] = 255
                if np.count_nonzero(part):
                    room_masks[ri] |= part
                    changed.add(ri)

    # Re-polygonize the grown rooms
    if changed:
        from .polygons import mask_to_polygon
        from .postprocess import orthogonalize_polygon, snap_to_grid
        for ri in changed:
            poly = mask_to_polygon(room_masks[ri], epsilon_factor=0.005, normalise=True)
            if poly and len(poly) >= 3:
                plan.rooms[ri].polygon = snap_to_grid(
                    orthogonalize_polygon(poly, 15.0), 0.004)
                xs = [p[0] for p in plan.rooms[ri].polygon]
                ys = [p[1] for p in plan.rooms[ri].polygon]
                plan.rooms[ri].bbox = (min(xs), min(ys), max(xs), max(ys))
    return plan


def carve_nested_rooms(plan: PlanModel, containment: float = 0.6) -> PlanModel:
    """
    When one room's polygon substantially contains a smaller room (e.g. a
    lounge mask drawn OVER a nested cupboard), carve the smaller room out
    of the larger one.  This is how real plans work: the CPD is cut out of
    the room that surrounds it — rooms never overlap.
    """
    if len(plan.rooms) < 2:
        return plan
    masks = [_rasterize([r.polygon]) for r in plan.rooms]
    areas = [max(int(np.count_nonzero(m)), 1) for m in masks]
    changed = set()
    order = sorted(range(len(plan.rooms)), key=lambda i: -areas[i])
    for big in order:
        for small in order[::-1]:
            if big == small or areas[small] >= areas[big]:
                continue
            if plan.rooms[big].floor_idx != plan.rooms[small].floor_idx:
                continue
            inter = int(np.count_nonzero(cv2.bitwise_and(masks[big], masks[small])))
            if inter / areas[small] >= containment:
                masks[big] = cv2.bitwise_and(masks[big], cv2.bitwise_not(masks[small]))
                changed.add(big)
    if changed:
        from .polygons import mask_to_polygon
        from .postprocess import orthogonalize_polygon, snap_to_grid
        for ri in changed:
            poly = mask_to_polygon(masks[ri], epsilon_factor=0.003, normalise=True)
            if poly and len(poly) >= 3:
                plan.rooms[ri].polygon = snap_to_grid(
                    orthogonalize_polygon(poly, 15.0), 0.004)
                xs = [p[0] for p in plan.rooms[ri].polygon]
                ys = [p[1] for p in plan.rooms[ri].polygon]
                plan.rooms[ri].bbox = (min(xs), min(ys), max(xs), max(ys))
    return plan


def validate_constraints(plan: PlanModel, door_tol: float = 0.02) -> Dict[str, object]:
    """Run the constraint checklist; merge failures into plan.quality."""
    checks: Dict[str, object] = {}

    floor = _rasterize(plan.floor_boundary) if plan.floor_boundary else None
    room_masks = [_rasterize([r.polygon]) for r in plan.rooms]
    union = np.zeros((_RASTER, _RASTER), np.uint8)
    overlap_px = 0
    covered = np.zeros_like(union)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for rm in room_masks:
        # erode 1px: rooms SHARING a wall line is not an overlap
        rm_in = cv2.erode(rm, erode_k)
        overlap_px += int(np.count_nonzero(cv2.bitwise_and(covered, rm_in)))
        covered |= rm
    # stairs count as covered floor area (not as overlap)
    for stair in plan.stairs:
        covered |= _rasterize([stair.polygon])
    union = covered
    union_area = max(int(np.count_nonzero(union)), 1)

    # no overlaps
    checks["overlap_fraction"] = overlap_px / union_area
    checks["no_overlaps"] = checks["overlap_fraction"] < 0.02

    if floor is not None:
        floor_area = max(int(np.count_nonzero(floor)), 1)
        # close 3px: hairline seams along shared walls are not gaps
        union_closed = cv2.morphologyEx(
            union, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        gap_px = int(np.count_nonzero(cv2.bitwise_and(floor, cv2.bitwise_not(union_closed))))
        floor_grown = cv2.dilate(floor, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        outside_px = int(np.count_nonzero(cv2.bitwise_and(union, cv2.bitwise_not(floor_grown))))
        checks["gap_fraction"] = gap_px / floor_area
        checks["no_gaps"] = checks["gap_fraction"] < 0.01
        checks["outside_fraction"] = outside_px / union_area
        checks["rooms_inside_floor"] = checks["outside_fraction"] < 0.02
    else:
        checks["no_gaps"] = None            # no floor boundary detected
        checks["rooms_inside_floor"] = None

    # doors on walls
    from .postprocess import _nearest_edge
    doors_ok = all(
        _nearest_edge(plan, d.center, door_tol) is not None for d in plan.doors
    ) if plan.doors else True
    checks["doors_on_walls"] = doors_ok

    # all angles ~90
    worst = 0.0
    for r in plan.rooms:
        npts = len(r.polygon)
        for i in range(npts):
            a, b = r.polygon[i], r.polygon[(i + 1) % npts]
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90.0
            worst = max(worst, min(ang, 90.0 - ang))
    checks["max_angle_deviation_deg"] = round(worst, 2)
    checks["orthogonal"] = worst <= 3.0

    # connectivity
    connected = {d.room_a for d in plan.doors} | {d.room_b for d in plan.doors}
    isolated = [i for i in range(len(plan.rooms))
                if i not in connected and not plan.adjacency.get(i)]
    checks["isolated_rooms"] = len(isolated)
    checks["connected"] = len(isolated) == 0

    # merge into quality
    q = plan.quality
    fail_map = {
        "no_overlaps": "rooms_overlap",
        "no_gaps": "coverage_gaps",
        "rooms_inside_floor": "rooms_outside_floor",
        "doors_on_walls": "door_off_wall",
        "orthogonal": "non_orthogonal_geometry",
    }
    for key, flag in fail_map.items():
        if checks.get(key) is False:
            q.geometry_issues.append(flag)
            if flag not in q.review_flags:
                q.review_flags.append(flag)
            q.needs_review = True
    return checks
