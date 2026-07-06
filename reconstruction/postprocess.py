"""
postprocess.py — geometry post-processing for the v2 reconstruction engine.

Implements the "hybrid pipeline" post-processing stages:
  - snap_to_grid / orthogonalize (90-degree enforcement on near-axis edges)
  - attach_doors_to_walls (doors must lie ON wall edges; swing oriented
    from the wall direction, opening into the room)
  - orient_stairs (direction arrow follows the stair's long axis)
  - apply_labels (merge hook: GPT-4o room labels / numbers / floors /
    no-access flags + red-pen samples into a PlanModel)

All functions are OpenCV/NumPy-free (pure python + math) and normalised
[0,1] coordinate space, so they run anywhere.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .plan_model import PlanModel, SampleAnnotation


# ── orthogonalization / grid snap ─────────────────────────────────────────────

def snap_to_grid(poly: Sequence[Tuple[float, float]], grid: float = 0.004) -> List[Tuple[float, float]]:
    """Snap polygon vertices to a regular grid (normalised units)."""
    if grid <= 0:
        return list(poly)
    snapped = [(round(x / grid) * grid, round(y / grid) * grid) for x, y in poly]
    return _dedupe(snapped)


def orthogonalize_polygon(
    poly: Sequence[Tuple[float, float]],
    angle_tol_deg: float = 15.0,
) -> List[Tuple[float, float]]:
    """
    Enforce right angles: edges within angle_tol_deg of horizontal/vertical
    are snapped to exactly horizontal/vertical.  Diagonal edges (beyond the
    tolerance) are left untouched, so genuinely angled walls survive.
    """
    pts = [list(p) for p in poly]
    n = len(pts)
    if n < 3:
        return [tuple(p) for p in pts]

    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx == 0 and dy == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        if ang < angle_tol_deg or ang > 180.0 - angle_tol_deg:      # ~horizontal
            ym = (a[1] + b[1]) / 2.0
            a[1] = b[1] = ym
        elif abs(ang - 90.0) < angle_tol_deg:                        # ~vertical
            xm = (a[0] + b[0]) / 2.0
            a[0] = b[0] = xm

    return _remove_collinear(_dedupe([tuple(p) for p in pts]))


def orthogonalize_plan(plan: PlanModel, angle_tol_deg: float = 15.0, grid: float = 0.004) -> PlanModel:
    """Orthogonalize + grid-snap all room and stair polygons."""
    for room in plan.rooms:
        if len(room.polygon) >= 3:
            room.polygon = snap_to_grid(
                orthogonalize_polygon(room.polygon, angle_tol_deg), grid)
    for stair in plan.stairs:
        if len(stair.polygon) >= 3:
            stair.polygon = snap_to_grid(
                orthogonalize_polygon(stair.polygon, angle_tol_deg), grid)
    return plan


def _dedupe(poly: List[Tuple[float, float]], eps: float = 1e-9) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in poly:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _remove_collinear(poly: List[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    if len(poly) < 4:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > eps:
            out.append(b)
    return out if len(out) >= 3 else poly


# ── shared walls: plan-wide coordinate clustering ─────────────────────────────

def share_walls(plan: PlanModel, tol: float = 0.008) -> PlanModel:
    """
    Force adjacent rooms to share EXACT wall coordinates (diagram rule:
    "Bedroom and Cupboard share the same wall with no gap").

    After orthogonalization, vertices sit on axis-parallel lines.  Cluster
    all vertex x-coordinates (and, independently, y-coordinates) per floor
    with a 1-D sweep; replace each coordinate with its cluster mean.  Two
    rooms drawn with a thin sliver between them collapse onto one shared
    wall line; genuinely separate walls (further apart than tol) survive.
    """
    floors = sorted({r.floor_idx for r in plan.rooms})
    for fi in floors:
        rooms = [r for r in plan.rooms if r.floor_idx == fi]
        stairs = [st for st in plan.stairs if getattr(st, "floor_idx", 0) == fi]
        xs, ys = [], []
        for r in rooms:
            xs += [p[0] for p in r.polygon]
            ys += [p[1] for p in r.polygon]
        # floor-boundary coordinates act as anchors so room edges lock
        # onto the floor outline exactly
        for bpoly in getattr(plan, "floor_boundary", []) or []:
            xs += [p[0] for p in bpoly]
            ys += [p[1] for p in bpoly]
        if not xs:
            continue
        xmap = _cluster_1d(xs, tol)
        ymap = _cluster_1d(ys, tol)

        def snap_pt(p):
            return (xmap.get(round(p[0], 9), p[0]), ymap.get(round(p[1], 9), p[1]))

        for r in rooms:
            r.polygon = _dedupe([snap_pt(p) for p in r.polygon])
            px = [p[0] for p in r.polygon]
            py = [p[1] for p in r.polygon]
            if px:
                r.bbox = (min(px), min(py), max(px), max(py))
        for st in stairs:
            st.polygon = _dedupe([snap_pt(p) for p in st.polygon])
    return plan


def _cluster_1d(values, tol):
    """Greedy 1-D clustering: sorted sweep, split when gap > tol.
    Returns {rounded original value -> cluster mean}."""
    uniq = sorted(set(round(v, 9) for v in values))
    mapping = {}
    cluster = [uniq[0]] if uniq else []
    for v in uniq[1:]:
        if v - cluster[-1] <= tol:
            cluster.append(v)
        else:
            mean = sum(cluster) / len(cluster)
            for c in cluster:
                mapping[c] = mean
            cluster = [v]
    if cluster:
        mean = sum(cluster) / len(cluster)
        for c in cluster:
            mapping[c] = mean
    return mapping


# ── doors: snap onto wall edges + orient the swing ────────────────────────────

def attach_doors_to_walls(plan: PlanModel, max_dist: float = 0.05) -> PlanModel:
    """
    Doors must lie ON walls: project each door centre onto the nearest room
    edge, set the swing angle from the wall direction and open the arc into
    the room the door belongs to.  Doors too far from any wall are dropped
    (no floating doors).
    """
    kept = []
    for door in plan.doors:
        hit = _nearest_edge(plan, door.center, max_dist)
        if hit is None:
            continue  # floating door — reject
        room_idx, proj, edge_ang, edge_len = hit
        door.center = proj
        # door opening width: keep detected width but never wider than the wall edge
        door.width = min(max(door.width, 0.02), edge_len * 0.9)
        if door.room_a < 0:
            door.room_a = room_idx
        door.angle_deg = _swing_angle(plan, door, room_idx, edge_ang)
        kept.append(door)
    plan.doors = kept
    return plan


def _nearest_edge(plan, pt, max_dist):
    best = None
    for i, room in enumerate(plan.rooms):
        poly = room.polygon
        n = len(poly)
        if n < 3:
            continue
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            proj, d, t = _project(pt, a, b)
            if d <= max_dist and (best is None or d < best[0]):
                ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                elen = math.hypot(b[0] - a[0], b[1] - a[1])
                best = (d, i, proj, ang, elen)
    if best is None:
        return None
    _, i, proj, ang, elen = best
    return i, proj, ang, elen


def _project(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll == 0:
        return a, math.hypot(p[0] - ax, p[1] - ay), 0.0
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / ll))
    proj = (ax + t * dx, ay + t * dy)
    return proj, math.hypot(p[0] - proj[0], p[1] - proj[1]), t


def _swing_angle(plan, door, room_idx, edge_ang):
    """Pick the 90-degree arc [a0, a0+90] whose middle points into the room."""
    room = plan.rooms[room_idx]
    cx, cy = room.centroid()
    phi = math.degrees(math.atan2(cy - door.center[1], cx - door.center[0]))
    best_a0, best_diff = edge_ang, 360.0
    for a0 in (edge_ang, edge_ang + 90, edge_ang + 180, edge_ang + 270):
        mid = (a0 + 45.0) % 360.0
        diff = abs((mid - phi + 180.0) % 360.0 - 180.0)
        if diff < best_diff:
            best_a0, best_diff = a0 % 360.0, diff
    return best_a0


# ── stairs: orient the UP arrow along the flight ──────────────────────────────

def orient_stairs(plan: PlanModel) -> PlanModel:
    """Direction arrow follows the stair polygon's long axis (treads are
    rendered perpendicular to it)."""
    for stair in plan.stairs:
        if len(stair.polygon) < 3:
            continue
        xs = [p[0] for p in stair.polygon]
        ys = [p[1] for p in stair.polygon]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        stair.direction_deg = 180.0 if w >= h else 270.0
    return plan


# ── GPT-4o merge hook ─────────────────────────────────────────────────────────

def apply_labels(
    plan: PlanModel,
    rooms_info: Optional[List[Dict[str, Any]]] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    floor_names: Optional[Dict[int, str]] = None,
) -> PlanModel:
    """
    Merge GPT-4o vision output into a PlanModel (the v2 counterpart of the
    legacy pipeline's merge step).

    rooms_info: [{"name": "Kitchen", "number": "003", "has_acm": false,
                  "no_access": false, "floor": 0,
                  "x_pct": 42.0, "y_pct": 61.0}, ...]
        Matched to detected rooms by nearest centroid when x_pct/y_pct are
        given, else by list order.

    samples: [{"id": "S001", "material": "TC", "acm_positive": true,
               "is_ref": false, "target_room_number": "003",
               "x_pct": 10.0, "y_pct": 20.0, "floor": 0}, ...]

    floor_names: {0: "Ground Floor", 1: "First Floor", 2: "Roof Space"}
    """
    floor_names = dict(floor_names or {})

    if rooms_info:
        used = set()
        for order, info in enumerate(rooms_info):
            idx = _match_room(plan, info, order, used)
            if idx is None:
                continue
            used.add(idx)
            room = plan.rooms[idx]
            if info.get("name"):
                room.label = str(info["name"])
            if info.get("number"):
                room.number = str(info["number"])
            if info.get("no_access"):
                room.no_access = True
                room.room_type = "no_access"
            if info.get("has_acm"):
                room.is_acm = True
                if room.room_type == "clear":
                    room.room_type = "acm"
            fi = info.get("floor")
            if fi is not None:
                room.floor_idx = int(fi)
            if "loft" in (room.label or "").lower():
                room.is_loft = True
                room.room_type = "loft" if room.room_type == "clear" else room.room_type
            elif fi is not None and room.is_loft:
                # GPT gave an explicit floor and a non-loft name: undo a
                # wrong loft guess from the geometry heuristics.
                room.is_loft = False
                room.room_type = "acm" if room.is_acm else "clear"
            room.floor_label = floor_names.get(room.floor_idx, room.floor_label)

    if samples:
        num_to_idx = {}
        for i, r in enumerate(plan.rooms):
            if r.number:
                num_to_idx.setdefault((r.number.lstrip("0") or "0", r.floor_idx), i)
        for s in samples:
            fi = int(s.get("floor", 0) or 0)
            room_idx = -1
            tgt = s.get("target_room_number")
            if tgt:
                room_idx = num_to_idx.get((str(tgt).lstrip("0") or "0", fi), -1)
            if room_idx < 0 and s.get("x_pct") is not None:
                room_idx = _room_at(plan, float(s["x_pct"]) / 100.0,
                                    float(s.get("y_pct", 0)) / 100.0, fi)
            if room_idx >= 0:
                target = plan.rooms[room_idx].centroid()
                fi = plan.rooms[room_idx].floor_idx
            else:
                target = (float(s.get("x_pct", 0)) / 100.0,
                          float(s.get("y_pct", 0)) / 100.0)
            plan.samples.append(SampleAnnotation(
                sample_id=str(s.get("id", "") or ""),
                material=str(s.get("material", "") or ""),
                is_ref=bool(s.get("is_ref", False)),
                acm_positive=bool(s.get("acm_positive", False)),
                target=target,
                room_idx=room_idx,
                floor_idx=fi,
            ))

    # Re-home stairs and hatch symbols to the floor of the room that
    # contains them (rooms may have just moved to explicit GPT floors)
    for st_ in plan.stairs:
        if len(st_.polygon) < 3:
            continue
        sx = sum(p[0] for p in st_.polygon) / len(st_.polygon)
        sy = sum(p[1] for p in st_.polygon) / len(st_.polygon)
        best_i, best_d = -1, float("inf")
        for i, r in enumerate(plan.rooms):
            if _point_in_poly(sx, sy, r.polygon):
                best_i, best_d = i, 0.0
                break
            cx, cy = r.centroid()
            d = (cx - sx) ** 2 + (cy - sy) ** 2
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            st_.floor_idx = plan.rooms[best_i].floor_idx

    # Refresh floor labels list
    floors = sorted({r.floor_idx for r in plan.rooms}) or [0]
    plan.floor_labels = [
        floor_names.get(fi) or next(
            (r.floor_label for r in plan.rooms if r.floor_idx == fi), f"Floor {fi}")
        for fi in floors
    ]
    plan.has_loft = any(r.is_loft for r in plan.rooms)
    return plan


def _match_room(plan, info, order, used):
    x = info.get("x_pct")
    y = info.get("y_pct")
    if x is not None and y is not None:
        px, py = float(x) / 100.0, float(y) / 100.0
        best, best_d = None, float("inf")
        for i, r in enumerate(plan.rooms):
            if i in used:
                continue
            cx, cy = r.centroid()
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best, best_d = i, d
        return best
    # fallback: list order (same index if free, else first unused)
    if order < len(plan.rooms) and order not in used:
        return order
    for i in range(len(plan.rooms)):
        if i not in used:
            return i
    return None


def _room_at(plan, x, y, floor_idx):
    """Room whose polygon contains (x, y), preferring the given floor."""
    hit = -1
    for i, r in enumerate(plan.rooms):
        if _point_in_poly(x, y, r.polygon):
            if r.floor_idx == floor_idx:
                return i
            hit = i
    return hit


def _point_in_poly(x, y, poly) -> bool:
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside
