"""
cleanup.py -- post-polygonisation geometry cleanup.

snap_plan        -- cluster near-coincident edge coords to produce shared walls
resolve_overlaps -- collapse thin slivers to shared walls; keep large overlaps
partition_loft   -- split loft/attic rooms from ground/first-floor rooms
"""
from __future__ import annotations

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


def snap_plan(plan, snap_distance=0.015):
    """Cluster near-coincident vertices to produce shared wall edges."""
    if not plan.rooms:
        return plan
    all_pts = []
    for room in plan.rooms:
        all_pts.extend(room.polygon)
    clusters = []
    cluster_id = {}
    for i, pt in enumerate(all_pts):
        assigned = -1
        for ci, cluster in enumerate(clusters):
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            if _dist(pt, (cx, cy)) <= snap_distance:
                assigned = ci
                break
        if assigned == -1:
            assigned = len(clusters)
            clusters.append([])
        clusters[assigned].append(pt)
        cluster_id[i] = assigned
    centroids = [
        (sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c))
        for c in clusters
    ]
    offset = 0
    for room in plan.rooms:
        n = len(room.polygon)
        new_poly = [centroids[cluster_id[offset + j]] for j in range(n)]
        room.polygon = _dedupe_poly(new_poly)
        offset += n
    return plan


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _dedupe_poly(poly):
    if not poly:
        return poly
    out = [poly[0]]
    for pt in poly[1:]:
        if _dist(pt, out[-1]) > 1e-6:
            out.append(pt)
    if len(out) > 1 and _dist(out[0], out[-1]) < 1e-6:
        out = out[:-1]
    return out if len(out) >= 3 else poly


def resolve_overlaps(plan, sliver_threshold=0.005):
    """Collapse thin overlap slivers between adjacent rooms."""
    if not _SHAPELY or len(plan.rooms) < 2:
        return plan
    rooms = plan.rooms
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if len(rooms[i].polygon) < 3 or len(rooms[j].polygon) < 3:
                continue
            try:
                pa = ShapelyPolygon(rooms[i].polygon)
                pb = ShapelyPolygon(rooms[j].polygon)
                if not pa.is_valid:
                    pa = pa.buffer(0)
                if not pb.is_valid:
                    pb = pb.buffer(0)
                intersection = pa.intersection(pb)
                if intersection.is_empty:
                    continue
                if intersection.area < sliver_threshold:
                    if pa.area <= pb.area:
                        pa = pa.difference(intersection)
                        if not pa.is_empty and pa.geom_type == "Polygon":
                            rooms[i].polygon = list(pa.exterior.coords[:-1])
                    else:
                        pb = pb.difference(intersection)
                        if not pb.is_empty and pb.geom_type == "Polygon":
                            rooms[j].polygon = list(pb.exterior.coords[:-1])
            except Exception:
                continue
    return plan


_LOFT_KEYWORDS = frozenset({
    "loft", "attic", "roof", "mezzanine", "eaves", "rafters",
    "goz loft", "goz", "roof space",
})


def partition_loft(plan):
    """
    Identify loft/attic rooms and assign them floor_idx = max+1.

    Detection (in priority order):
      1. Label contains a loft keyword.
      2. room.is_loft already True.
      3. ACM room whose centroid is > 0.20 from the main floor centroid
         (separate loft sketch drawn on the same survey page).
      4. Tiny room (area < 40pct avg) with centroid_y < 0.12.
    """
    if not plan.rooms:
        return plan

    max_floor = max(r.floor_idx for r in plan.rooms)
    loft_floor = max_floor + 1
    avg_area = sum(r.area for r in plan.rooms) / len(plan.rooms)

    non_acm = [r for r in plan.rooms if not r.is_acm]
    if non_acm:
        main_cx = sum(r.centroid()[0] for r in non_acm) / len(non_acm)
        main_cy = sum(r.centroid()[1] for r in non_acm) / len(non_acm)
    else:
        main_cx, main_cy = 0.5, 0.5

    for room in plan.rooms:
        label_lower = room.label.lower()
        is_loft = room.is_loft or any(kw in label_lower for kw in _LOFT_KEYWORDS)

        if not is_loft and room.is_acm and non_acm:
            # Only meaningful when clear rooms exist to define the main
            # floor; if EVERY room is ACM (whole-property asbestos), the
            # default centroid would falsely flag outlying rooms as loft.
            cx, cy = room.centroid()
            dist = ((cx - main_cx) ** 2 + (cy - main_cy) ** 2) ** 0.5
            if dist > 0.20:
                is_loft = True

        if not is_loft and room.area > 0 and room.area < avg_area * 0.4:
            cx, cy = room.centroid()
            if cy < 0.12:
                is_loft = True

        if is_loft:
            room.is_loft = True
            room.room_type = "loft"
            if room.floor_idx == max_floor:
                room.floor_idx = loft_floor
                room.floor_label = "Loft"

    plan.has_loft = any(r.is_loft for r in plan.rooms)

    while len(plan.floor_labels) <= loft_floor:
        plan.floor_labels.append(f"Floor {len(plan.floor_labels)}")
    if plan.has_loft:
        plan.floor_labels[loft_floor] = "Loft"

    return plan
