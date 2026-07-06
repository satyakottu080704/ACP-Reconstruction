"""
room_graph.py — room adjacency and door assignment.

Builds adjacency graph from shared walls / touching polygons using a plain
dict-based adjacency list (dependency-light; no NetworkX needed).

Public API:
  build_adjacency(plan)         — populate plan.adjacency in-place
  assign_doors_to_rooms(plan)   — set door.room_a / room_b
  to_room_graph_json(plan)      — {"nodes": [...], "edges": [...]}
  to_connected_to_json(plan)    — {"Living Room": {"connected_to": [...]}, ...}
"""
from __future__ import annotations

from typing import List, Dict, Any

from .plan_model import PlanModel, RoomPolygon

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


# ─── adjacency ────────────────────────────────────────────────────────────────

def build_adjacency(
    plan: PlanModel,
    touch_distance: float = 0.02,
) -> PlanModel:
    """
    Populate plan.adjacency: {room_idx: [adjacent_room_idx, ...]}

    Two rooms are adjacent if:
      a) their polygons share an edge (vertex within touch_distance), OR
      b) (Shapely) their polygons touch or intersect.
    """
    n = len(plan.rooms)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if _rooms_adjacent(plan.rooms[i], plan.rooms[j], touch_distance):
                if j not in adj[i]:
                    adj[i].append(j)
                if i not in adj[j]:
                    adj[j].append(i)

    plan.adjacency = adj
    return plan


def _rooms_adjacent(
    a: RoomPolygon,
    b: RoomPolygon,
    touch_distance: float,
) -> bool:
    """Check if two rooms are adjacent (share an edge or touch)."""
    if _SHAPELY and len(a.polygon) >= 3 and len(b.polygon) >= 3:
        try:
            pa = ShapelyPolygon(a.polygon).buffer(touch_distance * 0.5)
            pb = ShapelyPolygon(b.polygon).buffer(touch_distance * 0.5)
            return pa.intersects(pb)
        except Exception:
            pass

    # Fallback: check if any vertex of a is within touch_distance of any edge of b
    for px, py in a.polygon:
        for k in range(len(b.polygon)):
            ex1, ey1 = b.polygon[k]
            ex2, ey2 = b.polygon[(k + 1) % len(b.polygon)]
            d = _point_to_segment_dist(px, py, ex1, ey1, ex2, ey2)
            if d <= touch_distance:
                return True
    return False


def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nx_, ny_ = ax + t * dx, ay + t * dy
    return ((px - nx_) ** 2 + (py - ny_) ** 2) ** 0.5


# ─── door assignment ──────────────────────────────────────────────────────────

def assign_doors_to_rooms(
    plan: PlanModel,
    max_dist: float = 0.06,
) -> PlanModel:
    """
    Assign each door to the two nearest rooms (room_a, room_b).
    A door on a shared wall connects the two rooms it lies between.
    """
    for door in plan.doors:
        cx, cy = door.center
        dists = []
        for i, room in enumerate(plan.rooms):
            if len(room.polygon) >= 3:
                d = _point_to_polygon_dist(cx, cy, room.polygon)
                dists.append((d, i))
        dists.sort()
        if dists:
            door.room_a = dists[0][1]
        if len(dists) > 1 and dists[1][0] <= max_dist:
            door.room_b = dists[1][1]

    return plan


def _point_to_polygon_dist(px: float, py: float, polygon) -> float:
    """Minimum distance from point to polygon boundary."""
    min_d = float("inf")
    n = len(polygon)
    for k in range(n):
        ax, ay = polygon[k]
        bx, by = polygon[(k + 1) % n]
        d = _point_to_segment_dist(px, py, ax, ay, bx, by)
        min_d = min(min_d, d)
    return min_d


# ─── export helpers ───────────────────────────────────────────────────────────

def to_room_graph_json(plan: PlanModel) -> Dict[str, Any]:
    """
    Return a graph dict:
    {
        "nodes": [{"id": 0, "label": "Kitchen", "floor": 0, "type": "clear"}, ...],
        "edges": [{"source": 0, "target": 1, "via": "door"}, ...]
    }
    """
    nodes = [
        {
            "id": i,
            "label": r.label or f"Room {i}",
            "floor": r.floor_idx,
            "type": r.room_type,
            "is_acm": r.is_acm,
            "is_loft": r.is_loft,
            "centroid": list(r.centroid()),
        }
        for i, r in enumerate(plan.rooms)
    ]

    # Edges from adjacency + door assignments
    edge_set = set()
    edges = []

    # From adjacency graph
    for i, neighbours in plan.adjacency.items():
        for j in neighbours:
            key = (min(i, j), max(i, j))
            if key not in edge_set:
                edge_set.add(key)
                edges.append({"source": key[0], "target": key[1], "via": "wall"})

    # Upgrade wall edges to "door" where a door connects the same pair
    door_pairs = set()
    for d in plan.doors:
        if d.room_a >= 0 and d.room_b >= 0:
            door_pairs.add((min(d.room_a, d.room_b), max(d.room_a, d.room_b)))
    for e in edges:
        key = (min(e["source"], e["target"]), max(e["source"], e["target"]))
        if key in door_pairs:
            e["via"] = "door"

    return {"nodes": nodes, "edges": edges}


def to_connected_to_json(plan: PlanModel) -> Dict[str, Any]:
    """
    Return human-readable connectivity:
    {
        "Living Room": {"connected_to": ["Kitchen", "Hallway"]},
        ...
    }
    Rooms without a label get "Room <idx>".
    Uses door-based adjacency first; falls back to wall adjacency.
    """
    def label(i: int) -> str:
        if 0 <= i < len(plan.rooms):
            return plan.rooms[i].label or f"Room {i}"
        return f"Room {i}"

    # Build door-based connections first
    door_adj: Dict[int, List[int]] = {i: [] for i in range(len(plan.rooms))}
    for d in plan.doors:
        if d.room_a >= 0 and d.room_b >= 0:
            if d.room_b not in door_adj[d.room_a]:
                door_adj[d.room_a].append(d.room_b)
            if d.room_a not in door_adj[d.room_b]:
                door_adj[d.room_b].append(d.room_a)

    # Merge with wall adjacency
    merged: Dict[int, List[int]] = {}
    for i in range(len(plan.rooms)):
        via_door = door_adj.get(i, [])
        via_wall = plan.adjacency.get(i, [])
        combined = list({j for j in via_door + via_wall if j != i})
        merged[i] = combined

    result = {}
    for i, neighbours in merged.items():
        result[label(i)] = {"connected_to": [label(j) for j in neighbours]}

    # Store on plan too
    plan.connectivity = result
    return result
