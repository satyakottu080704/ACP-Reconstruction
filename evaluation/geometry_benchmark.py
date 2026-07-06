"""
geometry_benchmark.py — geometry IoU benchmark + regression gate.

Compares predicted PlanModel polygons against ground-truth annotations.

Metrics:
  Room IoU    — per-room polygon IoU, Hungarian-matched
  Polygon IoU — same as room IoU (all room-class polygons)
  Gate        — pass/fail: mean IoU must meet a minimum threshold

Ground truth format (JSON):
{
  "rooms": [
    {"polygon": [[x,y],...], "label": "Kitchen", "room_type": "clear"},
    ...
  ]
}
All coordinates normalised [0,1].

Usage:
  from evaluation.geometry_benchmark import run_benchmark, gate_check
  result = run_benchmark(pred_plan, gt_json_path)
  passed = gate_check(result, min_room_iou=0.7)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# optional deps resolved ONCE at module load — never imported inside functions
try:
    from shapely.geometry import Polygon as _SPoly
    _SHAPELY: bool = True
except ImportError:
    _SPoly = None  # type: ignore[assignment,misc]
    _SHAPELY = False

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _SCIPY: bool = True
except ImportError:
    _scipy_lsa = None  # type: ignore[assignment]
    _SCIPY = False


# IoU helpers

def polygon_iou(poly_a: list, poly_b: list) -> float:
    """Compute IoU for two normalised polygons. Uses Shapely if available."""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    if _SHAPELY and _SPoly is not None:
        try:
            a = _SPoly(poly_a)
            b = _SPoly(poly_b)
            if not a.is_valid:
                a = a.buffer(0)
            if not b.is_valid:
                b = b.buffer(0)
            inter = a.intersection(b).area
            union = a.union(b).area
            return float(inter / union) if union > 0 else 0.0
        except Exception:
            pass
    return _raster_iou(poly_a, poly_b, size=256)


def _greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy O(n^2) assignment — no scipy required."""
    used_rows: set = set()
    used_cols: set = set()
    rows: list = []
    cols: list = []
    indices = np.argsort(cost.ravel())
    for idx in indices:
        r, c = divmod(int(idx), cost.shape[1])
        if r not in used_rows and c not in used_cols:
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def _raster_iou(poly_a: list, poly_b: list, size: int = 256) -> float:
    import cv2
    mask_a = _rasterise(poly_a, size)
    mask_b = _rasterise(poly_b, size)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def _rasterise(poly: list, size: int) -> np.ndarray:
    import cv2
    pts = (np.array(poly) * size).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def hausdorff_distance(poly_a: list, poly_b: list) -> float:
    """Hausdorff distance between two normalised polygon boundaries."""
    if not poly_a or not poly_b:
        return 1.0
    if _SHAPELY and _SPoly is not None:
        try:
            a = _SPoly(poly_a)
            b = _SPoly(poly_b)
            return float(a.hausdorff_distance(b))
        except Exception:
            pass

    def _max_min(pts1: list, pts2: list) -> float:
        p1 = np.array(pts1)
        p2 = np.array(pts2)
        dists = np.sqrt(((p1[:, None] - p2[None, :]) ** 2).sum(-1))
        return float(dists.min(axis=1).max())

    return max(_max_min(poly_a, poly_b), _max_min(poly_b, poly_a))


# matching

def match_polygons(
    pred_polys: List[list],
    gt_polys: List[list],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian-match predicted polygons to ground-truth polygons.

    scipy is resolved at module load (_SCIPY/_scipy_lsa), NOT inside this
    function — avoids stale-.pyc issues with inline imports.

    Returns:
        matched:        list of (pred_idx, gt_idx) pairs above iou_threshold
        unmatched_pred: pred indices with no match
        unmatched_gt:   gt indices with no match
    """
    if not pred_polys or not gt_polys:
        return [], list(range(len(pred_polys))), list(range(len(gt_polys)))

    n_pred = len(pred_polys)
    n_gt = len(gt_polys)
    cost = np.zeros((n_pred, n_gt))
    for i, pp in enumerate(pred_polys):
        for j, gp in enumerate(gt_polys):
            cost[i, j] = 1.0 - polygon_iou(pp, gp)

    if _SCIPY and _scipy_lsa is not None:
        row_ind, col_ind = _scipy_lsa(cost)
    else:
        row_ind, col_ind = _greedy_assignment(cost)

    matched: list = []
    matched_pred: set = set()
    matched_gt: set = set()
    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost[r, c]
        if iou >= iou_threshold:
            matched.append((int(r), int(c)))
            matched_pred.add(int(r))
            matched_gt.add(int(c))

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]
    return matched, unmatched_pred, unmatched_gt


# benchmark

def run_benchmark(
    pred_plan: Any,
    gt_path: str,
    iou_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Compare a predicted PlanModel against a ground-truth JSON file."""
    gt_p = Path(gt_path)
    if not gt_p.exists():
        return {"error": f"Ground truth file not found: {gt_p}"}

    with gt_p.open(encoding="utf-8") as f:
        gt = json.load(f)

    gt_rooms = gt.get("rooms", [])
    pred_rooms = pred_plan.rooms

    pred_polys = [r.polygon for r in pred_rooms if len(r.polygon) >= 3]
    gt_polys = [
        r.get("polygon", []) for r in gt_rooms
        if len(r.get("polygon", [])) >= 3
    ]

    matched, unm_pred, unm_gt = match_polygons(pred_polys, gt_polys, iou_threshold)

    per_room_iou: list = []
    per_room_hd: list = []
    for pi, gi in matched:
        iou = polygon_iou(pred_polys[pi], gt_polys[gi])
        hd = hausdorff_distance(pred_polys[pi], gt_polys[gi])
        per_room_iou.append(iou)
        per_room_hd.append(hd)

    mean_iou = float(np.mean(per_room_iou)) if per_room_iou else 0.0
    mean_hd = float(np.mean(per_room_hd)) if per_room_hd else 1.0

    tp = len(matched)
    prec = tp / len(pred_polys) if pred_polys else 0.0
    rec = tp / len(gt_polys) if gt_polys else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "room_iou_mean": round(mean_iou, 4),
        "room_iou_per_room": [round(v, 4) for v in per_room_iou],
        "polygon_iou_mean": round(mean_iou, 4),
        "hausdorff_mean": round(mean_hd, 4),
        "matched_count": tp,
        "unmatched_pred": len(unm_pred),
        "unmatched_gt": len(unm_gt),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pred_room_count": len(pred_polys),
        "gt_room_count": len(gt_polys),
    }


def gate_check(result: Dict[str, Any], min_room_iou: float = 0.7) -> bool:
    """Regression gate: True if benchmark result meets minimum IoU."""
    if "error" in result:
        return False
    return result.get("room_iou_mean", 0.0) >= min_room_iou
