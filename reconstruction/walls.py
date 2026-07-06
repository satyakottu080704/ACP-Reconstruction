"""
walls.py — wall mask → centerline extraction (diagram stages 3-4:
"Wall Processing: mask → centerline" and "Line Merging & Cleaning").

Steps:
  1. refine the union wall raster (close pen gaps, remove specks)
  2. thickness field via distance transform
  3. skeletonize (cv2.ximgproc.thinning → scikit-image → morphological fallback)
  4. trace the skeleton pixel graph into chains (junction-to-junction)
  5. Douglas-Peucker each chain → line segments
  6. merge near-collinear segments; snap near-axis segments to 0/90 degrees

Output: List[WallSegment] with normalised endpoints + thickness.
Core deps only (OpenCV + NumPy); optional accelerators used when present.
"""
from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np
import cv2

from .plan_model import WallSegment


def extract_wall_centerlines(
    wall_mask: np.ndarray,
    image_size: Tuple[int, int],
    angle_tol_deg: float = 12.0,
    min_seg_px: float = 8.0,
) -> List[WallSegment]:
    """
    Convert a (H, W) uint8 union-of-wall-masks raster into merged, snapped
    centerline WallSegments (normalised coordinates).
    """
    w, h = image_size
    if wall_mask is None or wall_mask.size == 0 or wall_mask.max() == 0:
        return []
    m = (wall_mask > 0).astype(np.uint8) * 255

    # 1. refine
    thickness_px = _estimate_thickness(m)
    k = max(3, int(round(thickness_px)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 2/3. skeletonize
    skel = _skeletonize(m)

    # 4. trace chains
    chains = _trace_chains(skel)

    # 5. simplify chains -> segments
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        arr = np.array(chain, dtype=np.int32).reshape(-1, 1, 2)
        eps = max(1.5, thickness_px * 0.4)
        approx = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2)
        for i in range(len(approx) - 1):
            p1, p2 = tuple(approx[i]), tuple(approx[i + 1])
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= min_seg_px:
                segments.append((p1, p2))

    # normalise
    nsegs = [((x1 / w, y1 / h), (x2 / w, y2 / h)) for (x1, y1), (x2, y2) in segments]
    t_norm = thickness_px / max(w, h)

    # 6. snap angles, then merge collinear
    nsegs = [_snap_axis(s, angle_tol_deg) for s in nsegs]
    nsegs = _merge_collinear(nsegs, angle_tol_deg=7.0,
                             gap_tol=3.0 * t_norm, offset_tol=max(t_norm, 0.004))

    return [WallSegment(points=[a, b], thickness=t_norm) for a, b in nsegs]


# ── helpers ───────────────────────────────────────────────────────────────────

def _estimate_thickness(m: np.ndarray) -> float:
    dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[m > 0]
    if vals.size == 0:
        return 3.0
    return max(2.0, 2.0 * float(np.median(vals)))


def _skeletonize(m: np.ndarray) -> np.ndarray:
    binary = (m > 0).astype(np.uint8)
    try:
        return cv2.ximgproc.thinning(binary * 255) > 0
    except Exception:
        pass
    try:
        from skimage.morphology import skeletonize as sk_skel
        return sk_skel(binary.astype(bool))
    except Exception:
        pass
    # morphological skeleton fallback (coarser but dependency-free)
    skel = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def _trace_chains(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Trace the 1-px skeleton into junction-to-junction pixel chains."""
    ys, xs = np.nonzero(skel)
    pts: Set[Tuple[int, int]] = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return []

    def nbrs(p):
        x, y = p
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx or dy) and (x + dx, y + dy) in pts:
                    out.append((x + dx, y + dy))
        return out

    deg: Dict[Tuple[int, int], int] = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p, d in deg.items() if d != 2}
    chains: List[List[Tuple[int, int]]] = []
    visited: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def walk(start, nxt):
        chain = [start, nxt]
        visited.add((start, nxt))
        visited.add((nxt, start))
        prev, cur = start, nxt
        while cur not in nodes:
            options = [q for q in nbrs(cur) if q != prev and (cur, q) not in visited]
            if not options:
                break
            q = options[0]
            visited.add((cur, q))
            visited.add((q, cur))
            chain.append(q)
            prev, cur = cur, q
        return chain

    for n in nodes:
        for nb in nbrs(n):
            if (n, nb) not in visited:
                chains.append(walk(n, nb))

    # pure loops (no junction nodes at all)
    if not nodes and pts:
        start = next(iter(pts))
        nb = nbrs(start)
        if nb:
            chains.append(walk(start, nb[0]))
    return chains


def _snap_axis(seg, tol_deg):
    (x1, y1), (x2, y2) = seg
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    if ang < tol_deg or ang > 180.0 - tol_deg:          # horizontal
        ym = (y1 + y2) / 2.0
        return ((x1, ym), (x2, ym))
    if abs(ang - 90.0) < tol_deg:                        # vertical
        xm = (x1 + x2) / 2.0
        return ((xm, y1), (xm, y2))
    return seg


def _merge_collinear(segs, angle_tol_deg, gap_tol, offset_tol):
    """Iteratively merge near-collinear, near-touching segments."""
    segs = list(segs)
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            a = segs[i]
            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                merged = _try_merge(a, segs[j], angle_tol_deg, gap_tol, offset_tol)
                if merged is not None:
                    a = merged
                    used[j] = True
                    changed = True
            used[i] = True
            out.append(a)
        segs = out
    return segs


def _try_merge(s1, s2, angle_tol_deg, gap_tol, offset_tol):
    a1 = math.degrees(math.atan2(s1[1][1] - s1[0][1], s1[1][0] - s1[0][0])) % 180.0
    a2 = math.degrees(math.atan2(s2[1][1] - s2[0][1], s2[1][0] - s2[0][0])) % 180.0
    diff = min(abs(a1 - a2), 180.0 - abs(a1 - a2))
    if diff > angle_tol_deg:
        return None
    # direction of s1
    dx, dy = s1[1][0] - s1[0][0], s1[1][1] - s1[0][1]
    ll = math.hypot(dx, dy)
    if ll < 1e-9:
        return None
    ux, uy = dx / ll, dy / ll
    ox, oy = s1[0]

    def proj(p):
        return ((p[0] - ox) * ux + (p[1] - oy) * uy,
                -(p[0] - ox) * uy + (p[1] - oy) * ux)   # (along, offset)

    pts = [proj(s1[0]), proj(s1[1]), proj(s2[0]), proj(s2[1])]
    if max(abs(p[1]) for p in pts) > offset_tol:
        return None
    t1 = sorted((pts[0][0], pts[1][0]))
    t2 = sorted((pts[2][0], pts[3][0]))
    if t2[0] - t1[1] > gap_tol or t1[0] - t2[1] > gap_tol:
        return None
    lo, hi = min(t1[0], t2[0]), max(t1[1], t2[1])
    p_lo = (ox + lo * ux, oy + lo * uy)
    p_hi = (ox + hi * ux, oy + hi * uy)
    return (p_lo, p_hi)
