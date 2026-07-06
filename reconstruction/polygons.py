"""
polygons.py — polygon extraction and wall reconstruction from masks.

Core (OpenCV/NumPy only):
  mask_to_polygon      — contour → Douglas-Peucker simplify → orthogonal snap
  merge_polygons       — union of two mask polygons (via mask rasterisation)
  wall_centerlines     — skeleton + Hough lines from wall mask
  reconstruct_walls    — CLOSE→OPEN→CC→contour→approx pipeline for walls

Shapely optional: used for robust repair/merge when available.
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple

try:
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    from shapely.ops import unary_union
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_factor: float = 0.005,
    min_area_px: int = 50,
    snap_threshold: float = 0.01,
    normalise: bool = True,
) -> List[Tuple[float, float]]:
    """
    Extract the largest-area polygon from a binary mask.

    Pipeline:
      1. Find contours (RETR_EXTERNAL)
      2. Keep largest by area
      3. Douglas-Peucker simplification (epsilon = epsilon_factor * perimeter)
      4. Orthogonal snap (align near-horizontal/vertical edges)
      5. Optionally normalise to [0,1]

    Returns:
        List of (x, y) points (normalised if normalise=True), or [] on failure.
    """
    if mask is None or mask.size == 0:
        return []

    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Keep largest contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_px:
        return []

    # Douglas-Peucker simplification
    peri = cv2.arcLength(c, True)
    eps = epsilon_factor * peri
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).astype(float)

    # Orthogonal snap
    pts = _snap_orthogonal(pts, threshold_deg=15.0)

    # Repair with Shapely if available
    if _SHAPELY and len(pts) >= 3:
        try:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                pts = np.array(poly.exterior.coords[:-1])
        except Exception:
            pass

    if normalise:
        h, w = mask.shape[:2]
        pts[:, 0] /= (w or 1)
        pts[:, 1] /= (h or 1)
        pts = np.clip(pts, 0.0, 1.0)

    return [(float(x), float(y)) for x, y in pts]


def _snap_orthogonal(
    pts: np.ndarray,
    threshold_deg: float = 15.0,
) -> np.ndarray:
    """
    Snap near-horizontal and near-vertical edges to be exactly horizontal/vertical.
    This produces cleaner wall representations for floor plans.
    """
    if len(pts) < 3:
        return pts

    result = pts.copy()
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
        if angle < threshold_deg:
            # Nearly horizontal: equalise Y
            mid_y = (p1[1] + p2[1]) / 2
            result[i][1] = mid_y
            result[(i + 1) % n][1] = mid_y
        elif angle > 90 - threshold_deg:
            # Nearly vertical: equalise X
            mid_x = (p1[0] + p2[0]) / 2
            result[i][0] = mid_x
            result[(i + 1) % n][0] = mid_x
    return result


def merge_polygons(
    poly_a: List[Tuple[float, float]],
    poly_b: List[Tuple[float, float]],
    image_width: int = 1000,
    image_height: int = 1000,
) -> List[Tuple[float, float]]:
    """
    Return the union polygon of two normalised polygons.
    Uses Shapely if available, else falls back to mask-based union.
    """
    if _SHAPELY:
        try:
            a = ShapelyPolygon(poly_a)
            b = ShapelyPolygon(poly_b)
            union = unary_union([a, b])
            if union.is_empty:
                return poly_a
            if isinstance(union, MultiPolygon):
                union = max(union.geoms, key=lambda g: g.area)
            coords = list(union.exterior.coords[:-1])
            return [(float(x), float(y)) for x, y in coords]
        except Exception:
            pass

    # Fallback: rasterise both, OR the masks, extract contour
    from .masks import polygon_to_mask, mask_to_polygon  # local import to avoid circularity
    w, h = image_width, image_height
    m_a = polygon_to_mask(poly_a, w, h, normalised=True)
    m_b = polygon_to_mask(poly_b, w, h, normalised=True)
    union_mask = cv2.bitwise_or(m_a, m_b)
    return mask_to_polygon(union_mask, normalise=True)


def wall_centerlines(
    wall_mask: np.ndarray,
    min_line_length: int = 20,
    max_line_gap: int = 10,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Extract wall centre-lines from a binary wall mask.
    Returns list of ((x1,y1),(x2,y2)) line segments in pixel coords.

    Pipeline: thin mask → HoughLinesP.
    Requires cv2.ximgproc for thinning; falls back to skeleton via erosion.
    """
    if wall_mask is None or wall_mask.size == 0:
        return []

    m = (wall_mask > 0).astype(np.uint8) * 255

    # Try skeletonisation
    try:
        thin = cv2.ximgproc.thinning(m)
    except AttributeError:
        # ximgproc not available: approximate skeleton by iterative erosion
        thin = _approx_skeleton(m)

    lines = cv2.HoughLinesP(
        thin, rho=1, theta=np.pi / 180,
        threshold=15,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [((int(x1), int(y1)), (int(x2), int(y2))) for x1, y1, x2, y2 in lines[:, 0]]


def _approx_skeleton(mask: np.ndarray, iterations: int = 15) -> np.ndarray:
    """Approximate skeleton by repeated erosion + subtraction."""
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    for _ in range(iterations):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def reconstruct_walls(
    wall_mask: np.ndarray,
    close_k: int = 9,
    open_k: int = 3,
    min_wall_area: int = 200,
    epsilon_factor: float = 0.01,
) -> List[List[Tuple[float, float]]]:
    """
    Reconstruct continuous wall polygons from a (potentially noisy) wall mask.

    Pipeline:
      CLOSE (bridge gaps) → OPEN (remove noise) → connected components
      → contour per component → Douglas-Peucker approx → orthogonal snap.

    Returns:
        List of normalised polygon point lists, one per wall segment/region.
    """
    if wall_mask is None or wall_mask.size == 0:
        return []

    h, w = wall_mask.shape[:2]
    m = (wall_mask > 0).astype(np.uint8) * 255

    # Close gaps first, then open to clean noise
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    # Connected components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    walls = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_wall_area:
            continue
        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        eps = epsilon_factor * peri
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2).astype(float)
        pts = _snap_orthogonal(pts, threshold_deg=12.0)
        # Normalise
        pts[:, 0] /= (w or 1)
        pts[:, 1] /= (h or 1)
        pts = np.clip(pts, 0.0, 1.0)
        walls.append([(float(x), float(y)) for x, y in pts])

    return walls
