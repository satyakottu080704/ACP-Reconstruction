"""
masks.py — mask refinement utilities (OpenCV/NumPy core, no optional deps).

Operations:
  - morphological open/close to remove noise and fill small holes
  - connected-component filtering to remove tiny fragments
  - hole filling via floodFill
  - mask quality scoring (0-1)
  - polygon_to_mask helper (inverse of contour extraction)
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple


def refine_mask(
    mask: np.ndarray,
    open_k: int = 3,
    close_k: int = 7,
    min_area_px: int = 100,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Morphologically clean a binary mask.

    Args:
        mask: uint8 binary mask (0/255 or 0/1).
        open_k: kernel size for morphological opening (noise removal).
        close_k: kernel size for morphological closing (gap bridging).
        min_area_px: connected components smaller than this are removed.
        fill_holes: fill enclosed background regions inside the mask.

    Returns:
        Refined uint8 binary mask (values 0 or 255).
    """
    if mask is None or mask.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    m = mask.astype(np.uint8)
    if m.max() == 1:
        m = m * 255

    # Morphological open: remove isolated noise pixels
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    # Morphological close: fill small gaps / thin breaks
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # Remove small components
    if min_area_px > 0:
        m = _remove_small_components(m, min_area_px)

    # Fill enclosed holes
    if fill_holes:
        m = _fill_holes(m)

    return m


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions using floodFill from the border."""
    h, w = mask.shape[:2]
    flood = mask.copy()
    border_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, border_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, filled_inv)


def mask_quality_score(mask: np.ndarray) -> float:
    """
    Heuristic quality score in [0, 1] for a single binary mask.

    Penalties:
      - Very small area fraction (< 0.001 of image)
      - High perimeter/area ratio (jagged outline)
      - Multiple disconnected components
    """
    if mask is None or mask.size == 0:
        return 0.0

    m = (mask > 0).astype(np.uint8)
    total = m.size
    area = int(m.sum())
    if area == 0:
        return 0.0

    score = 1.0

    # Penalty: too small
    area_frac = area / total
    if area_frac < 0.001:
        score *= 0.5
    elif area_frac < 0.005:
        score *= 0.8

    # Penalty: fragmentation
    n, _, stats, _ = cv2.connectedComponentsWithStats(m * 255, connectivity=8)
    n_components = n - 1  # subtract background
    if n_components > 3:
        score *= max(0.3, 1.0 - (n_components - 1) * 0.15)
    elif n_components > 1:
        score *= 0.85

    # Penalty: jagged boundary (high perimeter relative to sqrt(area))
    contours, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        # compactness = 1 for a circle, < 1 for jagged shapes
        if compactness < 0.1:
            score *= 0.6
        elif compactness < 0.3:
            score *= 0.85

    return float(np.clip(score, 0.0, 1.0))


def polygon_to_mask(
    polygon: List[Tuple[float, float]],
    width: int,
    height: int,
    normalised: bool = True,
) -> np.ndarray:
    """
    Rasterise a polygon into a binary mask.

    Args:
        polygon: list of (x, y). If normalised=True, values are in [0,1].
        width, height: output mask dimensions.
        normalised: if True, scale polygon by (width, height).

    Returns:
        uint8 mask of shape (height, width), values 0 or 255.
    """
    if not polygon:
        return np.zeros((height, width), dtype=np.uint8)

    pts = np.array(polygon, dtype=np.float32)
    if normalised:
        pts[:, 0] *= width
        pts[:, 1] *= height
    pts = pts.astype(np.int32).reshape(-1, 1, 2)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def resize_mask_to_image(
    mask: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize a mask to match target image dimensions."""
    if mask.shape[0] == target_height and mask.shape[1] == target_width:
        return mask
    return cv2.resize(mask, (target_width, target_height),
                      interpolation=cv2.INTER_NEAREST)
