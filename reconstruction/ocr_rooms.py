"""
ocr_rooms.py — OCR text inside each room mask region.

Strategy:
  1. Crop the bounding box of the room mask from the source image.
  2. Mask out neighbouring room pixels (so overlapping labels don't bleed in).
  3. Run OCR on the isolated region.
  4. Assign the best OCR hit as room.ocr_text + room.ocr_confidence.

OCR backend cascade (first available wins):
  PaddleOCR → EasyOCR → pytesseract → no-op (returns empty string)

Usage:
  from reconstruction.ocr_rooms import run_ocr_on_rooms
  run_ocr_on_rooms(plan, source_image_bgr)
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple, Optional

from .plan_model import PlanModel
from .masks import polygon_to_mask


# ─── backend detection (lazy import) ─────────────────────────────────────────

def _get_ocr_backend():
    """Return (name, callable) for the first available OCR backend."""
    try:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        def _paddle_ocr(img_bgr: np.ndarray) -> List[Tuple[str, float]]:
            result = _ocr.ocr(img_bgr, cls=True)
            hits = []
            if result and result[0]:
                for line in result[0]:
                    text, conf = line[1]
                    hits.append((text, float(conf)))
            return hits

        return "paddle", _paddle_ocr
    except Exception:
        pass

    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        def _easy_ocr(img_bgr: np.ndarray) -> List[Tuple[str, float]]:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = _reader.readtext(rgb)
            return [(text, float(conf)) for _, text, conf in result]

        return "easyocr", _easy_ocr
    except Exception:
        pass

    try:
        import pytesseract

        def _tess_ocr(img_bgr: np.ndarray) -> List[Tuple[str, float]]:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DICT, lang="eng"
            )
            hits = []
            for i, text in enumerate(data["text"]):
                text = text.strip()
                if text:
                    conf = float(data["conf"][i]) / 100.0
                    hits.append((text, conf))
            return hits

        return "tesseract", _tess_ocr
    except Exception:
        pass

    return "none", None


_BACKEND_NAME: Optional[str] = None
_BACKEND_FN = None


def _backend():
    global _BACKEND_NAME, _BACKEND_FN
    if _BACKEND_NAME is None:
        _BACKEND_NAME, _BACKEND_FN = _get_ocr_backend()
    return _BACKEND_NAME, _BACKEND_FN


# ─── public API ───────────────────────────────────────────────────────────────

def run_ocr_on_rooms(
    plan: PlanModel,
    source_image: np.ndarray,
    min_conf: float = 0.3,
    min_text_len: int = 2,
    padding_px: int = 4,
) -> PlanModel:
    """
    Run OCR inside each room mask and populate room.ocr_text + room.ocr_confidence.

    Args:
        plan: PlanModel with normalised room polygons.
        source_image: full BGR image (H x W x 3).
        min_conf: discard OCR hits below this confidence.
        min_text_len: discard hits shorter than this (noise filter).
        padding_px: extra pixels to pad the crop region.

    Returns:
        plan (modified in-place).
    """
    name, fn = _backend()
    if fn is None or source_image is None or source_image.size == 0:
        return plan

    h, w = source_image.shape[:2]

    for room in plan.rooms:
        if len(room.polygon) < 3:
            continue

        # Pixel-space bounding box of the room
        px_poly = [(int(x * w), int(y * h)) for x, y in room.polygon]
        xs = [p[0] for p in px_poly]
        ys = [p[1] for p in px_poly]
        x1 = max(0, min(xs) - padding_px)
        y1 = max(0, min(ys) - padding_px)
        x2 = min(w, max(xs) + padding_px)
        y2 = min(h, max(ys) + padding_px)

        if x2 <= x1 or y2 <= y1:
            continue

        # Create a mask for just this room in the crop region
        room_mask = polygon_to_mask(room.polygon, w, h, normalised=True)
        crop_mask = room_mask[y1:y2, x1:x2]
        crop_img = source_image[y1:y2, x1:x2].copy()

        # Zero out pixels outside the room mask (mask neighbours)
        crop_img[crop_mask == 0] = 255  # white background

        # Pre-process for better OCR: upscale small crops
        ch, cw = crop_img.shape[:2]
        if cw < 64 or ch < 32:
            scale = max(64 / max(cw, 1), 32 / max(ch, 1), 1.0)
            crop_img = cv2.resize(
                crop_img,
                (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            hits = fn(crop_img)
        except Exception:
            continue

        # Pick the best hit
        best_text, best_conf = "", 0.0
        for text, conf in hits:
            text = text.strip()
            if len(text) >= min_text_len and conf >= min_conf and conf > best_conf:
                best_text, best_conf = text, conf

        if best_text:
            room.ocr_text = best_text
            room.ocr_confidence = best_conf
            # If the room has no label yet, use OCR result
            if not room.label:
                room.label = best_text

    return plan


def ocr_backend_name() -> str:
    """Return the name of the active OCR backend ('paddle', 'easyocr', 'tesseract', or 'none')."""
    name, _ = _backend()
    return name
