"""
service.py — framework-agnostic v2 reconstruction service core.

Contains all business logic (load model, run inference, reconstruct, export).
FastAPI app.py and any future framework wrapper delegate to this.

Gated by RECONSTRUCTION_ENGINE=v2. Falls back to legacy pipeline if not v2.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import cv2

# ── lazy model loader ─────────────────────────────────────────────────────────

_model = None
_model_path: Optional[str] = None


def _get_model():
    global _model, _model_path
    import sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    try:
        import config as _cfg
        mp = _cfg.MODEL_PATH
        imgsz = _cfg.MODEL_IMGSZ
        conf = _cfg.MODEL_CONF_THRESHOLD
    except Exception:
        mp = str(_root / "models" / "weights" / "best.pt")
        imgsz = 1280
        conf = 0.15

    if _model is None or mp != _model_path:
        try:
            from ultralytics import YOLO
            _model = YOLO(mp)
            _model_path = mp
            print(f"[service] loaded model: {mp}")
        except Exception as e:
            print(f"[service] could not load model: {e}")
            _model = None

    return _model, imgsz, conf


# ── public service functions ──────────────────────────────────────────────────

def predict_image(
    image_bytes: bytes,
    conf: Optional[float] = None,
    run_ocr: bool = True,
) -> Dict[str, Any]:
    """
    Run YOLO inference + v2 reconstruction on raw image bytes.
    Returns serialisable dict.
    """
    model, imgsz, default_conf = _get_model()
    if model is None:
        return {"error": "model not loaded"}

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "could not decode image"}

    # Clamp confidence to a safe operating range.
    # Values above 0.50 almost always return zero detections for hand-drawn
    # sketches — a common UI misconfiguration (e.g. slider left at 99%).
    _CONF_MIN, _CONF_MAX = 0.05, 0.50
    raw_conf = conf if conf is not None else default_conf
    effective_conf = max(_CONF_MIN, min(_CONF_MAX, raw_conf))
    if raw_conf != effective_conf:
        print(f"[service] conf {raw_conf:.2f} clamped to {effective_conf:.2f} "
              f"(valid range {_CONF_MIN}–{_CONF_MAX})")
    results = model(img, imgsz=imgsz, conf=effective_conf, verbose=False)

    try:
        from reconstruction.engine import from_yolo_result
        from reconstruction.quality import quality_report_dict
        plan = from_yolo_result(results[0], source_image=img, run_ocr=run_ocr)
        return {
            "room_count": len(plan.rooms),
            "wall_count": len(plan.walls),
            "door_count": len(plan.doors),
            "stair_count": len(plan.stairs),
            "has_loft": plan.has_loft,
            "quality": quality_report_dict(plan),
            "rooms": [
                {
                    "label": r.label,
                    "room_type": r.room_type,
                    "is_acm": r.is_acm,
                    "floor_idx": r.floor_idx,
                    "confidence": r.confidence,
                    "centroid": list(r.centroid()),
                    "bbox": list(r.bbox),
                }
                for r in plan.rooms
            ],
        }
    except Exception as e:
        return {"error": f"reconstruction failed: {e}"}


def export_image(
    image_bytes: bytes,
    fmt: str = "json",
    conf: Optional[float] = None,
) -> bytes:
    """Run inference + reconstruction + export to the requested format."""
    model, imgsz, default_conf = _get_model()
    if model is None:
        raise RuntimeError("model not loaded")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")

    _raw = conf if conf is not None else default_conf
    results = model(img, imgsz=imgsz, conf=max(0.05, min(0.50, _raw)), verbose=False)

    from reconstruction.engine import from_yolo_result
    plan = from_yolo_result(results[0], source_image=img)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"plan.{fmt.lower()}"
        _dispatch_export(plan, out_path, fmt)
        return out_path.read_bytes()


def _dispatch_export(plan, out_path: Path, fmt: str):
    fmt = fmt.lower()
    if fmt == "json":
        from exporters.json_export import export_json
        export_json(plan, out_path)
    elif fmt == "png":
        from exporters.png_export import export_png
        export_png(plan, out_path)
    elif fmt == "svg":
        from exporters.svg_export import export_svg
        export_svg(plan, out_path)
    elif fmt == "dxf":
        from exporters.dxf_export import export_dxf
        export_dxf(plan, out_path)
    elif fmt == "pdf":
        from exporters.pdf_export import export_pdf
        export_pdf(plan, out_path)
    elif fmt == "vsdx":
        from exporters.vsdx_export import export_vsdx
        export_vsdx(plan, out_path)
    else:
        raise ValueError(f"unknown format: {fmt}")


def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine": os.environ.get("RECONSTRUCTION_ENGINE", "legacy"),
        "model_loaded": _model is not None,
    }


def version_info() -> Dict[str, Any]:
    import sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    try:
        import config as _cfg
        return {
            "engine": "v2",
            "model_path": _cfg.MODEL_PATH,
            "model_imgsz": _cfg.MODEL_IMGSZ,
            "classes": _cfg.CLASSES,
            "num_classes": _cfg.NUM_CLASSES,
        }
    except Exception:
        return {"engine": "v2"}
