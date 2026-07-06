"""
app.py — FastAPI v2 reconstruction service.

Endpoints:
  GET  /health         — liveness
  GET  /version        — model + engine info
  GET  /metrics        — request counters
  POST /predict        — YOLO inference + basic room list
  POST /reconstruct    — full v2 reconstruction (returns JSON plan)
  POST /export         — inference + export to requested format (file download)
  POST /batch_predict  — multiple images (multipart)
  POST /ocr            — OCR room labels from a pre-segmented image

All endpoints accept: multipart/form-data with field "image" (image bytes).
/export also accepts query param ?format=json|png|svg|dxf|pdf|vsdx

Gated by RECONSTRUCTION_ENGINE (must be set; warns if not v2).

Run:
  uvicorn api.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from api.service import predict_image, export_image, health_check, version_info

app = FastAPI(title="Acorn v2 Floor Plan API", version="2.0.0")

# ── metrics counter ───────────────────────────────────────────────────────────
_counters: dict = defaultdict(int)
_start_time = time.time()


# ─── health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return health_check()


@app.get("/version")
async def version():
    return version_info()


@app.get("/metrics")
async def metrics():
    return {
        "uptime_s": round(time.time() - _start_time, 1),
        "requests": dict(_counters),
        "engine": os.environ.get("RECONSTRUCTION_ENGINE", "legacy"),
    }


# ─── predict ──────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: Optional[float] = Query(None, description="Confidence threshold (0.15–0.25 recommended)"),
):
    """
    Run YOLO inference + v2 reconstruction.
    Returns JSON with room list and quality report.
    """
    _counters["predict"] += 1
    data = await image.read()
    result = predict_image(data, conf=conf, run_ocr=True)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(result)


@app.post("/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    conf: Optional[float] = Query(None),
):
    """Full reconstruction — returns the complete PlanModel as JSON."""
    _counters["reconstruct"] += 1
    data = await image.read()
    result = predict_image(data, conf=conf, run_ocr=True)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return JSONResponse(result)


# ─── export ───────────────────────────────────────────────────────────────────

_MIME_TYPES = {
    "json": "application/json",
    "png": "image/png",
    "svg": "image/svg+xml",
    "dxf": "application/dxf",
    "pdf": "application/pdf",
    "vsdx": "application/vnd.visio",
}


@app.post("/export")
async def export(
    image: UploadFile = File(...),
    format: str = Query("json", description="json|png|svg|dxf|pdf|vsdx"),
    conf: Optional[float] = Query(None),
):
    """Inference + reconstruction + file export in the requested format."""
    _counters["export"] += 1
    fmt = format.lower()
    if fmt not in _MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")

    data = await image.read()
    try:
        file_bytes = export_image(data, fmt=fmt, conf=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=_MIME_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="plan.{fmt}"'},
    )


# ─── batch_predict ────────────────────────────────────────────────────────────

@app.post("/batch_predict")
async def batch_predict(
    images: list[UploadFile] = File(...),
    conf: Optional[float] = Query(None),
):
    """Run predict on multiple images."""
    _counters["batch_predict"] += 1
    results = []
    for img_file in images:
        data = await img_file.read()
        r = predict_image(data, conf=conf, run_ocr=True)
        r["filename"] = img_file.filename
        results.append(r)
    return JSONResponse({"results": results})


# ─── ocr ─────────────────────────────────────────────────────────────────────

@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    """
    Run OCR on a raw image without YOLO segmentation.
    Returns detected text regions.
    """
    _counters["ocr"] += 1
    import cv2
    import numpy as np
    from reconstruction.ocr_rooms import ocr_backend_name

    data = await image.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    backend = ocr_backend_name()
    if backend == "none":
        return JSONResponse({"backend": "none", "text": [], "note": "No OCR backend installed"})

    # Create a fake whole-image room polygon
    from reconstruction.plan_model import PlanModel, RoomPolygon
    from reconstruction.ocr_rooms import run_ocr_on_rooms

    plan = PlanModel(image_width=img.shape[1], image_height=img.shape[0])
    plan.rooms = [RoomPolygon(
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        label="",
    )]
    run_ocr_on_rooms(plan, img)

    return JSONResponse({
        "backend": backend,
        "text": [{"text": r.ocr_text, "confidence": r.ocr_confidence} for r in plan.rooms],
    })
