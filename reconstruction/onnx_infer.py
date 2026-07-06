"""
onnx_infer.py — ONNX Runtime predictor (torch-free serving).

Loads a YOLOv11-seg model exported to ONNX and runs inference,
then feeds the detections into the v2 reconstruction engine.

Export your model first:
    python mlops/export_onnx.py --weights models/weights/best.pt

Usage:
    from reconstruction.onnx_infer import OnnxPredictor
    predictor = OnnxPredictor("models/weights/best.onnx")
    plan = predictor.predict(image_bgr)
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

from .engine import reconstruct_from_masks
from .plan_model import PlanModel


class OnnxPredictor:
    """
    YOLOv11-seg ONNX Runtime inference wrapper.

    Requires: onnxruntime (CPU) or onnxruntime-gpu.
    Falls back gracefully if onnxruntime is not installed.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 1280,
        conf_threshold: float = 0.15,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self._session = None

        try:
            import onnxruntime as ort
            _providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(model_path, providers=_providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            print(f"[onnx] loaded {model_path} on {self._session.get_providers()}")
        except ImportError:
            print("[onnx] onnxruntime not installed; OnnxPredictor is a no-op. "
                  "Install with: pip install onnxruntime-gpu")
        except Exception as e:
            print(f"[onnx] failed to load {model_path}: {e}")

    def is_available(self) -> bool:
        return self._session is not None

    def predict(
        self,
        image: np.ndarray,
        run_ocr: bool = True,
    ) -> Optional[PlanModel]:
        """
        Run inference on a BGR image and return a PlanModel.

        Returns None if onnxruntime is unavailable or inference fails.
        """
        if not self.is_available():
            return None

        h_orig, w_orig = image.shape[:2]

        # Pre-process: letterbox to imgsz x imgsz
        img_resized, ratio, (dw, dh) = _letterbox(image, self.imgsz)
        img_input = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img_input = img_input[np.newaxis].astype(np.float32) / 255.0

        try:
            outputs = self._session.run(self._output_names, {self._input_name: img_input})
        except Exception as e:
            print(f"[onnx] inference error: {e}")
            return None

        detections = _parse_yolo_outputs(
            outputs, self.conf_threshold,
            orig_size=(w_orig, h_orig),
            input_size=self.imgsz,
            ratio=ratio, dw=dw, dh=dh,
        )
        return reconstruct_from_masks(
            detections, (w_orig, h_orig),
            source_image=image,
            run_ocr=run_ocr,
        )


# ─── pre/post-processing helpers ──────────────────────────────────────────────

def _letterbox(
    img: np.ndarray,
    size: int = 1280,
    colour: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox resize preserving aspect ratio."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    dw, dh = (size - new_w) // 2, (size - new_h) // 2
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), colour, dtype=np.uint8)
    out[dh:dh + new_h, dw:dw + new_w] = img_r
    return out, ratio, (dw, dh)


_CLS_NAMES = {0: "acm", 1: "door", 2: "floor", 3: "room", 4: "stairs", 5: "walls"}


def _parse_yolo_outputs(
    outputs: List[np.ndarray],
    conf_thresh: float,
    orig_size: Tuple[int, int],
    input_size: int,
    ratio: float,
    dw: int,
    dh: int,
) -> List[Dict[str, Any]]:
    """
    Parse YOLOv8/11 ONNX detection output to detection dicts.

    YOLOv8-seg ONNX output format:
      output0: (1, 4+nc+32, N)  — bboxes + class scores + mask coefficients
      output1: (1, 32, H/4, W/4) — mask prototypes

    This is a best-effort parser; export format may vary by ultralytics version.
    """
    w_orig, h_orig = orig_size
    detections = []

    if len(outputs) < 2:
        return detections

    try:
        pred = outputs[0][0]  # (4+nc+32, N) or (N, 4+nc+32)
        protos = outputs[1][0]  # (32, H, W)

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # normalise to (N, ...)

        nc = len(_CLS_NAMES)
        for row in pred:
            box = row[:4]
            scores = row[4:4 + nc]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < conf_thresh:
                continue

            # Decode bbox (xc, yc, w, h) → pixel coords in orig image
            xc, yc, bw, bh = box
            x1 = int((xc - bw / 2 - dw) / ratio)
            y1 = int((yc - bh / 2 - dh) / ratio)
            x2 = int((xc + bw / 2 - dw) / ratio)
            y2 = int((yc + bh / 2 - dh) / ratio)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            # Decode mask
            mask_coeffs = row[4 + nc:4 + nc + 32]
            proto_h, proto_w = protos.shape[1], protos.shape[2]
            mask_map = (mask_coeffs @ protos.reshape(32, -1)).reshape(proto_h, proto_w)
            mask_map = 1 / (1 + np.exp(-mask_map))  # sigmoid
            mask_resized = cv2.resize(mask_map, (input_size, input_size))
            # Crop to bbox area before rescaling to orig
            mask_cropped = mask_resized[
                max(0, int(yc - bh / 2)):int(yc + bh / 2),
                max(0, int(xc - bw / 2)):int(xc + bw / 2),
            ]
            if mask_cropped.size > 0:
                mask_orig = cv2.resize(mask_cropped, (x2 - x1, y2 - y1))
                full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = (mask_orig > 0.5).astype(np.uint8) * 255
            else:
                full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

            detections.append({
                "class_name": _CLS_NAMES.get(cls_id, f"class_{cls_id}"),
                "class_id": cls_id,
                "confidence": conf,
                "mask": full_mask,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })
    except Exception as e:
        print(f"[onnx] output parsing error: {e}")

    return detections
