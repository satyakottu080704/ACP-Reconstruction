"""
export_onnx.py — export YOLOv11-seg .pt weights to ONNX for torch-free serving.

The exported .onnx file can be loaded by reconstruction/onnx_infer.py
without requiring PyTorch.

Usage:
  python mlops/export_onnx.py --weights models/weights/best.pt
  python mlops/export_onnx.py --weights models/weights/best.pt --imgsz 1280 --opset 17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_onnx(
    weights_path: str,
    imgsz: int = 1280,
    opset: int = 17,
    dynamic: bool = False,
    half: bool = False,
    simplify: bool = True,
) -> Path:
    """
    Export a YOLOv11-seg model to ONNX.

    Returns:
        Path to the exported .onnx file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. pip install ultralytics")

    wp = Path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"Weights not found: {wp}")

    model = YOLO(str(wp))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        half=half,
        simplify=simplify,
    )
    out_path = Path(str(out))
    print(f"[export_onnx] exported: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="Path to .pt weights.")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Export image size (must match training imgsz=1280).")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true", help="Dynamic batch size.")
    ap.add_argument("--half", action="store_true", help="FP16 export (GPU only).")
    ap.add_argument("--no-simplify", action="store_true", help="Skip onnxsim.")
    args = ap.parse_args()

    try:
        out = export_onnx(
            args.weights,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            half=args.half,
            simplify=not args.no_simplify,
        )
        print(f"[done] {out}")
    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
