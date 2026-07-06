"""
cli.py — command-line interface for the v2 reconstruction engine.

Usage:
    python -m reconstruction.cli --image path/to/sketch.jpg --out output/ --formats json,svg,png
    python -m reconstruction.cli --image sketch.jpg  # defaults: all formats → ./output/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Acorn v2 floor-plan reconstruction engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m reconstruction.cli --image survey.jpg
  python -m reconstruction.cli --image survey.jpg --out results/ --formats json,svg,png
  python -m reconstruction.cli --image survey.jpg --no-ocr --conf 0.20
        """,
    )
    ap.add_argument("--image", required=True, help="Input sketch image (JPG/PNG).")
    ap.add_argument("--out", default="output", help="Output directory (created if absent).")
    ap.add_argument(
        "--formats", default="json,svg,dxf,png,pdf",
        help="Comma-separated export formats: json,svg,dxf,png,pdf,vsdx.",
    )
    ap.add_argument("--model", default=None,
                    help="Path to YOLO .pt weights (default: config.MODEL_PATH).")
    ap.add_argument("--onnx", default=None,
                    help="Path to ONNX weights (use instead of torch model).")
    ap.add_argument("--conf", type=float, default=None,
                    help="Confidence threshold (default: config.MODEL_CONF_THRESHOLD).")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Inference image size (default: config.MODEL_IMGSZ = 1280).")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR.")
    ap.add_argument("--no-snap", action="store_true", help="Skip geometry snapping.")
    ap.add_argument("--quality", action="store_true", help="Print quality report to stdout.")
    args = ap.parse_args()

    # ── Env check ──────────────────────────────────────────────────────────────
    engine = os.environ.get("RECONSTRUCTION_ENGINE", "legacy").strip().lower()
    if engine != "v2":
        print(
            "WARNING: RECONSTRUCTION_ENGINE is not set to 'v2'. "
            "Set it in your .env or environment to enable the v2 path.\n"
            "Continuing anyway (CLI always uses v2 directly)."
        )

    # ── Setup ─────────────────────────────────────────────────────────────────
    import sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    try:
        import config as _cfg
    except ImportError:
        class _cfg:
            MODEL_PATH = "models/weights/best.pt"
            MODEL_IMGSZ = 1280
            MODEL_CONF_THRESHOLD = 0.15

    model_path = args.model or _cfg.MODEL_PATH
    conf = args.conf if args.conf is not None else _cfg.MODEL_CONF_THRESHOLD
    imgsz = args.imgsz if args.imgsz is not None else _cfg.MODEL_IMGSZ

    import cv2

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        sys.exit(f"ERROR: image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"ERROR: could not read image: {image_path}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"[cli] input: {image_path}")

    if args.onnx:
        from reconstruction.onnx_infer import OnnxPredictor
        predictor = OnnxPredictor(args.onnx, imgsz=imgsz, conf_threshold=conf)
        if not predictor.is_available():
            sys.exit("ERROR: onnxruntime not available. Install it with: pip install onnxruntime")
        plan = predictor.predict(image, run_ocr=not args.no_ocr)
    else:
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ERROR: ultralytics not installed. Use --onnx for torch-free inference.")

        model = YOLO(model_path)
        results = model(image, imgsz=imgsz, conf=conf, verbose=False)
        from reconstruction.engine import from_yolo_result
        plan = from_yolo_result(
            results[0],
            source_image=image,
            run_ocr=not args.no_ocr,
            snap_distance=0.0 if args.no_snap else 0.015,
        )

    if plan is None:
        sys.exit("ERROR: reconstruction returned None")

    print(f"[cli] rooms={len(plan.rooms)} walls={len(plan.walls)} "
          f"doors={len(plan.doors)} stairs={len(plan.stairs)} "
          f"loft={plan.has_loft}")

    # ── Export ────────────────────────────────────────────────────────────────
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    try:
        from exporters import export_all
        paths = export_all(plan, out_dir, stem, formats=formats)
        for fmt, p in paths.items():
            print(f"[cli] {fmt}: {p}")
    except ImportError as e:
        print(f"[cli] exporters not available: {e}")

    # ── Quality report ─────────────────────────────────────────────────────────
    if args.quality or plan.quality.needs_review:
        from reconstruction.quality import quality_report_dict
        import json
        report = quality_report_dict(plan)
        print("\n[quality]", json.dumps(report, indent=2))
        if plan.quality.needs_review:
            print(f"\n⚠  Needs review: {plan.quality.review_flags}")


if __name__ == "__main__":
    main()
