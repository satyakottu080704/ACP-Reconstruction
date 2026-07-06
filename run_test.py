"""
run_test.py — End-to-end test: image → YOLO segmentation → v2 reconstruction → VSDX output.

Usage:
    python run_test.py --image path/to/sketch.jpg
    python run_test.py --image path/to/sketch.jpg --conf 0.15 --formats vsdx,png,json
    python run_test.py --image path/to/sketch.jpg --engine v2

Output is written to:
    output/visio/<stem>_ground.vsdx        (always)
    output/visio/<stem>_loft.vsdx          (only if loft detected)
    output/<stem>.png  / .json / .svg ...  (if requested)

Requirements:
    pip install ultralytics opencv-python
    OPENAI_API_KEY in .env (optional — for room label reading)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── project root on path ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# load .env if present
def _load_env():
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()


def run(image_path: str, conf: float, engine: str, formats: list[str]) -> dict:
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)

    import cv2
    import config

    # Override engine if requested
    if engine:
        os.environ["RECONSTRUCTION_ENGINE"] = engine
        config.RECONSTRUCTION_ENGINE = engine

    print(f"\n{'='*60}")
    print(f"  Acorn Floor Plan Reconstruction")
    print(f"  Image  : {img_path.name}")
    print(f"  Engine : {config.RECONSTRUCTION_ENGINE}")
    print(f"  Model  : {config.MODEL_PATH}")
    print(f"  Conf   : {conf}")
    print(f"{'='*60}\n")

    # ── 1. Load image ─────────────────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: could not read image: {img_path}")
        sys.exit(1)
    print(f"[load] image {img.shape[1]}×{img.shape[0]} px")

    # ── 2. YOLO inference ─────────────────────────────────────────────────────
    t0 = time.time()
    from ultralytics import YOLO
    model = YOLO(config.MODEL_PATH)
    results = model(
        img,
        imgsz=config.MODEL_IMGSZ,
        conf=conf,
        verbose=False,
    )
    result = results[0]
    dt_infer = time.time() - t0
    n_det = len(result.boxes) if result.boxes is not None else 0
    print(f"[yolo]  {n_det} detections in {dt_infer:.2f}s  (imgsz={config.MODEL_IMGSZ})")

    if n_det == 0:
        print("WARNING: no detections. Try lowering --conf (e.g. 0.10)")

    # Print class breakdown
    if result.boxes is not None and result.names:
        from collections import Counter
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        counts = Counter(result.names[c] for c in cls_ids)
        print(f"[yolo]  {dict(counts)}")

    # ── 3. Reconstruct ────────────────────────────────────────────────────────
    t1 = time.time()
    use_v2 = config.RECONSTRUCTION_ENGINE == "v2"
    if use_v2:
        from reconstruction.engine import from_yolo_result
        plan = from_yolo_result(result, source_image=img, run_ocr=True)
        dt_recon = time.time() - t1
        print(f"[v2]    {len(plan.rooms)} rooms, {len(plan.doors)} doors, "
              f"{len(plan.stairs)} stairs, loft={plan.has_loft}  ({dt_recon:.2f}s)")
        print(f"[v2]    quality score: {plan.quality.overall_score:.3f}")
        if plan.quality.review_flags:
            print(f"[v2]    flags: {plan.quality.review_flags}")
    else:
        print("[legacy] Using legacy pipeline — output will be via pipeline.py")
        plan = None  # legacy path outputs directly

    # ── 4. Export ─────────────────────────────────────────────────────────────
    output_dir = Path(config.OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    produced: dict = {}

    if use_v2 and plan is not None:
        from exporters import export_all
        from exporters.vsdx_export import export_vsdx

        # Always produce VSDX
        vsdx_out = output_dir / "visio" / f"{stem}.vsdx"
        t2 = time.time()
        vsdx_files = export_vsdx(plan, vsdx_out, stem=stem)
        produced["vsdx"] = vsdx_files
        print(f"[vsdx]  produced in {time.time()-t2:.2f}s")

        # Other requested formats
        other_fmts = [f for f in formats if f != "vsdx"]
        if other_fmts:
            exported = export_all(plan, output_dir, stem=stem, formats=other_fmts)
            produced.update(exported)

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Output files:")
    if "vsdx" in produced:
        for key, path in produced["vsdx"].items():
            tag = "LOFT" if key == "loft" else "GROUND"
            print(f"    [{tag}] {path}")
    for fmt, path in produced.items():
        if fmt != "vsdx":
            print(f"    [{fmt.upper()}]  {path}")
    print(f"{'='*60}\n")

    return produced


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to floor-plan sketch image")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="YOLO confidence threshold (default 0.15)")
    ap.add_argument("--engine", default="v2", choices=["v2", "legacy"],
                    help="Reconstruction engine (default v2)")
    ap.add_argument("--formats", default="vsdx,png,json",
                    help="Comma-separated output formats (default: vsdx,png,json)")
    args = ap.parse_args()

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    run(args.image, args.conf, args.engine, formats)


if __name__ == "__main__":
    main()
