"""
train.py — orchestrate train → evaluate → register → gated deploy.

Wraps train_floorplans.py, then calls mlops/deploy.py for the gate.

Usage:
  python mlops/train.py --data datasets/new_final_yolo --epochs 300
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="datasets/new_final_yolo")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--name", default="acorn_floorplans")
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--no-deploy", action="store_true", help="Skip deploy gate.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[mlops/train] starting training...")
    train_cmd = [
        sys.executable,
        str(root / "train_floorplans.py"),
        "--data", args.data,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--name", args.name,
    ]
    result = subprocess.run(train_cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"[mlops/train] training failed (exit {result.returncode})")

    # Locate produced weights
    weights_path = root / "runs" / "segment" / args.name / "weights" / "best.pt"
    if not weights_path.exists():
        sys.exit(f"[mlops/train] weights not found at {weights_path}")

    print(f"[mlops/train] training done. weights: {weights_path}")

    if args.no_deploy:
        print("[mlops/train] --no-deploy: skipping gate. Register manually with mlops/deploy.py.")
        return

    # ── Export ONNX ───────────────────────────────────────────────────────────
    onnx_path = weights_path.with_suffix(".onnx")
    try:
        export_cmd = [
            sys.executable,
            str(root / "mlops" / "export_onnx.py"),
            "--weights", str(weights_path),
            "--imgsz", str(args.imgsz),
        ]
        subprocess.run(export_cmd, check=False, timeout=300)
        print(f"[mlops/train] ONNX export: {onnx_path}")
    except Exception as e:
        print(f"[mlops/train] ONNX export failed (non-fatal): {e}")

    # ── Deploy gate ───────────────────────────────────────────────────────────
    from mlops.deploy import check_and_deploy
    promoted = check_and_deploy(
        str(weights_path),
        data_yaml=None,  # will parse results.csv as fallback
        imgsz=args.imgsz,
        tolerance=args.tolerance,
        dataset=args.data,
        notes=f"trained {args.epochs} epochs via mlops/train.py",
    )

    if promoted:
        print("[mlops/train] new model promoted to best.")
    else:
        print("[mlops/train] regression gate blocked promotion. Current best unchanged.")


if __name__ == "__main__":
    main()
