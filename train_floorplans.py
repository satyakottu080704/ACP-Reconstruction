#!/usr/bin/env python3
"""
Train a YOLOv11 instance-segmentation model on the Acorn floor-plan dataset.

Dataset: "New Final YOLO" — 7,294 images, 6 classes
(acm, door, floor, room, stairs, walls). Polygon/segmentation labels.
Target image size: 1280x1280 (= MODEL_IMGSZ; do NOT downscale).

Current production model: best.pt
  box mAP50 = 0.883, mask mAP50 = 0.803, trained at imgsz 1280.
  A 50-epoch re-run scored 0.855 / 0.769 — the regression gate blocks it.
  Do NOT replace best.pt unless the new run beats these numbers on the test set.

Usage:
    python train_floorplans.py                       # train, sensible defaults
    python train_floorplans.py --epochs 300 --model yolo11l-seg.pt
    python train_floorplans.py --data datasets/new_final_yolo

NOTE ON HARDWARE: needs a GPU. On CPU it is effectively untrainable.
Run on a CUDA machine or Colab. The script auto-detects the device.

NOTE ON STAIRS: 'stairs' is the weakest class (~8x rarer than rooms in the
dataset). Add more stairs examples before the next training run.

----------------------------------------------------------------------------
POST-TRAINING INTEGRATION (only promote if regression gate passes):

  1. Run mlops/deploy.py — it gates on box/mask mAP vs the current best.pt.
  2. If promoted: copy runs/.../weights/best.pt -> models/weights/best.pt
  3. Register the new weights in mlops/registry.py.
  4. config.py is already correct (6 classes, MODEL_IMGSZ=1280).
----------------------------------------------------------------------------
"""
import argparse
import sys
from pathlib import Path


def _build_data_yaml(dataset_dir: Path) -> Path:
    """
    Roboflow's data.yaml uses paths like '../train/images', which only
    resolve correctly from inside a sub-folder. Ultralytics resolves dataset
    paths relative to the yaml's own directory, so we rewrite a clean yaml
    with absolute paths to avoid silent "0 images found" failures.
    """
    for split in ("train", "valid", "test"):
        if not (dataset_dir / split / "images").is_dir():
            sys.exit(f"ERROR: missing {split}/images under {dataset_dir} — "
                     f"extract the dataset there first.")

    fixed = dataset_dir / "floorplans.yaml"
    fixed.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: 6\n"
        f"names: ['acm', 'door', 'floor', 'room', 'stairs', 'walls']\n",
        encoding="utf-8",
    )
    print(f"[data] wrote resolved dataset config: {fixed}")
    return fixed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/new_final_yolo",
                    help="Dataset root (contains train/ valid/ test/).")
    ap.add_argument("--model", default="yolo11m-seg.pt",
                    help="Base seg model: yolo11n/s/m/l-seg.pt (bigger = "
                         "slower but more accurate). Default: yolo11m-seg.pt.")
    ap.add_argument("--epochs", type=int, default=300,
                    help="Epoch CEILING. Early-stopping (--patience) keeps "
                         "the best checkpoint well before this.")
    ap.add_argument("--patience", type=int, default=50,
                    help="Stop if val metrics don't improve for N epochs.")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Training image size. MUST match MODEL_IMGSZ=1280 in "
                         "config.py. Changing this silently degrades inference accuracy.")
    ap.add_argument("--batch", type=int, default=-1,
                    help="Batch size. -1 = auto-fit to GPU memory.")
    ap.add_argument("--name", default="acorn_floorplans",
                    help="Run name under runs/segment/.")
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"ERROR: {e}. Install with: pip install ultralytics")

    if args.imgsz != 1280:
        print(f"WARNING: --imgsz {args.imgsz} != 1280. Inference uses MODEL_IMGSZ=1280; "
              "a mismatch silently degrades accuracy. Use 1280 unless you know why.")

    dataset_dir = Path(args.data).resolve()
    data_yaml = _build_data_yaml(dataset_dir)

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU detected. Segmentation training on CPU "
              "is impractically slow — use a GPU machine or Colab.")
    else:
        print(f"[device] CUDA GPU: {torch.cuda.get_device_name(0)}")

    print(f"[train] model={args.model} epochs<={args.epochs} "
          f"patience={args.patience} imgsz={args.imgsz} batch={args.batch}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        # Dataset already has augmentation baked in; keep extra modest.
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        plots=True,
    )

    print("\n[done] Best weights: runs/segment/"
          f"{args.name}/weights/best.pt")
    print("Next: run mlops/deploy.py to gate against the current best.pt "
          "before promoting (regression gate required).")


if __name__ == "__main__":
    main()
