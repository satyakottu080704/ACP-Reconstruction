"""
evaluate.py — evaluate a trained model and return mAP metrics.

Two strategies:
  1. Live ultralytics validation (preferred): model.val() on the test split.
  2. Parse results.csv from a completed training run.

Usage:
  from mlops.evaluate import evaluate_model
  metrics = evaluate_model("models/weights/best.pt", data_yaml="datasets/new_final_yolo/floorplans.yaml")
  print(metrics["box_map50"], metrics["mask_map50"])
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, Optional


def evaluate_model(
    weights_path: str,
    data_yaml: Optional[str] = None,
    imgsz: int = 1280,
    split: str = "test",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model and return metrics.

    Tries ultralytics .val() first; falls back to parsing results.csv.

    Returns dict with keys:
      box_map50, mask_map50, box_map50_95, mask_map50_95, epoch (if from CSV)
    """
    try:
        return _evaluate_live(weights_path, data_yaml, imgsz, split, device)
    except Exception as e:
        print(f"[evaluate] live eval failed ({e}); trying results.csv fallback")

    # Try to find results.csv in the same run directory
    wp = Path(weights_path)
    results_csv = wp.parent.parent / "results.csv"  # runs/segment/name/results.csv
    if results_csv.exists():
        return _evaluate_from_csv(results_csv)

    return {"error": "evaluation failed", "box_map50": 0.0, "mask_map50": 0.0}


def _evaluate_live(
    weights_path: str,
    data_yaml: Optional[str],
    imgsz: int,
    split: str,
    device: Optional[str],
) -> Dict[str, Any]:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(f"ultralytics not available: {e}")

    if data_yaml is None:
        raise ValueError("data_yaml required for live evaluation")

    if device is None:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(weights_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        split=split,
        device=device,
        verbose=False,
    )

    metrics = results.results_dict
    return {
        "box_map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mask_map50": float(metrics.get("metrics/mAP50(M)", 0.0)),
        "box_map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "mask_map50_95": float(metrics.get("metrics/mAP50-95(M)", 0.0)),
        "source": "live",
    }


def _evaluate_from_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse the best row from a YOLO results.csv file."""
    best_row: Dict[str, str] = {}
    best_box = -1.0

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                box = float(row.get("metrics/mAP50(B)", 0))
                if box > best_box:
                    best_box = box
                    best_row = row
            except ValueError:
                continue

    if not best_row:
        return {"error": "no valid rows in results.csv", "box_map50": 0.0, "mask_map50": 0.0}

    return {
        "box_map50": float(best_row.get("metrics/mAP50(B)", 0)),
        "mask_map50": float(best_row.get("metrics/mAP50(M)", 0)),
        "box_map50_95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        "mask_map50_95": float(best_row.get("metrics/mAP50-95(M)", 0)),
        "epoch": int(float(best_row.get("epoch", 0))),
        "source": "results_csv",
    }
