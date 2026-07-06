"""
per_class_metrics.py — hybrid per-class metric strategy for the 6-class
floor-plan model [acm, door, floor, room, stairs, walls].

Strategy (per-class evaluation instead of overall mAP alone):
  - Structural / layout-defining classes -> Box mAP50 (primary):
        rooms, doors, walls, ACM
    These define structure, position and connectivity; boxes are stable
    and sufficient for layout.
  - Area / region classes -> Mask mAP50-95 (primary):
        floor, stairs (and loft-hatch if ever added as a class)
    Shape/area accuracy matters for room filling and stair geometry.

Targets (gate thresholds):
    Box  mAP50    > 0.85  for structural classes  (excellent > 0.90)
    Mask mAP50-95 > 0.50  for area classes        (excellent > 0.65)

Usage:
    from mlops.per_class_metrics import evaluate_per_class, check_class_gates
    report = evaluate_per_class("models/weights/best.pt", "datasets/new_final_yolo/data.yaml")
    ok, failures = check_class_gates(report)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Class order must match training order (config.CLASSES)
CLASS_NAMES = ["acm", "door", "floor", "room", "stairs", "walls"]

# Which metric is primary for each class
BOX_MAP50_CLASSES = ["room", "door", "walls", "acm"]     # structural
MASK_MAP5095_CLASSES = ["floor", "stairs"]               # area/region

# Gate targets ("good" thresholds; "excellent" in parentheses)
BOX_MAP50_TARGET = 0.85          # excellent: 0.90
MASK_MAP5095_TARGET = 0.50       # excellent: 0.65
STAIRS_RELAXED_TARGET = 0.40     # stairs is ~8x rarer; relax until more data


def evaluate_per_class(
    weights: Union[str, Path],
    data_yaml: Union[str, Path],
    imgsz: int = 1280,          # MUST equal training imgsz (do not lower)
    conf: float = 0.001,        # val-time conf (ultralytics default for mAP)
    device: str = "",
) -> Dict[str, Any]:
    """
    Run ultralytics validation and return per-class metrics:

    {
      "per_class": {
        "room":  {"box_map50": 0.94, "box_map50_95": 0.71,
                  "mask_map50": 0.90, "mask_map50_95": 0.55,
                  "primary_metric": "box_map50", "primary_value": 0.94},
        ...
      },
      "overall": {"box_map50": ..., "mask_map50": ..., ...}
    }

    Requires ultralytics + torch (GPU host). Raises RuntimeError otherwise.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics is required for live per-class evaluation "
            "(run on the GPU host)") from e

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), imgsz=imgsz, conf=conf,
                  device=device or None, verbose=False)

    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    idx_of = {v: k for k, v in names.items()}

    def _per_class(metric_obj, attr):
        # ultralytics Metric: .ap50 and .ap are arrays indexed by ap_class_index
        vals = {}
        try:
            arr = getattr(metric_obj, attr)
            cls_idx = list(getattr(metric_obj, "ap_class_index", range(len(arr))))
            for ci, v in zip(cls_idx, arr):
                vals[names.get(int(ci), str(ci))] = float(v)
        except Exception:
            pass
        return vals

    box50 = _per_class(m.box, "ap50")
    box5095 = _per_class(m.box, "ap")
    seg50 = _per_class(m.seg, "ap50") if hasattr(m, "seg") else {}
    seg5095 = _per_class(m.seg, "ap") if hasattr(m, "seg") else {}

    per_class: Dict[str, Any] = {}
    for cname in CLASS_NAMES:
        entry = {
            "box_map50": box50.get(cname, 0.0),
            "box_map50_95": box5095.get(cname, 0.0),
            "mask_map50": seg50.get(cname, 0.0),
            "mask_map50_95": seg5095.get(cname, 0.0),
        }
        if cname in MASK_MAP5095_CLASSES:
            entry["primary_metric"] = "mask_map50_95"
        else:
            entry["primary_metric"] = "box_map50"
        entry["primary_value"] = entry[entry["primary_metric"]]
        per_class[cname] = entry

    overall = {
        "box_map50": float(m.box.map50),
        "box_map50_95": float(m.box.map),
        "mask_map50": float(m.seg.map50) if hasattr(m, "seg") else 0.0,
        "mask_map50_95": float(m.seg.map) if hasattr(m, "seg") else 0.0,
    }
    return {"per_class": per_class, "overall": overall,
            "imgsz": imgsz, "weights": str(weights)}


def check_class_gates(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Apply the hybrid per-class gates.  Returns (passed, failure_messages).

    Gates:
      room/door/walls/acm : box mAP50    >= 0.85
      floor               : mask mAP50-95 >= 0.50
      stairs              : mask mAP50-95 >= 0.40 (relaxed: rare class)
    """
    failures: List[str] = []
    pc = report.get("per_class", {})

    for cname in BOX_MAP50_CLASSES:
        v = pc.get(cname, {}).get("box_map50", 0.0)
        if v < BOX_MAP50_TARGET:
            failures.append(
                f"{cname}: box mAP50 {v:.3f} < target {BOX_MAP50_TARGET}")

    for cname in MASK_MAP5095_CLASSES:
        target = STAIRS_RELAXED_TARGET if cname == "stairs" else MASK_MAP5095_TARGET
        v = pc.get(cname, {}).get("mask_map50_95", 0.0)
        if v < target:
            failures.append(
                f"{cname}: mask mAP50-95 {v:.3f} < target {target}")

    return (len(failures) == 0), failures


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable per-class report table."""
    lines = [
        f"{'class':<8} {'box mAP50':>10} {'box 50-95':>10} "
        f"{'mask mAP50':>11} {'mask 50-95':>11}  primary",
        "-" * 62,
    ]
    for cname, e in report.get("per_class", {}).items():
        lines.append(
            f"{cname:<8} {e['box_map50']:>10.3f} {e['box_map50_95']:>10.3f} "
            f"{e['mask_map50']:>11.3f} {e['mask_map50_95']:>11.3f}  "
            f"{e['primary_metric']}={e['primary_value']:.3f}"
        )
    o = report.get("overall", {})
    lines.append("-" * 62)
    lines.append(
        f"overall  box mAP50 {o.get('box_map50', 0):.3f} | "
        f"mask mAP50 {o.get('mask_map50', 0):.3f} | "
        f"mask mAP50-95 {o.get('mask_map50_95', 0):.3f}"
    )
    ok, failures = check_class_gates(report)
    lines.append(f"GATE: {'PASS' if ok else 'FAIL'}")
    lines.extend(f"  - {f}" for f in failures)
    return "\n".join(lines)
