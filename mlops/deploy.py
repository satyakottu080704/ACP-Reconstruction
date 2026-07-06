"""
deploy.py — regression gate: promote a new model only if it doesn't regress.

Gate rule:
  new.box_map50  >= best.box_map50  - tolerance   AND
  new.mask_map50 >= best.mask_map50 - tolerance

If the gate passes: register + set_best in the registry.
If it fails: register as a tracked version (for history) but do NOT promote.

Known blocked run:
  50-epoch run: box=0.855 / mask=0.769  < current best 0.883 / 0.803
  → gate correctly blocks it.

Usage:
  python mlops/deploy.py --weights runs/.../best.pt --data datasets/.../floorplans.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def check_and_deploy(
    new_weights: str,
    data_yaml: Optional[str] = None,
    imgsz: int = 1280,
    tolerance: float = 0.005,
    registry_path: Optional[str] = None,
    dataset: str = "",
    notes: str = "",
    device: Optional[str] = None,
    per_class_gate: bool = False,
) -> bool:
    """
    Evaluate new_weights, compare against current best, promote if gate passes.

    Args:
        per_class_gate: additionally require the hybrid per-class gates
            (box mAP50 >= 0.85 for room/door/walls/acm; mask mAP50-95
            >= 0.50 for floor, >= 0.40 for stairs). Needs live eval
            (ultralytics + data_yaml).

    Returns:
        True if promoted, False if blocked.
    """
    from mlops.evaluate import evaluate_model
    from mlops.registry import ModelRegistry

    reg = ModelRegistry(registry_path=registry_path)
    best = reg.get_best()

    print(f"[deploy] evaluating {new_weights} ...")
    new_metrics = evaluate_model(new_weights, data_yaml=data_yaml, imgsz=imgsz, device=device)

    if "error" in new_metrics:
        print(f"[deploy] evaluation failed: {new_metrics['error']}")
        return False

    new_box = new_metrics["box_map50"]
    new_mask = new_metrics["mask_map50"]
    print(f"[deploy] new model:  box={new_box:.4f}  mask={new_mask:.4f}")

    if best:
        best_box = best["box_map50"]
        best_mask = best["mask_map50"]
        print(f"[deploy] current best: box={best_box:.4f}  mask={best_mask:.4f}")

        gate_box = new_box >= best_box - tolerance
        gate_mask = new_mask >= best_mask - tolerance

        if not gate_box:
            print(f"[deploy] BLOCKED — box mAP regression: {new_box:.4f} < {best_box:.4f} - {tolerance}")
        if not gate_mask:
            print(f"[deploy] BLOCKED — mask mAP regression: {new_mask:.4f} < {best_mask:.4f} - {tolerance}")

        if not gate_box or not gate_mask:
            # Register for tracking but don't promote
            version = reg.register(
                new_weights,
                box_map50=new_box,
                mask_map50=new_mask,
                dataset=dataset,
                notes=notes or "blocked by regression gate",
            )
            print(f"[deploy] registered as v{version} (not promoted — regression gate blocked)")
            return False
    else:
        print("[deploy] no current best — first registration, promoting automatically")

    # Optional hybrid per-class gate (box mAP50 for structural classes,
    # mask mAP50-95 for area classes)
    if per_class_gate and data_yaml:
        try:
            from mlops.per_class_metrics import (
                evaluate_per_class, check_class_gates, format_report)
            pc_report = evaluate_per_class(new_weights, data_yaml,
                                           imgsz=imgsz, device=device or "")
            print(format_report(pc_report))
            ok, failures = check_class_gates(pc_report)
            if not ok:
                version = reg.register(
                    new_weights, box_map50=new_box, mask_map50=new_mask,
                    dataset=dataset,
                    notes=(notes or "") + " | blocked by per-class gate: "
                          + "; ".join(failures),
                )
                print(f"[deploy] registered as v{version} "
                      f"(not promoted — per-class gate blocked)")
                return False
        except RuntimeError as e:
            print(f"[deploy] per-class gate skipped: {e}")

    # Gate passed (or no existing best): register and promote
    version = reg.register(
        new_weights,
        box_map50=new_box,
        mask_map50=new_mask,
        dataset=dataset,
        notes=notes or "promoted by deploy.py",
    )
    reg.set_best(version)
    print(f"[deploy] PROMOTED v{version} as new best (box={new_box:.4f} mask={new_mask:.4f})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="Path to new .pt weights.")
    ap.add_argument("--data", default=None, help="Path to data.yaml for live eval.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="Allowed mAP regression tolerance (default 0.005 = 0.5 pp).")
    ap.add_argument("--dataset", default="", help="Dataset description.")
    ap.add_argument("--notes", default="")
    ap.add_argument("--registry", default=None, help="Path to registry.json.")
    ap.add_argument("--per-class-gate", action="store_true",
                    help="Also require per-class gates (box mAP50 for "
                         "room/door/walls/acm, mask mAP50-95 for floor/stairs).")
    args = ap.parse_args()

    promoted = check_and_deploy(
        args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        tolerance=args.tolerance,
        registry_path=args.registry,
        dataset=args.dataset,
        notes=args.notes,
        per_class_gate=args.per_class_gate,
    )
    sys.exit(0 if promoted else 1)


if __name__ == "__main__":
    main()
