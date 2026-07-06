"""
store.py — auto-save low-quality predictions for re-annotation.

When the quality report flags needs_review=True OR overall_score < threshold,
the source image + prediction JSON are saved to ACTIVE_LEARNING_DIR
for a human annotator to review and correct.

This builds up a dataset of difficult / ambiguous sketches that are most
valuable for improving the model (active learning).

Usage:
  from active_learning.store import maybe_store
  maybe_store(plan, source_image_bgr, image_name="survey_001.jpg")

Directory structure:
  ACTIVE_LEARNING_DIR/
    queue/
      survey_001/
        image.jpg
        prediction.json
        quality.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def maybe_store(
    plan,
    source_image: Optional[np.ndarray],
    image_name: str = "unknown.jpg",
    quality_threshold: float = 0.75,
    store_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Save a prediction to the active learning queue if it needs review.

    Args:
        plan:              PlanModel from the v2 engine.
        source_image:      BGR numpy array (the sketch image).
        image_name:        original filename (used for directory naming).
        quality_threshold: auto-save if overall_score < this value.
        store_dir:         override for ACTIVE_LEARNING_DIR.

    Returns:
        Path to the saved directory, or None if not saved.
    """
    qr = plan.quality
    should_store = qr.needs_review or qr.overall_score < quality_threshold

    if not should_store:
        return None

    if store_dir is None:
        try:
            import sys
            from pathlib import Path as _P
            _root = _P(__file__).resolve().parents[1]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            import config as _cfg
            store_dir = _cfg.ACTIVE_LEARNING_DIR
        except Exception:
            store_dir = str(Path(__file__).resolve().parents[1] / "active_learning" / "queue")

    stem = Path(image_name).stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sample_dir = Path(store_dir) / f"{ts}_{stem}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    if source_image is not None and source_image.size > 0:
        cv2.imwrite(str(sample_dir / "image.jpg"), source_image)

    # Save prediction JSON
    try:
        from exporters.json_export import export_json
        export_json(plan, sample_dir / "prediction.json")
    except Exception as e:
        # Fallback: minimal JSON
        minimal = {
            "room_count": len(plan.rooms),
            "quality": {
                "overall_score": qr.overall_score,
                "review_flags": qr.review_flags,
            },
        }
        (sample_dir / "prediction.json").write_text(
            json.dumps(minimal, indent=2), encoding="utf-8"
        )

    # Save quality report
    try:
        from reconstruction.quality import quality_report_dict
        qd = quality_report_dict(plan)
    except Exception:
        qd = {"overall_score": qr.overall_score, "needs_review": qr.needs_review}

    (sample_dir / "quality.json").write_text(
        json.dumps(qd, indent=2), encoding="utf-8"
    )

    print(f"[active_learning] stored sample → {sample_dir} "
          f"(score={qr.overall_score:.3f}, flags={qr.review_flags})")
    return sample_dir


def queue_size(store_dir: Optional[str] = None) -> int:
    """Return the number of samples currently in the queue."""
    if store_dir is None:
        try:
            import config as _cfg
            store_dir = _cfg.ACTIVE_LEARNING_DIR
        except Exception:
            store_dir = str(Path(__file__).resolve().parents[1] / "active_learning" / "queue")

    p = Path(store_dir)
    if not p.exists():
        return 0
    return sum(1 for d in p.iterdir() if d.is_dir())
