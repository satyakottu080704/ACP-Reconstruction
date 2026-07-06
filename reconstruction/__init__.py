"""
reconstruction — v2 mask-first floor-plan geometry platform.

Gated by RECONSTRUCTION_ENGINE=v2 (default: "legacy").
All modules import gracefully on OpenCV + NumPy alone;
Shapely / NetworkX / ezdxf are optional.
"""
from .plan_model import (
    PlanModel, RoomPolygon, WallSegment, DoorSegment, StairPolygon,
    SampleAnnotation, from_legacy_floorplan,
)
from .engine import reconstruct_from_masks, from_yolo_result
from .postprocess import apply_labels

__all__ = [
    "PlanModel", "RoomPolygon", "WallSegment", "DoorSegment", "StairPolygon",
    "SampleAnnotation", "from_legacy_floorplan", "apply_labels",
    "reconstruct_from_masks", "from_yolo_result",
]
