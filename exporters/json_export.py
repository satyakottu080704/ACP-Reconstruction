"""
json_export.py — export PlanModel to JSON (includes room graph + connectivity).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from reconstruction.plan_model import PlanModel
from reconstruction.room_graph import to_room_graph_json, to_connected_to_json
from reconstruction.quality import quality_report_dict


def export_json(plan: PlanModel, out_path: Union[str, Path]) -> Path:
    """
    Write the plan to a JSON file.

    Structure:
    {
        "meta": {...},
        "rooms": [...],
        "walls": [...],
        "doors": [...],
        "stairs": [...],
        "samples": [...],
        "room_graph": {"nodes": [...], "edges": [...]},
        "connectivity": {"Living Room": {"connected_to": [...]}, ...},
        "quality": {...}
    }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Room graph
    graph = to_room_graph_json(plan)
    connectivity = to_connected_to_json(plan)

    data = {
        "meta": {
            "engine_version": plan.engine_version,
            "image_width": plan.image_width,
            "image_height": plan.image_height,
            "source_image": plan.source_image,
            "has_loft": plan.has_loft,
            "floor_labels": plan.floor_labels,
        },
        "rooms": [
            {
                "id": i,
                "label": r.label,
                "number": r.number,
                "no_access": r.no_access,
                "room_type": r.room_type,
                "floor_idx": r.floor_idx,
                "floor_label": r.floor_label,
                "is_acm": r.is_acm,
                "is_loft": r.is_loft,
                "confidence": r.confidence,
                "mask_quality": r.mask_quality,
                "ocr_text": r.ocr_text,
                "ocr_confidence": r.ocr_confidence,
                "area": r.area,
                "bbox": list(r.bbox),
                "centroid": list(r.centroid()),
                "polygon": [list(pt) for pt in r.polygon],
            }
            for i, r in enumerate(plan.rooms)
        ],
        "walls": [
            {
                "id": i,
                "points": [list(pt) for pt in w.points],
                "thickness": w.thickness,
                "is_exterior": w.is_exterior,
            }
            for i, w in enumerate(plan.walls)
        ],
        "doors": [
            {
                "id": i,
                "center": list(d.center),
                "width": d.width,
                "angle_deg": d.angle_deg,
                "room_a": d.room_a,
                "room_b": d.room_b,
                "confidence": d.confidence,
            }
            for i, d in enumerate(plan.doors)
        ],
        "stairs": [
            {
                "id": i,
                "polygon": [list(pt) for pt in s.polygon],
                "direction_deg": s.direction_deg,
                "floor_idx": s.floor_idx,
                "label": s.label,
                "confidence": s.confidence,
            }
            for i, s in enumerate(plan.stairs)
        ],
        "samples": [
            {
                "id": s.sample_id,
                "material": s.material,
                "text": s.text,
                "is_ref": s.is_ref,
                "acm_positive": s.acm_positive,
                "target": list(s.target),
                "room_idx": s.room_idx,
                "floor_idx": s.floor_idx,
            }
            for s in plan.samples
        ],
        "adjacency": {str(k): v for k, v in plan.adjacency.items()},
        "room_graph": graph,
        "connectivity": connectivity,
        "quality": quality_report_dict(plan),
    }

    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
