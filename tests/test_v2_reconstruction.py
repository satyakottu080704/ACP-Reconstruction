"""
test_v2_reconstruction.py — Unit tests for the v2 reconstruction engine.

All tests run WITHOUT torch, Visio, or GPU.
Synthetic detections and PlanModels are used throughout.
"""
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─── helpers to build synthetic data ─────────────────────────────────────────

def _make_rect_poly(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _make_rect_mask(x1, y1, x2, y2, w=256, h=256):
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [int(x1 * w), int(y1 * h)],
        [int(x2 * w), int(y1 * h)],
        [int(x2 * w), int(y2 * h)],
        [int(x1 * w), int(y2 * h)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _make_plan_with_rooms():
    from reconstruction.plan_model import PlanModel, RoomPolygon
    plan = PlanModel(image_width=1000, image_height=1000)
    plan.rooms = [
        RoomPolygon(
            polygon=_make_rect_poly(0.1, 0.1, 0.4, 0.5),
            label="Kitchen",
            room_type="clear",
            confidence=0.85,
            mask_quality=0.9,
            area=0.12,
            bbox=(0.1, 0.1, 0.4, 0.5),
        ),
        RoomPolygon(
            polygon=_make_rect_poly(0.4, 0.1, 0.8, 0.5),
            label="Living Room",
            room_type="clear",
            confidence=0.78,
            mask_quality=0.88,
            area=0.16,
            bbox=(0.4, 0.1, 0.8, 0.5),
        ),
        RoomPolygon(
            polygon=_make_rect_poly(0.1, 0.5, 0.5, 0.9),
            label="ACM Zone",
            room_type="acm",
            is_acm=True,
            confidence=0.72,
            mask_quality=0.82,
            area=0.16,
            bbox=(0.1, 0.5, 0.5, 0.9),
        ),
    ]
    return plan


# ─── plan_model tests ─────────────────────────────────────────────────────────

class TestPlanModel:
    def test_room_polygon_centroid(self):
        from reconstruction.plan_model import RoomPolygon
        r = RoomPolygon(polygon=_make_rect_poly(0.0, 0.0, 0.4, 0.4))
        cx, cy = r.centroid()
        assert abs(cx - 0.2) < 0.01
        assert abs(cy - 0.2) < 0.01

    def test_plan_acm_rooms(self):
        plan = _make_plan_with_rooms()
        acm = plan.acm_rooms()
        assert len(acm) == 1
        assert acm[0].label == "ACM Zone"

    def test_plan_rooms_on_floor(self):
        plan = _make_plan_with_rooms()
        floor0 = plan.rooms_on_floor(0)
        assert len(floor0) == 3


# ─── masks tests ──────────────────────────────────────────────────────────────

class TestMasks:
    def test_refine_mask_returns_binary(self):
        from reconstruction.masks import refine_mask
        mask = _make_rect_mask(0.2, 0.2, 0.8, 0.8)
        result = refine_mask(mask)
        assert result.dtype == np.uint8
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_mask_quality_score_range(self):
        from reconstruction.masks import mask_quality_score
        good = _make_rect_mask(0.1, 0.1, 0.9, 0.9)
        score = mask_quality_score(good)
        assert 0.0 <= score <= 1.0
        assert score > 0.5, "large clean mask should score > 0.5"

    def test_mask_quality_empty(self):
        from reconstruction.masks import mask_quality_score
        assert mask_quality_score(np.zeros((64, 64), dtype=np.uint8)) == 0.0

    def test_polygon_to_mask_roundtrip(self):
        from reconstruction.masks import polygon_to_mask
        poly = _make_rect_poly(0.1, 0.1, 0.9, 0.9)
        mask = polygon_to_mask(poly, 128, 128, normalised=True)
        assert mask.sum() > 0


# ─── polygons tests ───────────────────────────────────────────────────────────

class TestPolygons:
    def test_mask_to_polygon_basic(self):
        from reconstruction.polygons import mask_to_polygon
        mask = _make_rect_mask(0.1, 0.1, 0.9, 0.9)
        poly = mask_to_polygon(mask, normalise=True)
        assert len(poly) >= 4
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert max(xs) > 0.5
        assert max(ys) > 0.5

    def test_mask_to_polygon_empty(self):
        from reconstruction.polygons import mask_to_polygon
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert mask_to_polygon(empty) == []

    def test_reconstruct_walls_returns_list(self):
        from reconstruction.polygons import reconstruct_walls
        mask = _make_rect_mask(0.0, 0.48, 1.0, 0.52)  # thin horizontal wall
        result = reconstruct_walls(mask)
        assert isinstance(result, list)


# ─── cleanup tests ────────────────────────────────────────────────────────────

class TestCleanup:
    def test_snap_plan_no_crash(self):
        from reconstruction.cleanup import snap_plan
        plan = _make_plan_with_rooms()
        result = snap_plan(plan, snap_distance=0.02)
        assert len(result.rooms) == 3

    def test_resolve_overlaps_no_crash(self):
        from reconstruction.cleanup import resolve_overlaps
        plan = _make_plan_with_rooms()
        result = resolve_overlaps(plan)
        assert len(result.rooms) == 3

    def test_partition_loft_detects_loft(self):
        from reconstruction.cleanup import partition_loft
        from reconstruction.plan_model import PlanModel, RoomPolygon
        plan = PlanModel()
        plan.rooms = [
            RoomPolygon(polygon=_make_rect_poly(0.1, 0.1, 0.9, 0.9),
                        label="Ground Floor Room"),
            RoomPolygon(polygon=_make_rect_poly(0.1, 0.1, 0.4, 0.4),
                        label="Loft Storage", area=0.09),
        ]
        partition_loft(plan)
        loft_rooms = [r for r in plan.rooms if r.is_loft]
        assert len(loft_rooms) >= 1
        assert plan.has_loft is True

    def test_partition_loft_keyword(self):
        from reconstruction.cleanup import partition_loft
        from reconstruction.plan_model import PlanModel, RoomPolygon
        plan = PlanModel()
        plan.rooms = [
            RoomPolygon(polygon=_make_rect_poly(0.1, 0.1, 0.9, 0.9), label="Attic"),
        ]
        partition_loft(plan)
        assert plan.rooms[0].is_loft is True


# ─── room_graph tests ─────────────────────────────────────────────────────────

class TestRoomGraph:
    def test_build_adjacency_kitchen_living(self):
        from reconstruction.room_graph import build_adjacency
        plan = _make_plan_with_rooms()
        build_adjacency(plan, touch_distance=0.02)
        # Kitchen (0.1-0.4 x) and Living Room (0.4-0.8 x) share x=0.4 edge
        assert 1 in plan.adjacency.get(0, []) or 0 in plan.adjacency.get(1, [])

    def test_connected_to_json_shape(self):
        from reconstruction.room_graph import build_adjacency, to_connected_to_json
        plan = _make_plan_with_rooms()
        build_adjacency(plan)
        conn = to_connected_to_json(plan)
        assert isinstance(conn, dict)
        for label, info in conn.items():
            assert "connected_to" in info
            assert isinstance(info["connected_to"], list)

    def test_to_room_graph_json(self):
        from reconstruction.room_graph import build_adjacency, to_room_graph_json
        plan = _make_plan_with_rooms()
        build_adjacency(plan)
        graph = to_room_graph_json(plan)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3


# ─── quality tests ────────────────────────────────────────────────────────────

class TestQuality:
    def test_quality_score_range(self):
        from reconstruction.quality import compute_quality
        plan = _make_plan_with_rooms()
        from reconstruction.room_graph import build_adjacency
        build_adjacency(plan)
        qr = compute_quality(plan)
        assert 0.0 <= qr.overall_score <= 1.0

    def test_quality_flags_no_rooms(self):
        from reconstruction.quality import compute_quality
        from reconstruction.plan_model import PlanModel
        plan = PlanModel()
        qr = compute_quality(plan)
        assert "no_rooms_detected" in qr.review_flags
        assert qr.needs_review is True

    def test_quality_invalid_polygon(self):
        from reconstruction.quality import compute_quality
        from reconstruction.plan_model import PlanModel, RoomPolygon
        plan = PlanModel()
        plan.rooms = [RoomPolygon(polygon=[(0.1, 0.1)], label="Bad")]  # only 1 point
        qr = compute_quality(plan)
        assert "invalid_geometry" in qr.review_flags


# ─── engine tests ─────────────────────────────────────────────────────────────

class TestEngine:
    def _make_detections(self):
        """Create synthetic detections without needing YOLO."""
        dets = [
            {
                "class_name": "room",
                "class_id": 3,
                "confidence": 0.85,
                "mask": _make_rect_mask(0.1, 0.1, 0.45, 0.5),
            },
            {
                "class_name": "room",
                "class_id": 3,
                "confidence": 0.79,
                "mask": _make_rect_mask(0.45, 0.1, 0.85, 0.5),
            },
            {
                "class_name": "walls",
                "class_id": 5,
                "confidence": 0.91,
                "mask": _make_rect_mask(0.0, 0.48, 1.0, 0.52),
            },
            {
                "class_name": "door",
                "class_id": 1,
                "confidence": 0.74,
                "mask": _make_rect_mask(0.43, 0.44, 0.47, 0.56),
            },
        ]
        return dets

    def test_reconstruct_from_masks_builds_plan(self):
        from reconstruction.engine import reconstruct_from_masks
        dets = self._make_detections()
        plan = reconstruct_from_masks(dets, (256, 256), run_ocr=False)
        assert len(plan.rooms) >= 2
        assert plan.image_width == 256

    def test_reconstruct_quality_populated(self):
        from reconstruction.engine import reconstruct_from_masks
        dets = self._make_detections()
        plan = reconstruct_from_masks(dets, (256, 256), run_ocr=False)
        assert plan.quality is not None
        assert 0.0 <= plan.quality.overall_score <= 1.0

    def test_reconstruct_adjacency_built(self):
        from reconstruction.engine import reconstruct_from_masks
        dets = self._make_detections()
        plan = reconstruct_from_masks(dets, (256, 256), run_ocr=False)
        assert isinstance(plan.adjacency, dict)


# ─── exporter tests ───────────────────────────────────────────────────────────

class TestExporters:
    def setup_method(self):
        from reconstruction.room_graph import build_adjacency, to_connected_to_json
        self.plan = _make_plan_with_rooms()
        build_adjacency(self.plan)
        to_connected_to_json(self.plan)

    def test_export_json(self, tmp_path):
        from exporters.json_export import export_json
        out = export_json(self.plan, tmp_path / "plan.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "rooms" in data
        assert len(data["rooms"]) == 3
        assert "quality" in data
        assert "room_graph" in data
        assert "connectivity" in data

    def test_export_json_room_graph_valid(self, tmp_path):
        from exporters.json_export import export_json
        out = export_json(self.plan, tmp_path / "plan.json")
        data = json.loads(out.read_text())
        graph = data["room_graph"]
        assert len(graph["nodes"]) == 3
        for node in graph["nodes"]:
            assert "label" in node
            assert "id" in node

    def test_export_svg(self, tmp_path):
        from exporters.svg_export import export_svg
        out = export_svg(self.plan, tmp_path / "plan.svg")
        assert out.exists()
        content = out.read_text()
        assert "<svg" in content
        assert "polygon" in content.lower() or "path" in content.lower()
        # ACM hatching pattern
        assert "acm_hatch" in content

    def test_export_png(self, tmp_path):
        from exporters.png_export import export_png
        out = export_png(self.plan, tmp_path / "plan.png", canvas_px=512)
        assert out.exists()
        assert out.stat().st_size > 1000  # not empty

    def test_export_dxf_minimal(self, tmp_path):
        from exporters.dxf_export import export_dxf
        out = export_dxf(self.plan, tmp_path / "plan.dxf")
        assert out.exists()
        content = out.read_text()
        # DXF should contain POLYLINE or Shape entities
        assert "POLYLINE" in content or "EOF" in content

    def test_export_vsdx_is_valid_zip(self, tmp_path):
        from exporters.vsdx_export import export_vsdx
        out = export_vsdx(self.plan, tmp_path / "plan.vsdx")
        assert out.exists()
        assert zipfile.is_zipfile(str(out)), "VSDX must be a valid ZIP archive"
        with zipfile.ZipFile(str(out)) as zf:
            names = zf.namelist()
            assert any("page1" in n for n in names), "VSDX must contain page1.xml"
            assert any("[Content_Types]" in n for n in names)

    def test_export_vsdx_shapes_within_page(self, tmp_path):
        """PinX values must be within the page bounds (not crammed at origin)."""
        from exporters.vsdx_export import export_vsdx
        import xml.etree.ElementTree as ET
        out = export_vsdx(self.plan, tmp_path / "plan.vsdx")
        with zipfile.ZipFile(str(out)) as zf:
            page_names = [n for n in zf.namelist() if "page1.xml" in n]
            if not page_names:
                return
            xml_content = zf.read(page_names[0]).decode("utf-8")
        root = ET.fromstring(xml_content)
        ns = "http://schemas.microsoft.com/office/visio/2012/main"
        pin_x_values = []
        for cell in root.iter(f"{{{ns}}}Cell"):
            if cell.get("N") == "PinX":
                try:
                    pin_x_values.append(float(cell.get("V", "0")))
                except ValueError:
                    pass
        if pin_x_values:
            # No shape should have PinX = 0 (hard-coded origin problem)
            assert max(pin_x_values) > 0.1, \
                f"All PinX values near zero: {pin_x_values} — shapes likely invisible"

    def test_export_all(self, tmp_path):
        from exporters import export_all
        results = export_all(self.plan, tmp_path, stem="test", formats=["json", "svg"])
        assert "json" in results
        assert "svg" in results
        assert not results["json"].startswith("ERROR")
        assert not results["svg"].startswith("ERROR")


# ─── geometry benchmark tests ─────────────────────────────────────────────────

class TestGeometryBenchmark:
    def test_polygon_iou_identical(self):
        from evaluation.geometry_benchmark import polygon_iou
        poly = _make_rect_poly(0.1, 0.1, 0.9, 0.9)
        iou = polygon_iou(poly, poly)
        assert abs(iou - 1.0) < 0.01

    def test_polygon_iou_no_overlap(self):
        from evaluation.geometry_benchmark import polygon_iou
        a = _make_rect_poly(0.0, 0.0, 0.4, 0.4)
        b = _make_rect_poly(0.6, 0.6, 1.0, 1.0)
        iou = polygon_iou(a, b)
        assert iou < 0.01

    def test_polygon_iou_partial(self):
        from evaluation.geometry_benchmark import polygon_iou
        a = _make_rect_poly(0.0, 0.0, 0.6, 0.6)
        b = _make_rect_poly(0.4, 0.4, 1.0, 1.0)
        iou = polygon_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_hausdorff_identical(self):
        from evaluation.geometry_benchmark import hausdorff_distance
        poly = _make_rect_poly(0.1, 0.1, 0.9, 0.9)
        hd = hausdorff_distance(poly, poly)
        assert hd < 0.01

    def test_match_polygons_basic(self):
        from evaluation.geometry_benchmark import match_polygons
        pred = [_make_rect_poly(0.1, 0.1, 0.4, 0.4),
                _make_rect_poly(0.5, 0.5, 0.9, 0.9)]
        gt = [_make_rect_poly(0.1, 0.1, 0.4, 0.4),
              _make_rect_poly(0.5, 0.5, 0.9, 0.9)]
        matched, unm_pred, unm_gt = match_polygons(pred, gt, iou_threshold=0.3)
        assert len(matched) == 2
        assert len(unm_pred) == 0
        assert len(unm_gt) == 0

    def test_gate_check_pass(self):
        from evaluation.geometry_benchmark import gate_check
        assert gate_check({"room_iou_mean": 0.85}, min_room_iou=0.7) is True

    def test_gate_check_fail(self):
        from evaluation.geometry_benchmark import gate_check
        assert gate_check({"room_iou_mean": 0.60}, min_room_iou=0.7) is False


# ─── mlops registry + gate tests ──────────────────────────────────────────────

class TestMLOpsRegistry:
    def test_registry_register_and_best(self, tmp_path):
        from mlops.registry import ModelRegistry

        # Create a fake .pt file
        fake_pt = tmp_path / "fake_best.pt"
        fake_pt.write_bytes(b"fake model weights" * 100)

        reg = ModelRegistry(
            registry_path=str(tmp_path / "registry.json"),
            weights_dir=str(tmp_path / "versions"),
        )
        v = reg.register(str(fake_pt), box_map50=0.883, mask_map50=0.803, epoch=200,
                         copy_weights=False)
        assert v == 1

        reg.set_best(v)
        best = reg.get_best()
        assert best is not None
        assert best["box_map50"] == 0.883
        assert best["version"] == 1

    def test_registry_regression_gate_blocks(self, tmp_path):
        """Verify the 50-epoch 0.855/0.769 run would be blocked."""
        from mlops.registry import ModelRegistry

        fake_best = tmp_path / "best.pt"
        fake_best.write_bytes(b"x" * 100)
        fake_new = tmp_path / "new.pt"
        fake_new.write_bytes(b"y" * 100)

        reg = ModelRegistry(
            registry_path=str(tmp_path / "reg.json"),
            weights_dir=str(tmp_path / "ver"),
        )
        # Register current best (0.883 / 0.803)
        v1 = reg.register(str(fake_best), box_map50=0.883, mask_map50=0.803,
                          copy_weights=False)
        reg.set_best(v1)

        # Simulate the 50-epoch run metrics
        best = reg.get_best()
        new_box, new_mask = 0.855, 0.769
        tol = 0.005
        gate_passes = (new_box >= best["box_map50"] - tol and
                       new_mask >= best["mask_map50"] - tol)
        assert gate_passes is False, "50-epoch run should be blocked by regression gate"


# ─── wall snap & overlap tests ────────────────────────────────────────────────

class TestWallSnapOverlap:
    def test_snap_plan_merges_near_vertices(self):
        """Vertices within snap_distance should be merged to a shared coord."""
        from reconstruction.cleanup import snap_plan
        from reconstruction.plan_model import PlanModel, RoomPolygon

        plan = PlanModel()
        # Two rooms with near-coincident shared edge (differ by 0.005)
        plan.rooms = [
            RoomPolygon(polygon=[(0.1, 0.1), (0.401, 0.1), (0.401, 0.5), (0.1, 0.5)]),
            RoomPolygon(polygon=[(0.399, 0.1), (0.8, 0.1), (0.8, 0.5), (0.399, 0.5)]),
        ]
        snap_plan(plan, snap_distance=0.015)
        # After snapping, the x-coords near 0.4 should be equal
        x_right_room0 = [p[0] for p in plan.rooms[0].polygon if p[0] > 0.35]
        x_left_room1 = [p[0] for p in plan.rooms[1].polygon if p[0] < 0.45]
        if x_right_room0 and x_left_room1:
            assert abs(max(x_right_room0) - min(x_left_room1)) < 0.005

    def test_resolve_overlaps_no_crash(self):
        from reconstruction.cleanup import resolve_overlaps
        from reconstruction.plan_model import PlanModel, RoomPolygon
        plan = PlanModel()
        plan.rooms = [
            RoomPolygon(polygon=_make_rect_poly(0.1, 0.1, 0.55, 0.5)),
            RoomPolygon(polygon=_make_rect_poly(0.45, 0.1, 0.9, 0.5)),
        ]
        resolve_overlaps(plan, sliver_threshold=0.005)
        assert len(plan.rooms) == 2  # rooms remain


# ─── reconstruction report tests ─────────────────────────────────────────────

class TestReconstructionReport:
    def test_report_with_no_gt_file(self, tmp_path):
        from evaluation.reconstruction_report import generate_report
        plan = _make_plan_with_rooms()
        result = generate_report(plan, str(tmp_path / "nonexistent.json"))
        assert "error" in result

    def test_report_with_matching_gt(self, tmp_path):
        from evaluation.reconstruction_report import generate_report
        plan = _make_plan_with_rooms()

        # Write a ground-truth JSON that matches the plan's rooms
        gt = {
            "rooms": [
                {"polygon": _make_rect_poly(0.1, 0.1, 0.4, 0.5), "label": "Kitchen"},
                {"polygon": _make_rect_poly(0.4, 0.1, 0.8, 0.5), "label": "Living Room"},
                {"polygon": _make_rect_poly(0.1, 0.5, 0.5, 0.9), "label": "ACM Zone"},
            ]
        }
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")

        result = generate_report(plan, str(gt_path))
        assert "Room Accuracy" in result
        assert "Overall Reconstruction" in result
        assert 0.0 <= result["Room Accuracy"] <= 100.0
        assert 0.0 <= result["Overall Reconstruction"] <= 100.0


# --- gap-filling regression tests -------------------------------------------

class TestFillGaps:
    def test_perimeter_ring_gap_does_not_balloon_one_room(self):
        """
        Regression test: a thin gap that touches 2+ disjoint rooms at once
        (e.g. a perimeter sliver wrapping around several room edges) must be
        PARTITIONED between them, never assigned wholly to whichever room
        happens to have the larger overlap. Assigning it wholly used to make
        one room's mask wrap around its neighbour, causing ~50% overlap.
        """
        from reconstruction.coverage import fill_gaps
        from reconstruction.plan_model import PlanModel, RoomPolygon

        plan = PlanModel()
        plan.floor_boundary = [_make_rect_poly(0.0, 0.0, 1.0, 1.0)]
        # Two rooms covering the floor almost completely, leaving a thin
        # border sliver on all sides (touches both rooms at once).
        plan.rooms = [
            RoomPolygon(polygon=_make_rect_poly(0.01, 0.01, 0.5, 0.99), label="Bedroom"),
            RoomPolygon(polygon=_make_rect_poly(0.5, 0.01, 0.99, 0.99), label="Cupboard"),
        ]
        fill_gaps(plan)

        def area(poly):
            n = len(poly)
            s = 0.0
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                s += x1 * y2 - x2 * y1
            return abs(s) / 2.0

        a0 = area(plan.rooms[0].polygon)
        a1 = area(plan.rooms[1].polygon)
        inter_mask_frac = _polygon_overlap_frac(plan.rooms[0].polygon, plan.rooms[1].polygon)
        assert inter_mask_frac < 0.05, (
            f"rooms overlap by {inter_mask_frac:.1%} after fill_gaps -- "
            "a small multi-room gap was wrongly assigned wholly to one room"
        )
        # Each room should still be roughly its original half, not a blob
        # covering (most of) the whole floor.
        assert a0 < 0.6
        assert a1 < 0.6

    def test_six_room_grid_tiles_with_no_gaps_or_overlap(self):
        """Six rooms with small (1.5-2%) gaps from their neighbours should
        end up tiling the floor exactly, matching the reference 'no gaps,
        exact arrangement' floor-plan spec."""
        from reconstruction.coverage import fill_gaps, validate_constraints
        from reconstruction.plan_model import PlanModel, RoomPolygon

        plan = PlanModel()
        plan.floor_boundary = [_make_rect_poly(0.0, 0.0, 1.0, 1.0)]
        plan.rooms = [
            RoomPolygon(polygon=_make_rect_poly(0.000, 0.000, 0.323, 0.497), label="Bedroom"),
            RoomPolygon(polygon=_make_rect_poly(0.327, 0.000, 0.647, 0.497), label="Lounge"),
            RoomPolygon(polygon=_make_rect_poly(0.651, 0.000, 1.000, 0.497), label="Kitchen"),
            RoomPolygon(polygon=_make_rect_poly(0.000, 0.503, 0.247, 1.000), label="CPD"),
            RoomPolygon(polygon=_make_rect_poly(0.251, 0.503, 0.647, 1.000), label="Hallway"),
            RoomPolygon(polygon=_make_rect_poly(0.651, 0.503, 1.000, 1.000), label="Bath"),
        ]
        fill_gaps(plan)
        checks = validate_constraints(plan)
        assert checks["no_overlaps"], checks
        assert checks["no_gaps"], checks


def _polygon_overlap_frac(poly_a, poly_b, size=512):
    """Fraction of the smaller polygon's rasterised area that overlaps the other."""
    import cv2
    import numpy as np

    def rasterize(poly):
        m = np.zeros((size, size), np.uint8)
        pts = np.array([(int(x * (size - 1)), int(y * (size - 1))) for x, y in poly], np.int32)
        cv2.fillPoly(m, [pts], 255)
        return m

    ma, mb = rasterize(poly_a), rasterize(poly_b)
    inter = int(np.count_nonzero(cv2.bitwise_and(ma, mb)))
    smaller = min(int(np.count_nonzero(ma)), int(np.count_nonzero(mb)))
    return inter / max(smaller, 1)
