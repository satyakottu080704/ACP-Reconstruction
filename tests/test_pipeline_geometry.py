"""
test_pipeline_geometry.py — Regression tests for the legacy pipeline.py
geometry post-processing (the path main.py --image actually runs).

These cover a real, reproduced defect: a batch run on Input/1.jpg (a genuine
Acorn "Hats Close" survey sketch) produced a 1.vsdx where:
  - "007 Store Room"'s box sat almost entirely inside "005 Bedroom"'s box
    (both kept their full overlapping rectangles -> double-counted floor
    area and one room's boundary drawn through the middle of the other).
  - An unlabeled "010 Room 10" box floated above the whole building outline,
    touching nothing, with no GPT-4o label match and no doors -- a phantom
    room.
  - Both "001 Lounge" and "007 Store Room" rendered ACM-positive (red fill)
    even though only the sketch's Loft actually has diagonal hatching --
    traced to the GPT-4o prompt treating ordinary red-underlined sample
    callouts (e.g. "S001 TC") near a room as proof of ACM hatching.

These tests exercise the two geometry fixes (pipeline._resolve_room_box_
overlaps, pipeline._drop_isolated_unmatched_rooms) using the exact
coordinates recovered from that .vsdx. They run WITHOUT torch/openai/Visio
-- pipeline.py only needs cv2/numpy at import time.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pipeline as p  # noqa: E402


def _overlap_area(a, b):
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


class TestResolveRoomBoxOverlaps:
    def test_nested_store_room_no_longer_overlaps_bedroom(self):
        """Reproduces the exact 1.vsdx overlap: Store Room's box sat mostly
        inside Bedroom's box. After the fix, neither room's rectangle
        should intersect the other's at all."""
        store = p.Room(bbox=(786, 402, 139, 180), area=139 * 180,
                        label="Store Room", number="007")
        bed = p.Room(bbox=(792, 216, 482, 366), area=482 * 366,
                      label="Bedroom", number="005")
        lounge = p.Room(bbox=(382, 216, 403, 595), area=403 * 595,
                         label="Lounge", number="001")
        rooms = [store, bed, lounge]

        p._resolve_room_box_overlaps(rooms, sketch_w=1655, sketch_h=1169)

        assert _overlap_area(store, bed) == 0
        assert _overlap_area(store, lounge) == 0
        assert _overlap_area(bed, lounge) == 0
        # Neither room collapsed to nothing.
        for r in rooms:
            assert r.bbox[2] > 0 and r.bbox[3] > 0

    def test_non_overlapping_rooms_untouched(self):
        """Rooms that don't overlap at all must not be modified."""
        a = p.Room(bbox=(0, 0, 100, 100), area=10000, label="A", number="001")
        b = p.Room(bbox=(200, 200, 100, 100), area=10000, label="B", number="002")
        rooms = [a, b]
        before = [tuple(r.bbox) for r in rooms]
        p._resolve_room_box_overlaps(rooms, sketch_w=1000, sketch_h=1000)
        after = [tuple(r.bbox) for r in rooms]
        assert before == after

    def test_incidental_small_overlap_left_alone(self):
        """A tiny overlap (wall-thickness noise, well under the 15% of the
        smaller room's area threshold) should not trigger a clip -- avoids
        needlessly nibbling legitimately-adjacent rooms."""
        a = p.Room(bbox=(0, 0, 200, 200), area=40000, label="A", number="001")
        b = p.Room(bbox=(198, 0, 200, 200), area=40000, label="B", number="002")
        rooms = [a, b]
        before = [tuple(r.bbox) for r in rooms]
        p._resolve_room_box_overlaps(rooms, sketch_w=1000, sketch_h=1000)
        after = [tuple(r.bbox) for r in rooms]
        assert before == after


class TestDropIsolatedUnmatchedRooms:
    def test_floating_unmatched_room_is_dropped(self):
        """Reproduces the 1.vsdx '010 Room 10' phantom: an unmatched model
        box floating above the building, touching no other room."""
        hall = p.Room(bbox=(388, 811, 125, 253), area=125 * 253,
                       label="Hall", number="006")
        kitchen = p.Room(bbox=(513, 823, 284, 247), area=284 * 247,
                          label="Kitchen", number="002")
        orphan = p.Room(bbox=(446, 120, 206, 96), area=206 * 96,
                         label="Room 10", number="010")
        rooms = [hall, kitchen, orphan]

        p._drop_isolated_unmatched_rooms(rooms, {2}, sketch_w=1655, sketch_h=1169)

        assert len(rooms) == 2
        assert all(r.label != "Room 10" for r in rooms)

    def test_unmatched_room_touching_a_neighbour_is_kept(self):
        """An unmatched box that genuinely shares a wall with another room
        must be kept -- only fully isolated boxes are dropped."""
        hall = p.Room(bbox=(388, 811, 125, 253), area=125 * 253,
                       label="Hall", number="006")
        adjacent = p.Room(bbox=(388, 700, 125, 111), area=125 * 111,
                           label="Room X", number="099")
        rooms = [hall, adjacent]

        p._drop_isolated_unmatched_rooms(rooms, {1}, sketch_w=1655, sketch_h=1169)

        assert len(rooms) == 2

    def test_matched_rooms_never_dropped_even_if_isolated(self):
        """A room WITH a GPT-4o label match must never be dropped by this
        pass, even if it happens to be geometrically isolated -- only
        unmatched (unmatched_indices) boxes are candidates for removal."""
        named_isolated = p.Room(bbox=(900, 10, 50, 50), area=2500,
                                 label="Porch", number="011")
        hall = p.Room(bbox=(388, 811, 125, 253), area=125 * 253,
                       label="Hall", number="006")
        rooms = [named_isolated, hall]

        p._drop_isolated_unmatched_rooms(rooms, set(), sketch_w=1655, sketch_h=1169)

        assert len(rooms) == 2
