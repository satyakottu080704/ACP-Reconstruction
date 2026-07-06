"""
Professional Visio Floor Plan Generator
=========================================
Generates survey-standard .vsdx floor plans with:
- Thin black wall lines (~1pt) with door gaps
- Room numbers (001, 002...) with labels
- Door arcs (quarter-circle swing indicators)
- Sample annotations outside rooms with red arrows
- ACM fill (pink) for positive rooms, white for clear
- Floor titles, stair hatching, loft X-marks
- Utility markers (ATM, DB, Gas, Water)
- Cable route visualization
- Legend and caveat annotations

Requires: Microsoft Visio + pywin32
"""

import os
import math
import time
import gc
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _draw_diagonal_hatch(page, x1: float, y1: float, x2: float, y2: float,
                          color_rgb: str = "RGB(196,60,70)",
                          spacing: float = 0.14,
                          weight: str = "0.5 pt") -> None:
    """Draw evenly-spaced 45-degree diagonal hatch lines clipped to a
    rectangle -- this is how Acorn's own manual plans mark a room with a
    lab-confirmed positive ACM sample: a light diagonal hatch of individual
    line strokes, NOT a solid colour fill. (Verified against real Acorn
    survey outputs -- solid fill does not appear in any of them.)
    """
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0 or spacing <= 0:
        return
    diag_count = int((width + height) / spacing) + 2
    for k in range(diag_count):
        ox = -height + k * spacing
        lx1, ly1 = x1 + ox, y1
        lx2, ly2 = x1 + ox + height, y2
        # Clip the 45-degree segment to the rectangle's x-range.
        if lx1 < x1:
            dy = x1 - lx1
            lx1, ly1 = x1, y1 + dy
        if lx2 > x2:
            dy = lx2 - x2
            lx2, ly2 = x2, y2 - dy
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        line = page.DrawLine(lx1, ly1, lx2, ly2)
        line.Cells("LineColor").FormulaU = color_rgb
        line.Cells("LineWeight").FormulaU = weight
        line.Cells("LinePattern").FormulaU = "1"


def _resolve_template_path() -> str:
    """Locate the Visio template without depending on a personal machine path.

    Order: ``VISIO_TEMPLATE_PATH`` env override, then the repo-local skeleton
    ``utils/visio/template.vsdx``. Returns the first existing path, or the
    repo-local path (callers check existence and fall back to a blank doc).
    """
    env_path = os.environ.get("VISIO_TEMPLATE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    return os.path.join(os.path.dirname(__file__), "template.vsdx")


def _room_is_explicit_loft(room: Dict[str, Any]) -> bool:
    """Return True only when this room itself is identified as loft/attic."""
    label = str(room.get("label") or "").lower()
    floor = str(room.get("floor") or "").lower()
    orig_floor = str(room.get("_orig_floor_name") or "").lower()
    return any(word in text for text in (label, floor, orig_floor)
               for word in ("loft", "attic"))


def _partition_loft_rooms(
    rooms: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate loft rooms without moving unrelated rooms that share an AI floor index."""
    loft_rooms = [room for room in rooms if _room_is_explicit_loft(room)]
    main_rooms = [room for room in rooms if room not in loft_rooms]
    return main_rooms, loft_rooms


def _landscape_dimensions(width: float, height: float) -> Tuple[float, float]:
    """Return page dimensions in landscape orientation."""
    return (width, height) if width >= height else (height, width)


def _conservative_gap_threshold(room_rects, scale_mult=1.0):
    """Maximum Visio-space gap that may be treated as detection noise."""
    short_sides = sorted(
        min(abs(rect[2] - rect[0]), abs(rect[3] - rect[1]))
        for rect in room_rects
        if abs(rect[2] - rect[0]) > 0 and abs(rect[3] - rect[1]) > 0
    )
    if not short_sides:
        return 0.12 * scale_mult
    median_short = short_sides[len(short_sides) // 2]
    return max(0.04 * scale_mult, min(0.30 * scale_mult, median_short * 0.10))


def _clamp_point_to_bounds(x, y, bounds):
    """Keep an annotation target inside the reconstructed plan footprint."""
    x1, y1, x2, y2 = bounds
    return max(x1, min(x, x2)), max(y1, min(y, y2))


def _remove_off_page_template_shapes(page) -> int:
    """Delete the template's stencil/palette shapes that sit outside the page."""
    try:
        page_width = page.PageSheet.Cells("PageWidth").ResultIU
        page_height = page.PageSheet.Cells("PageHeight").ResultIU
    except Exception:
        return 0

    removed = 0
    for index in range(page.Shapes.Count, 0, -1):
        shape = page.Shapes.Item(index)
        try:
            pin_x = shape.Cells("PinX").ResultIU
            pin_y = shape.Cells("PinY").ResultIU
            width = abs(shape.Cells("Width").ResultIU)
            height = abs(shape.Cells("Height").ResultIU)
        except Exception:
            continue
        outside = (
            pin_x + width / 2 < 0
            or pin_y + height / 2 < 0
            or pin_x - width / 2 > page_width
            or pin_y - height / 2 > page_height
        )
        if outside:
            shape.Delete()
            removed += 1
    return removed


def generate_visio_from_detected(
    detected: Dict[str, Any],
    project_number: str,
    output_path: str = None,
    client_type: str = "standard",
    detection_hints: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Generate professional Visio floor plan matching Acorn survey standards.

    Multi-floor: when rooms carry a `floor_idx` and `detected['floors']` has
    more than one entry, generate ONE PAGE PER FLOOR in the same .vsdx file.
    Page names use `detected['floors'][i]['title']` (e.g. "Ground Floor").
    Pages are ordered by floor_idx ascending (Ground -> First -> Loft).

    Matches existing professional plans (N-72131, N-73337, N-73440 etc.):
    - Thin black wall lines (~1pt) with door gaps
    - Room numbers (001, 002...) with name below
    - Door arcs (quarter-circle swing indicators)
    - Sample annotations OUTSIDE rooms with red arrows pointing in
    - Muted red ACM fill, blue no-access fill, white clear fill
    - Underlined floor title top-left ("Ground Floor:")
    - Stair hatching and loft X-marks
    """
    try:
        import win32com.client
        import pythoncom
    except ImportError:
        print("[VISIO] win32com not available - cannot generate Visio files")
        return None

    all_rooms = detected.get("rooms", [])
    if not all_rooms:
        print("[VISIO] No rooms to render")
        return None

    if not output_path:
        output_dir = Path(__file__).parent.parent / "output" / "generated_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{project_number}_Generated.vsdx")

    hints = detection_hints or {}
    # ---- Group rooms by floor ----
    import os
    template_path = _resolve_template_path()

    # Determine page grouping mode.
    # We support:
    # 1. "split_loft": main floors in 1 tab/page, Loft in its own tab/page
    # 2. "single_page": all floors in 1 tab/page
    # 3. "multi_page": separate tab/page per floor
    # Never combine separate surveyed floors by default. Their coordinates
    # are relative to different drawings and become misleading when rendered
    # together on one page.
    page_mode = "multi_page"
    if hints.get("multi_page") is True:
        page_mode = "multi_page"
    elif hints.get("split_loft") is True:
        page_mode = "split_loft"
    elif hints.get("single_page") is True or hints.get("use_single_page") is True:
        page_mode = "single_page"
    elif hints.get("use_single_page") is False:
        page_mode = "multi_page"

    if page_mode == "single_page":
        print("[VISIO] Grouping all floors onto a single page (single_page mode)")
        floor_order = [(0, "Floor Plans")]
        _DEFAULT_FLOOR_NAMES = {0: "Ground Floor", 1: "First Floor",
                                2: "Second Floor", 3: "Loft"}
        for r in all_rooms:
            orig_idx = int(r.get("floor_idx", 0) or 0)
            r["_orig_floor_idx"] = orig_idx
            r["_orig_floor_name"] = r.get("floor", "")
            r["floor_idx"] = 0  # one page
            # Keep the original floor NAME so each floor renders as its own
            # labelled section ("Ground Floor:", "Loft:") and the floor-
            # separation logic below keeps the sections from overlapping.
            if not r.get("floor"):
                r["floor"] = _DEFAULT_FLOOR_NAMES.get(orig_idx, f"Floor {orig_idx}")
        floor_groups = {0: all_rooms}
    elif page_mode == "split_loft":
        # A loft room gets its own tab even if AI assigned the same floor_idx
        # to neighbouring rooms. Never promote an entire floor based only on
        # one room's label.
        main_rooms, loft_rooms = _partition_loft_rooms(all_rooms)
        
        floor_groups = {}
        floor_order = []
        
        if main_rooms:
            main_idx = 0
            floor_order.append((main_idx, "Floor Plans"))
            _DEFAULT_FLOOR_NAMES = {0: "Ground Floor", 1: "First Floor",
                                    2: "Second Floor", 3: "Loft"}
            for r in main_rooms:
                orig_idx = int(r.get("floor_idx", 0) or 0)
                r["_orig_floor_idx"] = orig_idx
                r["_orig_floor_name"] = r.get("floor", "")
                r["floor_idx"] = main_idx  # all main floors share one page
                # Keep the original floor NAME so each renders as its own
                # labelled section on the shared tab (the loft is on its own tab).
                if not r.get("floor"):
                    r["floor"] = _DEFAULT_FLOOR_NAMES.get(orig_idx, f"Floor {orig_idx}")
            floor_groups[main_idx] = main_rooms
            
        if loft_rooms:
            loft_idx = 1 if main_rooms else 0
            floor_order.append((loft_idx, "Loft"))
            for r in loft_rooms:
                r["_orig_floor_idx"] = int(r.get("floor_idx", 0) or 0)
                r["_orig_floor_name"] = r.get("floor", "")
                r["floor_idx"] = loft_idx
                r["floor"] = "Loft"
            floor_groups[loft_idx] = loft_rooms
            
        print(f"[VISIO] Grouping floors using split_loft mode. Main floors count: {len(main_rooms)}, Loft floors count: {len(loft_rooms)}")
    else: # multi_page
        print("[VISIO] Grouping floors into separate tabs (multi_page mode)")
        floor_groups = {}
        for r in all_rooms:
            idx = int(r.get("floor_idx", 0) or 0)
            r["_orig_floor_idx"] = idx
            r["_orig_floor_name"] = r.get("floor", "")
            floor_groups.setdefault(idx, []).append(r)

        floor_defs = detected.get("floors") or []
        if floor_defs and all(isinstance(f, dict) and "idx" in f for f in floor_defs):
            floor_order = [(f["idx"], f.get("title") or f"Floor {f['idx']}")
                           for f in floor_defs if int(f["idx"]) in floor_groups]
        else:
            DEFAULT_NAMES = {0: "Ground Floor", 1: "First Floor", 2: "Second Floor", 3: "Loft"}
            floor_order = [(idx, DEFAULT_NAMES.get(idx, f"Floor {idx}"))
                           for idx in sorted(floor_groups.keys())]
        if not floor_order:
            floor_order = [(0, detected.get("floor_title") or "Floor Plan")]

    use_single_page = (page_mode == "single_page")
    print(f"[VISIO] Rendering {len(floor_order)} page(s): "
          f"{[t for _, t in floor_order]}")

    all_sample_details = detected.get("sample_details", [])

    visio = None
    try:
        pythoncom.CoInitialize()

        visio = win32com.client.DispatchEx("Visio.Application")
        try:
            visio.Visible = False
        except Exception:
            pass
        try:
            visio.AlertResponse = 7  # Auto-dismiss all dialog popups (IDNO)
        except Exception:
            pass
        try:
            visio.EventsEnabled = False
        except Exception:
            pass
        try:
            visio.Settings.ShowSmartTags = False
        except Exception:
            pass

        template_path = _resolve_template_path()
        has_template = os.path.exists(template_path)
        if has_template:
            print(f"[VISIO] Creating new document from template: {template_path}")
            doc = visio.Documents.Add(template_path)
            removed = _remove_off_page_template_shapes(doc.Pages.Item(1))
            print(f"[VISIO] Removed {removed} off-page template palette shape(s)")
            # Rename first page of template if in single page mode
            use_single_page = (page_mode == "single_page")
            if use_single_page:
                page = doc.Pages.Item(1)
                page.Name = "Floor Plans"
        else:
            print("[VISIO] Template not found, creating blank document")
            doc = visio.Documents.Add("")

        # Record original template shapes to replicate on subsequent pages
        template_shapes = []
        try:
            ref_page = doc.Pages.Item(1)
            for s in ref_page.Shapes:
                template_shapes.append(s)
        except Exception as e:
            print(f"[VISIO] Failed to collect template shapes: {e}")

        # Read page settings from page 1 to propagate to subsequent pages if in multi-page/split-loft mode
        ref_cells = {}
        try:
            ref_page = doc.Pages.Item(1)
            for c in ["PageWidth", "PageHeight", "PageScale", "DrawingScale", "DrawingSizeType", "DrawingScaleType"]:
                try:
                    ref_cells[c] = ref_page.PageSheet.Cells(c).FormulaU
                except Exception:
                    pass
        except Exception as e:
            print(f"[VISIO] Could not read reference page settings: {e}")

        for _floor_pos, (_fidx, _ftitle) in enumerate(floor_order):
            rooms_all_floors = all_rooms
            rooms = list(floor_groups.get(_fidx, []))
            if not rooms:
                continue
            
            orig_floors = sorted(list(set(r.get("_orig_floor_idx", 0) for r in rooms)))
            page_use_grid = len(orig_floors) > 1

            # Per-page Visio page selection / creation
            if _floor_pos == 0:
                page = doc.Pages.Item(1)
            else:
                page = doc.Pages.Add()
                # Apply reference page settings to ensure identical dimensions and scale
                if ref_cells:
                    for cell_name, cell_formula in ref_cells.items():
                        try:
                            page.PageSheet.Cells(cell_name).FormulaU = cell_formula
                        except Exception as ce:
                            print(f"[VISIO] Failed to copy cell {cell_name} to page {_ftitle}: {ce}")
                
                # Duplicate template shapes from Page 1 to keep consistent borders/logos/forms
                if template_shapes:
                    try:
                        for s in template_shapes:
                            page.Drop(s, s.Cells("PinX").ResultIU, s.Cells("PinY").ResultIU)
                    except Exception as se:
                        print(f"[VISIO] Failed to copy template shapes to page {_ftitle}: {se}")
            page.Name = _ftitle
            page.Background = 0
            
            try:
                _page_count = doc.Pages.Count
            except Exception:
                _page_count = "?"
            print(f"[VISIO] Page {_floor_pos + 1}/{len(floor_order)}: '{_ftitle}' "
                  f"({len(rooms)} rooms, represents floors {orig_floors}) - doc now has {_page_count} page(s)")
            
            # Filter samples for this page
            _typed = [s for s in all_sample_details if s.get("target_floor_idx") is not None]
            _untyped = [s for s in all_sample_details if s.get("target_floor_idx") is None]
            
            # A typed sample goes to this page if its target_floor_idx is in the set of original floors represented on this page
            sample_details = [s for s in _typed if int(s.get("target_floor_idx", 0)) in orig_floors]
            if _typed:
                print(f"[VISIO] Page '{_ftitle}' sample filter: -> {[s.get('id') for s in sample_details]}")
                      
            _sketch_size = detected.get("sketch_size") or [0, 0]
            _bboxes_f = [r.get("bbox") for r in rooms if r.get("bbox")]
            if _bboxes_f and _sketch_size[0] and _sketch_size[1]:
                _fx1 = min(b[0] for b in _bboxes_f)
                _fy1 = min(b[1] for b in _bboxes_f)
                _fx2 = max(b[0] + b[2] for b in _bboxes_f)
                _fy2 = max(b[1] + b[3] for b in _bboxes_f)
                sample_details += [s for s in _untyped
                    if s.get("location") and
                    _fx1 <= s["location"][0] <= _fx2 and
                    _fy1 <= s["location"][1] <= _fy2]
            elif _floor_pos == 0:
                sample_details += _untyped
            
            has_bbox = any(room.get("bbox") for room in rooms)
            # Build the floor set from the rooms' actual floor names so that a
            # single page carrying multiple floors (single_page mode) renders
            # each as its own labelled, non-overlapping section.
            floor_set = {(room.get("floor") or _ftitle) for room in rooms} or {_ftitle}

            # Query adaptive drawing scale multiplier from page width in internal units
            try:
                raw_page_w = page.PageSheet.Cells("PageWidth").ResultIU
                raw_page_h = page.PageSheet.Cells("PageHeight").ResultIU
                scale_mult = raw_page_w / 16.54 if raw_page_w > 50 else 1.0
            except Exception:
                raw_page_w = 16.54
                raw_page_h = 11.69
                scale_mult = 1.0

            PAGE_W = raw_page_w
            PAGE_H = raw_page_h
            PAGE_W, PAGE_H = _landscape_dimensions(PAGE_W, PAGE_H)
            try:
                page.PageSheet.Cells("PageWidth").ResultIU = PAGE_W
                page.PageSheet.Cells("PageHeight").ResultIU = PAGE_H
            except Exception:
                pass

            # Margins (scaled dynamically by scale_mult)
            MARGIN_L = 2.5 * scale_mult
            MARGIN_R = 2.5 * scale_mult
            MARGIN_T = 1.0 * scale_mult
            MARGIN_B = 1.2 * scale_mult
            DRAW_W = PAGE_W - MARGIN_L - MARGIN_R
            DRAW_H = PAGE_H - MARGIN_T - MARGIN_B

            # Professional styling
            WALL_WEIGHT = "0.95 pt"
            WALL_WEIGHT_OUTER = "2.5 pt"  # Bold outer boundary walls (PropertyBox style)
            WALL_RGB = "RGB(32,32,32)"
            DOOR_DEFAULT_RGB = "RGB(36,94,168)"      # blue for standard internal doors
            DOOR_RESTRICTED_RGB = "RGB(0,96,168)"    # blue for no-access adjacency
            DOOR_ACM_RGB = "RGB(176,40,40)"          # red near ACM rooms
            STAIR_RGB = "RGB(80,80,80)"
            hints = detection_hints or {}
            # Acorn spec: ACM = light pink (matches the team's manual plans),
            # no-access = blue, normal = white.
            acm_rgb = hints.get("acm_fill_rgb", (245, 208, 212))
            no_access_rgb = hints.get("no_access_fill_rgb", (50, 100, 200))
            cable_rgb = hints.get("cable_route_fill_rgb", (0, 200, 0))
            FILL_ACM = f"RGB({acm_rgb[0]},{acm_rgb[1]},{acm_rgb[2]})"
            FILL_NO_ACCESS = f"RGB({no_access_rgb[0]},{no_access_rgb[1]},{no_access_rgb[2]})"
            FILL_CLEAR = "RGB(255,255,255)"
            FILL_OUT_SCOPE = "RGB(236,236,236)"
            fill_map = {
                "acm": FILL_ACM,
                "no_access": FILL_NO_ACCESS,
                "out_of_scope": FILL_OUT_SCOPE,
                "clear": FILL_CLEAR,
            }            # ---- Coordinate transform (Quadrant Grid or Single Page Layout) ----
            floor_bounds = {}
            floor_transforms = {}
            if page_use_grid:
                for ofidx in orig_floors:
                    f_rooms = [r for r in rooms if r.get("_orig_floor_idx") == ofidx]
                    if not f_rooms:
                        continue
                    min_bx_f = min(r.get("bbox", [0,0,1,1])[0] for r in f_rooms)
                    min_by_f = min(r.get("bbox", [0,0,1,1])[1] for r in f_rooms)
                    max_bx_f = max(r.get("bbox", [0,0,1,1])[0] + r.get("bbox", [0,0,1,1])[2] for r in f_rooms)
                    max_by_f = max(r.get("bbox", [0,0,1,1])[1] + r.get("bbox", [0,0,1,1])[3] for r in f_rooms)
                    floor_bounds[ofidx] = (min_bx_f, min_by_f, max_bx_f, max_by_f)

                for ofidx in orig_floors:
                    if ofidx not in floor_bounds:
                        continue
                    min_bx_f, min_by_f, max_bx_f, max_by_f = floor_bounds[ofidx]
                    
                    if len(orig_floors) == 1:
                        cell_x1 = MARGIN_L
                        cell_y1 = MARGIN_B
                        cell_x2 = PAGE_W - MARGIN_R
                        cell_y2 = PAGE_H - MARGIN_T
                    elif len(orig_floors) == 2:
                        col_w = DRAW_W / 2
                        col_idx = 0 if ofidx == orig_floors[0] else 1
                        cell_x1 = MARGIN_L + col_idx * col_w
                        cell_y1 = MARGIN_B
                        cell_x2 = cell_x1 + col_w
                        cell_y2 = PAGE_H - MARGIN_T
                    else:
                        col_w = DRAW_W / 2
                        row_h = DRAW_H / 2
                        if ofidx == 1:    # Top-Left
                            cell_x1, cell_y1 = MARGIN_L, MARGIN_B + row_h
                        elif ofidx == 2:  # Bottom-Left
                            cell_x1, cell_y1 = MARGIN_L, MARGIN_B
                        elif ofidx == 3:  # Top-Right
                            cell_x1, cell_y1 = MARGIN_L + col_w, MARGIN_B + row_h
                        elif ofidx == 0:  # Bottom-Right
                            cell_x1, cell_y1 = MARGIN_L + col_w, MARGIN_B
                        else:
                            seq_idx = orig_floors.index(ofidx)
                            if seq_idx == 0:
                                cell_x1, cell_y1 = MARGIN_L, MARGIN_B + row_h
                            elif seq_idx == 1:
                                cell_x1, cell_y1 = MARGIN_L, MARGIN_B
                            elif seq_idx == 2:
                                cell_x1, cell_y1 = MARGIN_L + col_w, MARGIN_B + row_h
                            else:
                                cell_x1, cell_y1 = MARGIN_L + col_w, MARGIN_B
                        cell_x2 = cell_x1 + col_w
                        cell_y2 = cell_y1 + row_h
                    
                    cell_pad_x = 0.4 * scale_mult
                    cell_pad_y = 0.4 * scale_mult
                    cell_draw_w = max(10.0, cell_x2 - cell_x1 - 2 * cell_pad_x)
                    cell_draw_h = max(10.0, cell_y2 - cell_y1 - 2 * cell_pad_y)
                    
                    src_w = max(1, max_bx_f - min_bx_f)
                    src_h = max(1, max_by_f - min_by_f)
                    v_scale_f = min(cell_draw_w / src_w, cell_draw_h / src_h)
                    draw_actual_w_f = src_w * v_scale_f
                    draw_actual_h_f = src_h * v_scale_f
                    v_offset_x_f = cell_x1 + cell_pad_x + (cell_draw_w - draw_actual_w_f) / 2
                    v_offset_y_f = cell_y1 + cell_pad_y + (cell_draw_h - draw_actual_h_f) / 2
                    
                    floor_transforms[ofidx] = (min_bx_f, min_by_f, v_scale_f, draw_actual_w_f, draw_actual_h_f, v_offset_x_f, v_offset_y_f)

            if has_bbox:
                _sketch_size = detected.get("sketch_size") or [0, 0]
                if len(rooms) == 1 and _sketch_size[0] > 0 and _sketch_size[1] > 0:
                    min_bx = 0
                    min_by = 0
                    max_bx = _sketch_size[0]
                    max_by = _sketch_size[1]
                else:
                    min_bx = min(r.get("bbox", [0,0,1,1])[0] for r in rooms)
                    min_by = min(r.get("bbox", [0,0,1,1])[1] for r in rooms)
                    max_bx = max(r.get("bbox", [0,0,1,1])[0] + r.get("bbox", [0,0,1,1])[2] for r in rooms)
                    max_by = max(r.get("bbox", [0,0,1,1])[1] + r.get("bbox", [0,0,1,1])[3] for r in rooms)
                src_w = max(1, max_bx - min_bx)
                src_h = max(1, max_by - min_by)
                v_scale = min(DRAW_W / src_w, DRAW_H / src_h)
                draw_actual_w = src_w * v_scale
                draw_actual_h = src_h * v_scale
                v_offset_x = MARGIN_L + (DRAW_W - draw_actual_w) / 2
                v_offset_y = MARGIN_B + (DRAW_H - draw_actual_h) / 2
            else:
                min_bx, min_by = 0, 0
                v_scale = 0.005
                draw_actual_w = DRAW_W
                draw_actual_h = DRAW_H
                v_offset_x = MARGIN_L
                v_offset_y = MARGIN_B

            def get_floor_from_coords(px, py):
                if not page_use_grid:
                    return 0
                sketch_size = detected.get("sketch_size") or [0, 0]
                sketch_w, sketch_h = sketch_size[0], sketch_size[1]
                if sketch_w <= 0 or sketch_h <= 0:
                    return 0
                if px < sketch_w * 0.5:
                    if py < sketch_h * 0.5:
                        return 1  # First Floor
                    else:
                        return 2  # Second Floor
                else:
                    if py < sketch_h * 0.5:
                        return 3  # Loft
                    else:
                        return 0  # Ground Floor

            def to_visio(px, py, ofidx=None):
                if page_use_grid:
                    if ofidx is None:
                        ofidx = get_floor_from_coords(px, py)
                    if ofidx in floor_transforms:
                        min_bx_f, min_by_f, v_scale_f, _, draw_actual_h_f, v_offset_x_f, v_offset_y_f = floor_transforms[ofidx]
                        vx = v_offset_x_f + (px - min_bx_f) * v_scale_f
                        vy = v_offset_y_f + draw_actual_h_f - (py - min_by_f) * v_scale_f
                        return vx, vy
                vx = v_offset_x + (px - min_bx) * v_scale
                vy = v_offset_y + draw_actual_h - (py - min_by) * v_scale
                return vx, vy

            # NOTE: Floor titles are drawn AFTER gap-closing (see below)

            # ---- Compute room Visio positions ----
            room_rects = []
            pending_stair_access = []  # [(loft_x1, loft_mid_y)]
            repair_topology = _env_flag("VISIO_REPAIR_TOPOLOGY", False)
            infer_shared_wall_doors = _env_flag("VISIO_INFER_SHARED_WALL_DOORS", True)
            debug_geometry = _env_flag("VISIO_DEBUG_GEOMETRY", False)

            def draw_survey_stair_symbol(sx1, sy1, sx2, sy2, direction="up"):
                """Draw the Acorn template-style editable stair symbol."""
                room_w = max(sx2 - sx1, 0.01)
                room_h = max(sy2 - sy1, 0.01)
                is_horizontal = room_w >= room_h
                tread_count = max(7, min(13, int(max(room_w, room_h) / (0.14 * scale_mult))))
                symbol_box = page.DrawRectangle(sx1, sy1, sx2, sy2)
                symbol_box.Cells("FillPattern").FormulaU = "0"
                symbol_box.Cells("LineWeight").FormulaU = "0.36 pt"
                symbol_box.Cells("LineColor").FormulaU = "RGB(50,50,50)"
                tread_pad = 0.02 * scale_mult

                def _style_line(line, weight="0.36 pt"):
                    line.Cells("LineWeight").FormulaU = weight
                    line.Cells("LineColor").FormulaU = "RGB(0,0,0)"
                    return line

                if is_horizontal:
                    for li in range(tread_count):
                        lx = sx1 + (li + 1) * room_w / (tread_count + 1)
                        _style_line(page.DrawLine(lx, sy1 + tread_pad, lx, sy2 - tread_pad), "0.24 pt")
                    mid_y = (sy1 + sy2) / 2.0
                    tip_x = sx1 + min(room_w * 0.24, 0.30 * scale_mult)
                    tail_x = sx1 + min(room_w * 0.07, 0.10 * scale_mult)
                    wing = min(room_h * 0.34, 0.24 * scale_mult)
                    if str(direction).lower() == "down":
                        tip_x = sx2 - min(room_w * 0.24, 0.30 * scale_mult)
                        tail_x = sx2 - min(room_w * 0.07, 0.10 * scale_mult)
                        stem = page.DrawLine(tail_x, mid_y, sx2 + 0.18 * scale_mult, mid_y)
                        lbl_x1, lbl_x2 = sx2 + 0.20 * scale_mult, sx2 + 0.62 * scale_mult
                        label = "Down"
                    else:
                        stem = page.DrawLine(sx1 - 0.18 * scale_mult, mid_y, tail_x, mid_y)
                        lbl_x1, lbl_x2 = sx1 - 0.55 * scale_mult, sx1 - 0.20 * scale_mult
                        label = "Up"
                    _style_line(page.DrawLine(tail_x, mid_y - wing, tip_x, mid_y), "0.50 pt")
                    _style_line(page.DrawLine(tail_x, mid_y + wing, tip_x, mid_y), "0.50 pt")
                    _style_line(stem, "0.50 pt")
                    lbl = page.DrawRectangle(lbl_x1, mid_y - 0.08 * scale_mult, lbl_x2, mid_y + 0.08 * scale_mult)
                else:
                    for li in range(tread_count):
                        ly = sy1 + (li + 1) * room_h / (tread_count + 1)
                        _style_line(page.DrawLine(sx1 + tread_pad, ly, sx2 - tread_pad, ly), "0.24 pt")
                    mid_x = (sx1 + sx2) / 2.0
                    tip_y = sy1 + min(room_h * 0.24, 0.30 * scale_mult)
                    tail_y = sy1 + min(room_h * 0.07, 0.10 * scale_mult)
                    wing = min(room_w * 0.34, 0.24 * scale_mult)
                    if str(direction).lower() == "down":
                        tip_y = sy2 - min(room_h * 0.24, 0.30 * scale_mult)
                        tail_y = sy2 - min(room_h * 0.07, 0.10 * scale_mult)
                        stem = page.DrawLine(mid_x, tail_y, mid_x, sy2 + 0.18 * scale_mult)
                        lbl = page.DrawRectangle(mid_x - 0.22 * scale_mult, sy2 + 0.20 * scale_mult, mid_x + 0.22 * scale_mult, sy2 + 0.36 * scale_mult)
                        label = "Down"
                    else:
                        stem = page.DrawLine(mid_x, sy1 - 0.18 * scale_mult, mid_x, tail_y)
                        lbl = page.DrawRectangle(mid_x - 0.14 * scale_mult, sy1 - 0.38 * scale_mult, mid_x + 0.14 * scale_mult, sy1 - 0.20 * scale_mult)
                        label = "Up"
                    _style_line(page.DrawLine(mid_x - wing, tail_y, mid_x, tip_y), "0.50 pt")
                    _style_line(page.DrawLine(mid_x + wing, tail_y, mid_x, tip_y), "0.50 pt")
                    _style_line(stem, "0.50 pt")
                lbl.Text = label
                lbl.Cells("Char.Size").FormulaU = "5 pt"
                lbl.Cells("Char.Color").FormulaU = "RGB(0,0,0)"
                lbl.Cells("LinePattern").FormulaU = "0"
                lbl.Cells("FillPattern").FormulaU = "0"
                lbl.Cells("Para.HorzAlign").FormulaU = "1"

            for i, room in enumerate(rooms):
                bbox = room.get("bbox")
                if bbox and has_bbox:
                    ofidx = room.get("_orig_floor_idx", 0) if (page_use_grid and len(rooms) > 1) else int(room.get("floor_idx", 0) or 0)
                    vx1, vy1 = to_visio(bbox[0], bbox[1], ofidx)
                    vx2, vy2 = to_visio(bbox[0] + bbox[2], bbox[1] + bbox[3], ofidx)
                    x1, y1 = min(vx1, vx2), min(vy1, vy2)
                    x2, y2 = max(vx1, vx2), max(vy1, vy2)
                else:
                    cols = max(1, min(4, math.ceil(math.sqrt(len(rooms)))))
                    gx = i % cols
                    gy = i // cols
                    cell_w = DRAW_W / cols
                    cell_h = DRAW_H / max(1, math.ceil(len(rooms) / cols))
                    x1 = MARGIN_L + gx * cell_w + 0.05
                    y1 = PAGE_H - MARGIN_T - (gy + 1) * cell_h + 0.05
                    x2 = x1 + cell_w - 0.1
                    y2 = y1 + cell_h - 0.1
                room_rects.append((x1, y1, x2, y2))
            if debug_geometry:
                print("[DEBUG] initial room_rects:", room_rects)

            # ---- PASS 0: BUILDING OUTLINE ALIGNMENT ----
            # All rooms on the same floor should share outer edges (building boundary).
            # Find the overall bounding box and extend edge rooms to match.
            rects_pre = [list(r) for r in room_rects]
            if repair_topology and len(rects_pre) > 1:
                # Group by original floor to align separately
                f_indices = {}
                for idx, room in enumerate(rooms):
                    ofidx = room.get("_orig_floor_idx", 0) if page_use_grid else int(room.get("floor_idx", 0) or 0)
                    f_indices.setdefault(ofidx, []).append(idx)
                
                for ofidx, indices in f_indices.items():
                    if len(indices) <= 1:
                        continue
                    all_left = min(rects_pre[idx][0] for idx in indices)
                    all_bottom = min(rects_pre[idx][1] for idx in indices)
                    all_right = max(rects_pre[idx][2] for idx in indices)
                    all_top = max(rects_pre[idx][3] for idx in indices)
                    EDGE_TOL = 0.25 * scale_mult
                    for idx in indices:
                        r = rects_pre[idx]
                        if abs(r[0] - all_left) < EDGE_TOL:
                            r[0] = all_left
                        if abs(r[1] - all_bottom) < EDGE_TOL:
                            r[1] = all_bottom
                        if abs(r[2] - all_right) < EDGE_TOL:
                            r[2] = all_right
                        if abs(r[3] - all_top) < EDGE_TOL:
                            r[3] = all_top
                room_rects = [tuple(r) for r in rects_pre]
            if debug_geometry:
                print("[DEBUG] after PASS 0:", room_rects)

            # ---- GAP-CLOSING: snap nearby room edges together ----
            # If two rooms' edges are within GAP_THRESHOLD inches, extend them to meet
            # IMPORTANT: Only close gaps between rooms on the SAME floor
            GAP_THRESHOLD_IN = _conservative_gap_threshold(room_rects, scale_mult)
            OVERLAP_MIN_IN = 0.10 * scale_mult   # minimum overlap to consider rooms adjacent (lowered for better detection)
            rects_mut = [list(r) for r in room_rects]  # [x1, y1, x2, y2] mutable
            for i in (range(len(rects_mut)) if repair_topology else range(0)):
                ix1, iy1, ix2, iy2 = rects_mut[i]
                i_floor = (rooms[i].get("floor") or "")
                for j in range(i + 1, len(rects_mut)):
                    j_floor = (rooms[j].get("floor") or "")
                    # Skip gap-closing between rooms on different floors
                    if i_floor != j_floor and i_floor and j_floor:
                        continue
                    jx1, jy1, jx2, jy2 = rects_mut[j]

                    # Vertical overlap (rooms side-by-side horizontally)
                    v_overlap = min(iy2, jy2) - max(iy1, jy1)
                    # Horizontal overlap (rooms stacked vertically)
                    h_overlap = min(ix2, jx2) - max(ix1, jx1)

                    if v_overlap > OVERLAP_MIN_IN:
                        # Right edge of i near left edge of j (or overlapping slightly)
                        gap = jx1 - ix2
                        if -GAP_THRESHOLD_IN <= gap <= GAP_THRESHOLD_IN:
                            mid = (ix2 + jx1) / 2
                            rects_mut[i][2] = mid  # extend/retract i's right
                            rects_mut[j][0] = mid  # pull/retract j's left
                        # Left edge of i near right edge of j (or overlapping slightly)
                        gap = ix1 - jx2
                        if -GAP_THRESHOLD_IN <= gap <= GAP_THRESHOLD_IN:
                            mid = (jx2 + ix1) / 2
                            rects_mut[j][2] = mid  # extend/retract j's right
                            rects_mut[i][0] = mid  # pull/retract i's left

                    if h_overlap > OVERLAP_MIN_IN:
                        # Top edge of i near bottom edge of j (Visio Y-up: y2 > y1)
                        gap = jy1 - iy2
                        if -GAP_THRESHOLD_IN <= gap <= GAP_THRESHOLD_IN:
                            mid = (iy2 + jy1) / 2
                            rects_mut[i][3] = mid  # extend/retract i's top
                            rects_mut[j][1] = mid  # pull/retract j's bottom
                        # Bottom edge of i near top edge of j
                        gap = iy1 - jy2
                        if -GAP_THRESHOLD_IN <= gap <= GAP_THRESHOLD_IN:
                            mid = (jy2 + iy1) / 2
                            rects_mut[j][3] = mid  # extend/retract j's top
                            rects_mut[i][1] = mid  # pull/retract i's bottom

                    # Also align edges that are very close (nearly shared wall)
                    if v_overlap > OVERLAP_MIN_IN:
                        # Align top/bottom edges if nearly matching
                        if abs(iy1 - jy1) < OVERLAP_MIN_IN:
                            shared_y = min(iy1, jy1)
                            rects_mut[i][1] = shared_y
                            rects_mut[j][1] = shared_y
                        if abs(iy2 - jy2) < OVERLAP_MIN_IN:
                            shared_y = max(iy2, jy2)
                            rects_mut[i][3] = shared_y
                            rects_mut[j][3] = shared_y

                    if h_overlap > OVERLAP_MIN_IN:
                        # Align left/right edges if nearly matching
                        if abs(ix1 - jx1) < OVERLAP_MIN_IN:
                            shared_x = min(ix1, jx1)
                            rects_mut[i][0] = shared_x
                            rects_mut[j][0] = shared_x
                        if abs(ix2 - jx2) < OVERLAP_MIN_IN:
                            shared_x = max(ix2, jx2)
                            rects_mut[i][2] = shared_x
                            rects_mut[j][2] = shared_x
            if debug_geometry:
                print("[DEBUG] after PASS 1:", rects_mut)

            # ---- PASS 2: Fill open gaps where a room can extend to meet a neighbor ----
            # For rooms that share a horizontal or vertical edge but have a large gap,
            # extend the smaller room IF no other room occupies that gap space.
            # Only extends between rooms on the SAME floor.
            # PASS 1 already closes small detector-noise gaps. A larger
            # extension invents topology and stretches rooms across blank
            # areas, so use the same conservative threshold here.
            MAX_EXTEND_IN = GAP_THRESHOLD_IN
            changed = True
            passes = 0
            while repair_topology and changed and passes < 1:
                changed = False
                passes += 1
                for i in range(len(rects_mut)):
                    ix1, iy1, ix2, iy2 = rects_mut[i]
                    i_floor = (rooms[i].get("floor") or "")
                    for j in range(len(rects_mut)):
                        if i == j:
                            continue
                        j_floor = (rooms[j].get("floor") or "")
                        if i_floor != j_floor and i_floor and j_floor:
                            continue
                        jx1, jy1, jx2, jy2 = rects_mut[j]

                        # Case A: rooms share a y-range and room i's right edge < room j's left edge
                        # (room i could extend rightward to meet room j)
                        y_overlap = min(iy2, jy2) - max(iy1, jy1)
                        if y_overlap > OVERLAP_MIN_IN and ix2 < jx1:
                            gap = jx1 - ix2
                            if 0 < gap <= MAX_EXTEND_IN:
                                # Check no other room occupies this gap
                                gap_blocked = False
                                for k in range(len(rects_mut)):
                                    if k == i or k == j:
                                        continue
                                    kx1, ky1, kx2, ky2 = rects_mut[k]
                                    k_y_overlap = min(iy2, ky2) - max(iy1, ky1)
                                    if k_y_overlap > OVERLAP_MIN_IN and kx1 < jx1 and kx2 > ix2:
                                        gap_blocked = True
                                        break
                                if not gap_blocked:
                                    rects_mut[i][2] = jx1  # extend i's right to j's left
                                    changed = True

                        # Case B: rooms share a y-range and room i's left edge > room j's right edge
                        # (room i could extend leftward to meet room j)
                        if y_overlap > OVERLAP_MIN_IN and ix1 > jx2:
                            gap = ix1 - jx2
                            if 0 < gap <= MAX_EXTEND_IN:
                                gap_blocked = False
                                for k in range(len(rects_mut)):
                                    if k == i or k == j:
                                        continue
                                    kx1, ky1, kx2, ky2 = rects_mut[k]
                                    k_y_overlap = min(iy2, ky2) - max(iy1, ky1)
                                    if k_y_overlap > OVERLAP_MIN_IN and kx1 < ix1 and kx2 > jx2:
                                        gap_blocked = True
                                        break
                                if not gap_blocked:
                                    rects_mut[i][0] = jx2  # extend i's left to j's right
                                    changed = True

                        # Case C: rooms share an x-range and room i's top < room j's bottom
                        x_overlap = min(ix2, jx2) - max(ix1, jx1)
                        if x_overlap > OVERLAP_MIN_IN and iy2 < jy1:
                            gap = jy1 - iy2
                            if 0 < gap <= MAX_EXTEND_IN:
                                gap_blocked = False
                                for k in range(len(rects_mut)):
                                    if k == i or k == j:
                                        continue
                                    kx1, ky1, kx2, ky2 = rects_mut[k]
                                    k_x_overlap = min(ix2, kx2) - max(ix1, kx1)
                                    if k_x_overlap > OVERLAP_MIN_IN and ky1 < jy1 and ky2 > iy2:
                                        gap_blocked = True
                                        break
                                if not gap_blocked:
                                    rects_mut[i][3] = jy1  # extend i's top to j's bottom
                                    changed = True

                        # Case D: rooms share an x-range and room i's bottom > room j's top
                        if x_overlap > OVERLAP_MIN_IN and iy1 > jy2:
                            gap = iy1 - jy2
                            if 0 < gap <= MAX_EXTEND_IN:
                                gap_blocked = False
                                for k in range(len(rects_mut)):
                                    if k == i or k == j:
                                        continue
                                    kx1, ky1, kx2, ky2 = rects_mut[k]
                                    k_x_overlap = min(ix2, kx2) - max(ix1, kx1)
                                    if k_x_overlap > OVERLAP_MIN_IN and ky1 < iy1 and ky2 > jy2:
                                        gap_blocked = True
                                        break
                                if not gap_blocked:
                                    rects_mut[i][1] = jy2  # extend i's bottom to j's top
                                    changed = True

            room_rects = [tuple(r) for r in rects_mut]
            if debug_geometry:
                print("[DEBUG] after PASS 2:", room_rects)

            # ---- FLOOR SEPARATION: ensure visual gap between different floor sections ----
            # In Visio Y-up coordinates: higher floors should be at higher Y values
            # Add a gap between the lowest room of one floor and highest room of the floor below
            # Detached rooms (e.g., Loft repositioned to the right) are placed beside the
            # nearest main floor section instead of being stacked vertically.
            detached_floors = {}  # init for single-floor case
            if len(floor_set) > 1:
                FLOOR_GAP_IN = 0.6  # visual gap between floor sections in inches
                # Build floor bounding boxes and room indices per floor
                floor_bounds = {}   # floor_name -> [min_y, max_y, min_x, max_x]
                floor_indices = {}  # floor_name -> [room indices]
                for i, room in enumerate(rooms):
                    fl = room.get("floor") or ""
                    if not fl:
                        continue
                    x1, y1, x2, y2 = room_rects[i]
                    if fl not in floor_bounds:
                        floor_bounds[fl] = [y1, y2, x1, x2]
                        floor_indices[fl] = [i]
                    else:
                        floor_bounds[fl][0] = min(floor_bounds[fl][0], y1)
                        floor_bounds[fl][1] = max(floor_bounds[fl][1], y2)
                        floor_bounds[fl][2] = min(floor_bounds[fl][2], x1)
                        floor_bounds[fl][3] = max(floor_bounds[fl][3], x2)
                        floor_indices[fl].append(i)

                # Identify detached floors: single room positioned far to the right
                # (e.g., Loft repositioned by overlap filter in detector.py)
                main_floors = {}
                detached_floors = {}
                for fl_name, bounds in floor_bounds.items():
                    indices = floor_indices[fl_name]
                    fl_min_x = bounds[2]
                    # A floor is "detached" if it has 1 room and its left edge is
                    # far right of the other floors' right edges
                    other_max_x = max(
                        (b[3] for fn, b in floor_bounds.items() if fn != fl_name),
                        default=0)
                    if len(indices) == 1 and fl_min_x > other_max_x + 0.2:
                        detached_floors[fl_name] = bounds
                    else:
                        main_floors[fl_name] = bounds

                # Sort main floors by their current Visio Y position (ascending)
                sorted_floors = sorted(main_floors.items(), key=lambda x: x[1][0])
                rects_list = [list(r) for r in room_rects]

                # Add vertical gaps between main floor sections
                for fi in range(len(sorted_floors) - 1):
                    lower_name, lower_bounds = sorted_floors[fi]
                    upper_name, upper_bounds = sorted_floors[fi + 1]
                    current_gap = upper_bounds[0] - lower_bounds[1]
                    if current_gap < FLOOR_GAP_IN:
                        shift = FLOOR_GAP_IN - current_gap
                        for i in floor_indices.get(upper_name, []):
                            rects_list[i][1] += shift
                            rects_list[i][3] += shift
                        # Update bounds for cascading shifts
                        for fj in range(fi + 1, len(sorted_floors)):
                            sorted_floors[fj] = (
                                sorted_floors[fj][0],
                                [sorted_floors[fj][1][0] + shift,
                                 sorted_floors[fj][1][1] + shift,
                                 sorted_floors[fj][1][2],
                                 sorted_floors[fj][1][3]]
                            )
                        print(f"[VISIO] Floor gap: shifted '{upper_name}' up by {shift:.2f}in "
                              f"(gap was {current_gap:.2f}in)")

                # Position detached floors beside the highest main floor section
                if detached_floors and sorted_floors:
                    # Place detached rooms to the right of the topmost main floor
                    top_floor_name, top_floor_bounds = sorted_floors[-1]
                    top_max_x = top_floor_bounds[3]
                    top_max_y = max(rects_list[i][3] for i in floor_indices[top_floor_name])
                    detach_x_start = top_max_x + 0.8  # gap to the right

                    for fl_name, bounds in detached_floors.items():
                        for i in floor_indices[fl_name]:
                            room_w = rects_list[i][2] - rects_list[i][0]
                            room_h = rects_list[i][3] - rects_list[i][1]
                            # Align top of detached room with top of the main floor
                            rects_list[i][0] = detach_x_start
                            rects_list[i][2] = detach_x_start + room_w
                            rects_list[i][3] = top_max_y
                            rects_list[i][1] = top_max_y - room_h
                            print(f"[VISIO] Detached '{fl_name}' placed at x={detach_x_start:.2f}, "
                                  f"aligned with '{top_floor_name}'")

                room_rects = [tuple(r) for r in rects_list]

            # ---- Sort rooms spatially for auto-numbering fallback ----
            # Only auto-number rooms that don't already have a number from GPT-4o/OCR.
            # Skip numbers already taken by other rooms to avoid duplicates.
            taken_numbers = set()
            for room in rooms:
                rn = room.get("room_number")
                if rn:
                    taken_numbers.add(rn.lstrip("0") or "0")

            room_order = sorted(range(len(rooms)), key=lambda i: (
                -room_rects[i][3],
                room_rects[i][0],
            ))
            room_numbers = {}
            next_num = 1
            for rank, orig_idx in enumerate(room_order):
                if rooms[orig_idx].get("room_number"):
                    room_numbers[orig_idx] = rooms[orig_idx]["room_number"]
                else:
                    # Find next available number
                    while str(next_num) in taken_numbers:
                        next_num += 1
                    room_numbers[orig_idx] = f"{next_num:03d}"
                    taken_numbers.add(str(next_num))
                    next_num += 1

            # ---- FLOOR TITLE(S) - positioned using final Visio room_rects ----
            if page_use_grid:
                # Group rooms by original floor and draw their individual titles
                orig_floors = sorted(list(set(r.get("_orig_floor_idx", 0) for r in rooms)))
                DEFAULT_NAMES = {0: "Ground Floor", 1: "First Floor", 2: "Second Floor", 3: "Loft"}
                for ofidx in orig_floors:
                    fl_indices = [i for i, r in enumerate(rooms) if r.get("_orig_floor_idx", 0) == ofidx]
                    if not fl_indices:
                        continue
                    
                    # Try to get the floor name from the room objects, else fallback to DEFAULT_NAMES
                    room_floor_names = [rooms[i].get("_orig_floor_name") for i in fl_indices if rooms[i].get("_orig_floor_name")]
                    if room_floor_names:
                        fl_name = room_floor_names[0]
                    else:
                        fl_name = DEFAULT_NAMES.get(ofidx, f"Floor {ofidx}")
                    fl_rects = [room_rects[i] for i in fl_indices]
                    fl_top_y = max(r[3] for r in fl_rects)
                    fl_left_x = min(r[0] for r in fl_rects)
                    
                    title_x = fl_left_x
                    title_y = fl_top_y + 0.15 * scale_mult
                    
                    title_text = f"{fl_name}:"
                    
                    # Try to reuse the existing template title shape if available
                    title_shape = None
                    for s_idx in range(1, page.Shapes.Count + 1):
                        s = page.Shapes.Item(s_idx)
                        clean_text = s.Text.replace(":", "").strip().lower()
                        clean_target = fl_name.lower()
                        if clean_text == clean_target:
                            title_shape = s
                            break
                            
                    tx_start = title_x
                    tx_end = title_x + 3.0 * scale_mult
                    ty_start = title_y
                    ty_end = title_y + 0.35 * scale_mult
                    
                    if title_shape:
                        # Reposition and size existing stencil
                        title_shape.Cells("Width").ResultIU = tx_end - tx_start
                        title_shape.Cells("Height").ResultIU = ty_end - ty_start
                        title_shape.Cells("PinX").ResultIU = tx_start + (tx_end - tx_start) / 2
                        title_shape.Cells("PinY").ResultIU = ty_start + (ty_end - ty_start) / 2
                    else:
                        # Create new title shape
                        title_shape = page.DrawRectangle(tx_start, ty_start, tx_end, ty_end)
                        title_shape.Text = title_text
                        title_shape.Cells("Char.Size").FormulaU = "12 pt"
                        title_shape.Cells("Char.Style").FormulaU = "5"  # Bold + Underline
                        title_shape.Cells("Char.Color").FormulaU = "RGB(0,0,0)"
                        title_shape.Cells("LinePattern").FormulaU = "0"
                        title_shape.Cells("FillPattern").FormulaU = "0"
                        title_shape.Cells("Para.HorzAlign").FormulaU = "0"
            elif len(floor_set) > 1:
                # Multiple floors: place title above each floor's rooms (using Visio coords)
                for fl_name in floor_set:
                    fl_indices = [i for i, r in enumerate(rooms)
                                  if (r.get("floor") or "") == fl_name]
                    if not fl_indices:
                        continue
                    # Use the actual Visio-space rects (after gap-closing)
                    fl_rects = [room_rects[i] for i in fl_indices]
                    fl_top_y = max(r[3] for r in fl_rects)    # highest Y (Visio Y-up)
                    fl_left_x = min(r[0] for r in fl_rects)   # leftmost X
                    title_x = fl_left_x
                    title_y = fl_top_y + 0.15  # above the top-most room

                    title_shape = page.DrawRectangle(
                        title_x, title_y, title_x + 3, title_y + 0.3)
                    title_shape.Text = f"{fl_name}:"
                    title_shape.Cells("Char.Size").FormulaU = "11 pt"
                    title_shape.Cells("Char.Style").FormulaU = "5"  # Bold + Underline
                    title_shape.Cells("Char.Color").FormulaU = "RGB(0,0,0)"
                    title_shape.Cells("LinePattern").FormulaU = "0"
                    title_shape.Cells("FillPattern").FormulaU = "0"
                    title_shape.Cells("Para.HorzAlign").FormulaU = "0"
            else:
                # Single floor on this page: one title above the plan.
                # Use the per-page floor title (_ftitle from the outer loop),
                # NOT the global detected.floor_title - otherwise non-Ground
                # pages (e.g. First Floor, Loft) all show the same incorrect
                # header that was set from GPT-4o's single floor_name.
                #
                # Prefer:
                #   1. The floor name attached to rooms on this page
                #   2. _ftitle (the floor_pages title from the outer loop)
                #   3. detected.floor_title (legacy single-floor fallback)
                room_floor_name = next(iter(floor_set), None) if floor_set else None
                floor_title = (room_floor_name
                               or _ftitle
                               or detected.get("floor_title")
                               or "Ground Floor")
                all_top = max(r[3] for r in room_rects)
                title_x = min(r[0] for r in room_rects)
                title_y = all_top + 0.15 * scale_mult
                
                # Try to reuse the existing template title shape if available
                title_shape = None
                for s_idx in range(1, page.Shapes.Count + 1):
                    s = page.Shapes.Item(s_idx)
                    clean_text = s.Text.replace(":", "").strip().lower()
                    if any(x in clean_text for x in ["ground floor", "first floor", "second floor", "loft", "floor plans"]):
                        title_shape = s
                        break
                        
                tx_start = title_x
                tx_end = title_x + 3.0 * scale_mult
                ty_start = title_y
                ty_end = title_y + 0.35 * scale_mult
                
                if title_shape:
                    title_shape.Text = f"{floor_title}:"
                    title_shape.Cells("Width").ResultIU = tx_end - tx_start
                    title_shape.Cells("Height").ResultIU = ty_end - ty_start
                    title_shape.Cells("PinX").ResultIU = tx_start + (tx_end - tx_start) / 2
                    title_shape.Cells("PinY").ResultIU = ty_start + (ty_end - ty_start) / 2
                else:
                    title_shape = page.DrawRectangle(tx_start, ty_start, tx_end, ty_end)
                    title_shape.Text = f"{floor_title}:"
                    title_shape.Cells("Char.Size").FormulaU = "12 pt"
                    title_shape.Cells("Char.Style").FormulaU = "5"  # Bold + Underline
                    title_shape.Cells("Char.Color").FormulaU = "RGB(0,0,0)"
                    title_shape.Cells("LinePattern").FormulaU = "0"
                    title_shape.Cells("FillPattern").FormulaU = "0"
                    title_shape.Cells("Para.HorzAlign").FormulaU = "0"

            # ---- DETECT SHARED WALLS for door placement ----
            # Find pairs of rooms that share a wall edge (within tolerance)
            WALL_TOL = 0.05 * scale_mult  # tolerance for shared wall detection
            DOOR_W = 0.3543 * scale_mult  # 900mm door width in scaled inches
            ARC_R = 0.3543 * scale_mult   # 900mm radius in scaled inches
            shared_walls = []  # [(room_i, room_j, wall_type, wall_coord, overlap_start, overlap_end)]
            for i in (range(len(room_rects)) if infer_shared_wall_doors else range(0)):
                if rooms[i].get("annotation_only"):
                    continue
                ix1, iy1, ix2, iy2 = room_rects[i]
                i_floor = (rooms[i].get("floor") or "")
                for j in range(i + 1, len(room_rects)):
                    if rooms[j].get("annotation_only"):
                        continue
                    # Only detect shared walls between rooms on the same floor
                    j_floor = (rooms[j].get("floor") or "")
                    if i_floor != j_floor and i_floor and j_floor:
                        continue
                    jx1, jy1, jx2, jy2 = room_rects[j]
                    # Vertical shared wall (right of i = left of j)
                    if abs(ix2 - jx1) < WALL_TOL:
                        ov_start = max(iy1, jy1)
                        ov_end = min(iy2, jy2)
                        if ov_end - ov_start > DOOR_W * 1.5:
                            shared_walls.append((i, j, "vertical", ix2, ov_start, ov_end))
                    # Vertical shared wall (left of i = right of j)
                    if abs(ix1 - jx2) < WALL_TOL:
                        ov_start = max(iy1, jy1)
                        ov_end = min(iy2, jy2)
                        if ov_end - ov_start > DOOR_W * 1.5:
                            shared_walls.append((j, i, "vertical", jx2, ov_start, ov_end))
                    # Horizontal shared wall (top of i = bottom of j) Ã¢â‚¬â€ Visio Y-up
                    if abs(iy2 - jy1) < WALL_TOL:
                        ov_start = max(ix1, jx1)
                        ov_end = min(ix2, jx2)
                        if ov_end - ov_start > DOOR_W * 1.5:
                            shared_walls.append((i, j, "horizontal", iy2, ov_start, ov_end))
                    # Horizontal shared wall (bottom of i = top of j)
                    if abs(iy1 - jy2) < WALL_TOL:
                        ov_start = max(ix1, jx1)
                        ov_end = min(ix2, jx2)
                        if ov_end - ov_start > DOOR_W * 1.5:
                            shared_walls.append((j, i, "horizontal", jy2, ov_start, ov_end))

            # Build door positions: one door per shared wall, centered on overlap
            # door_positions[room_idx] = list of (wall_side, door_start, door_end) for masking
            door_positions = {i: [] for i in range(len(rooms))}
            door_arcs = []  # [(arc_cx, arc_cy, arc_r, arc_type, door_rgb)]
            for ri, rj, wtype, wcoord, ov_start, ov_end in shared_walls:
                # Place door at center of shared wall overlap
                mid = (ov_start + ov_end) / 2
                d_start = mid - DOOR_W / 2
                d_end = mid + DOOR_W / 2
                ri_type = (rooms[ri].get("type") or "clear").lower()
                rj_type = (rooms[rj].get("type") or "clear").lower()
                if "no_access" in {ri_type, rj_type}:
                    door_rgb = DOOR_RESTRICTED_RGB
                elif "acm" in {ri_type, rj_type}:
                    door_rgb = DOOR_ACM_RGB
                else:
                    door_rgb = DOOR_DEFAULT_RGB
                if wtype == "vertical":
                    # Door gap on vertical wall at x=wcoord, from y=d_start to y=d_end
                    door_positions[ri].append(("right", d_start, d_end))
                    door_positions[rj].append(("left", d_start, d_end))
                    # Arc swings into the smaller room (room j by convention)
                    door_arcs.append((wcoord, d_start, ARC_R, "vertical_right", door_rgb))
                else:
                    # Door gap on horizontal wall at y=wcoord, from x=d_start to x=d_end
                    door_positions[ri].append(("top", d_start, d_end))
                    door_positions[rj].append(("bottom", d_start, d_end))
                    door_arcs.append((d_end, wcoord, ARC_R, "horizontal_up", door_rgb))

            # ---- ROOMS - draw as 4 individual wall lines (with door gaps) + fill ----
            available_masters = [doc.Masters.Item(i).Name for i in range(1, doc.Masters.Count + 1)]
            wall_master_name = None
            for name in ["Wall.10", "Wall"]:
                if name in available_masters:
                    wall_master_name = name
                    break

            for i, room in enumerate(rooms):
                x1, y1, x2, y2 = room_rects[i]
                bbox = room.get("bbox", [])
                if debug_geometry:
                    print(f"[VISIO] Room {i}: bbox_px={bbox} -> visio=({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f})")
                room_type = room.get("type", "clear")
                if room.get("no_access"):
                    room_type = "no_access"
                label = room.get("label", "")
                room_num = room.get("room_number") or room_numbers[i]
                if room_num and room_num.isdigit():
                    room_num = room_num.zfill(3)
                fill_color = fill_map.get(room_type, FILL_CLEAR)

                if room.get("annotation_only"):
                    ann = page.DrawRectangle(x1, y1, x2, y2)
                    ann.Cells("LineColor").FormulaU = "RGB(204,0,0)"
                    ann.Cells("LineWeight").FormulaU = "1.25 pt"
                    ann.Cells("LinePattern").FormulaU = "2"
                    ann.Cells("FillForegnd").FormulaU = "RGB(255,255,255)"
                    ann.Cells("FillPattern").FormulaU = "0"
                    if room_type == "acm":
                        _draw_diagonal_hatch(page, x1, y1, x2, y2)
                    text_box = page.DrawRectangle(x1, y1, x2, y2)
                    text_box.Text = label or "Marked Area"
                    text_box.Cells("Char.Size").FormulaU = "7 pt"
                    text_box.Cells("Char.Style").FormulaU = "3"
                    text_box.Cells("Char.Color").FormulaU = "RGB(204,0,0)"
                    text_box.Cells("Para.HorzAlign").FormulaU = "1"
                    text_box.Cells("VerticalAlign").FormulaU = "1"
                    text_box.Cells("LinePattern").FormulaU = "0"
                    text_box.Cells("FillPattern").FormulaU = "0"
                    continue

                label_lower = label.lower() if label else ""
                room_w_for_stairs = max(x2 - x1, 0.01)
                room_h_for_stairs = max(y2 - y1, 0.01)
                infer_connector_stairs = os.environ.get("VISIO_INFER_CONNECTOR_STAIRS", "true").strip().lower() in {"1", "true", "yes", "on"}
                compact_connector = (
                    min(room_w_for_stairs, room_h_for_stairs) <= 0.95 * scale_mult
                    and max(room_w_for_stairs, room_h_for_stairs) <= 2.35 * scale_mult
                )
                connector_stair_label = any(
                    kw in label_lower for kw in ["landing", "lobby", "hall", "hallway", "corridor"]
                ) and not any(ex in label_lower for ex in ["cupboard", "cpd", "wc", "wet", "bath", "kitchen", "bed"])
                neighbor_labels = []
                if infer_connector_stairs and compact_connector:
                    for nj, other_room in enumerate(rooms):
                        if nj == i:
                            continue
                        ox1, oy1, ox2, oy2 = room_rects[nj]
                        h_overlap = min(x2, ox2) - max(x1, ox1)
                        v_overlap = min(y2, oy2) - max(y1, oy1)
                        touches_h = h_overlap > min(room_w_for_stairs, ox2 - ox1) * 0.35 and (
                            abs(y1 - oy2) < 0.08 * scale_mult or abs(y2 - oy1) < 0.08 * scale_mult
                        )
                        touches_v = v_overlap > min(room_h_for_stairs, oy2 - oy1) * 0.35 and (
                            abs(x1 - ox2) < 0.08 * scale_mult or abs(x2 - ox1) < 0.08 * scale_mult
                        )
                        if touches_h or touches_v:
                            neighbor_labels.append(str(other_room.get("label") or "").lower())
                if infer_connector_stairs and compact_connector and len(neighbor_labels) < 2:
                    cx_mid = (x1 + x2) / 2.0
                    cy_mid = (y1 + y2) / 2.0
                    near_limit = max(0.22 * scale_mult, min(room_w_for_stairs, room_h_for_stairs) * 0.55)
                    directional = {"left": None, "right": None, "above": None, "below": None}
                    for nj, other_room in enumerate(rooms):
                        if nj == i:
                            continue
                        ox1, oy1, ox2, oy2 = room_rects[nj]
                        olab = str(other_room.get("label") or "").lower()
                        if min(x2, ox2) - max(x1, ox1) > min(room_w_for_stairs, ox2 - ox1) * 0.20:
                            if oy1 >= y2:
                                gap = oy1 - y2
                                if gap <= near_limit and (directional["above"] is None or gap < directional["above"][0]):
                                    directional["above"] = (gap, olab)
                            if y1 >= oy2:
                                gap = y1 - oy2
                                if gap <= near_limit and (directional["below"] is None or gap < directional["below"][0]):
                                    directional["below"] = (gap, olab)
                        if min(y2, oy2) - max(y1, oy1) > min(room_h_for_stairs, oy2 - oy1) * 0.20:
                            if ox1 >= x2:
                                gap = ox1 - x2
                                if gap <= near_limit and (directional["right"] is None or gap < directional["right"][0]):
                                    directional["right"] = (gap, olab)
                            if x1 >= ox2:
                                gap = x1 - ox2
                                if gap <= near_limit and (directional["left"] is None or gap < directional["left"][0]):
                                    directional["left"] = (gap, olab)
                    neighbor_labels.extend(item[1] for item in directional.values() if item)
                has_wet_or_bath_neighbor = any(any(k in nl for k in ["wet", "bath", "wc"]) for nl in neighbor_labels)
                has_connector_neighbor = any(any(k in nl for k in ["lobby", "landing", "hall", "corridor"]) for nl in neighbor_labels)
                adjacency_stair_connector = compact_connector and has_wet_or_bath_neighbor and has_connector_neighbor
                explicit_stair_label = (
                    any(kw in label_lower for kw in ["stair", "stairs", "staircase", "stairwell", "steps"])
                    and not any(ex in label_lower for ex in ["understairs", "under stairs", "cpd", "cupboard", "wc", "store"])
                )
                # Landing/lobby/hall are connector rooms, not automatically stairs.
                # Draw room-level stairs only when the extractor/model or the room
                # label explicitly says stairs. The separate stair-gap pass below
                # still draws a stair symbol in blank gaps between wet/connector rooms.
                is_stairs = bool(room.get("has_stairs", False)) or explicit_stair_label
                is_loft = any(kw in label_lower for kw in ["loft", "attic", "roof space"])

                # Draw the room background. Real Acorn plans never solid-fill
                # a room: clear/no_access rooms stay plain white, and a room
                # with a lab-confirmed positive ACM sample gets a light
                # diagonal hatch (individual line strokes) instead of a flat
                # colour wash -- see _draw_diagonal_hatch.
                fill_rect = page.DrawRectangle(x1, y1, x2, y2)
                fill_rect.Cells("LinePattern").FormulaU = "0"  # no border line
                fill_rect.Cells("FillBkgnd").FormulaU = "RGB(255,255,255)"
                if room_type == "acm":
                    fill_rect.Cells("FillForegnd").FormulaU = "RGB(255,255,255)"
                    fill_rect.Cells("FillPattern").FormulaU = "0"
                    _draw_diagonal_hatch(page, x1, y1, x2, y2)
                else:
                    fill_rect.Cells("FillForegnd").FormulaU = fill_color
                    fill_rect.Cells("FillPattern").FormulaU = "1"

                # Collect door gaps for each side of this room
                continuous_walls = os.environ.get("CONTINUOUS_WALLS", "true").lower() == "true"
                if continuous_walls:
                    left_doors = []
                    right_doors = []
                    top_doors = []
                    bottom_doors = []
                else:
                    doors = door_positions.get(i, [])
                    left_doors = [(ds, de) for side, ds, de in doors if side == "left"]
                    right_doors = [(ds, de) for side, ds, de in doors if side == "right"]
                    top_doors = [(ds, de) for side, ds, de in doors if side == "top"]
                    bottom_doors = [(ds, de) for side, ds, de in doors if side == "bottom"]

                # Determine which walls are OUTER (building boundary) vs INNER (shared)
                # A wall is outer if no other room's edge matches it within tolerance
                BOUNDARY_TOL = 0.08
                def _is_outer_wall(wall_coord, is_vertical, room_idx):
                    """Check if this wall edge is on the building boundary (no neighbor)."""
                    for j in range(len(room_rects)):
                        if j == room_idx or rooms[j].get("annotation_only"):
                            continue
                        jx1, jy1, jx2, jy2 = room_rects[j]
                        if is_vertical:
                            # Check if any other room has a matching vertical edge
                            if abs(jx1 - wall_coord) < BOUNDARY_TOL or abs(jx2 - wall_coord) < BOUNDARY_TOL:
                                # Check vertical overlap
                                iy1, iy2 = room_rects[room_idx][1], room_rects[room_idx][3]
                                v_overlap = min(iy2, jy2) - max(iy1, jy1)
                                if v_overlap > 0.1:
                                    return False
                        else:
                            # Check if any other room has a matching horizontal edge
                            if abs(jy1 - wall_coord) < BOUNDARY_TOL or abs(jy2 - wall_coord) < BOUNDARY_TOL:
                                ix1, ix2 = room_rects[room_idx][0], room_rects[room_idx][2]
                                h_overlap = min(ix2, jx2) - max(ix1, jx1)
                                if h_overlap > 0.1:
                                    return False
                    return True

                left_is_outer = _is_outer_wall(x1, True, i)
                right_is_outer = _is_outer_wall(x2, True, i)
                bottom_is_outer = _is_outer_wall(y1, False, i)
                top_is_outer = _is_outer_wall(y2, False, i)

                def draw_wall_segments(p1x, p1y, p2x, p2y, gaps, is_vertical_wall, is_outer=False):
                    """Draw wall using native Wall shape if available, else fallback to DrawLine."""
                    weight = WALL_WEIGHT_OUTER if is_outer else WALL_WEIGHT
                    
                    def draw_segment(bx, by, ex, ey):
                        if wall_master_name:
                            try:
                                shape = page.Drop(doc.Masters.Item(wall_master_name), 0, 0)
                                shape.Cells("BeginX").ResultIU = bx
                                shape.Cells("BeginY").ResultIU = by
                                shape.Cells("EndX").ResultIU = ex
                                shape.Cells("EndY").ResultIU = ey
                                shape.Cells("LineColor").FormulaU = "0"
                                return shape
                            except Exception as e:
                                print(f"[VISIO] Failed to drop native wall segment: {e}")
                        
                        # Fallback to DrawLine
                        line = page.DrawLine(bx, by, ex, ey)
                        line.Cells("LineWeight").FormulaU = weight
                        line.Cells("LineColor").FormulaU = WALL_RGB
                        return line

                    if not gaps:
                        draw_segment(p1x, p1y, p2x, p2y)
                        return
                    # Sort gaps along the wall
                    if is_vertical_wall:
                        gaps_sorted = sorted(gaps, key=lambda g: g[0])
                        cur = min(p1y, p2y)
                        wall_end = max(p1y, p2y)
                        wx = p1x
                        for gs, ge in gaps_sorted:
                            if gs > cur:
                                draw_segment(wx, cur, wx, gs)
                            cur = ge
                        if cur < wall_end:
                            draw_segment(wx, cur, wx, wall_end)
                    else:
                        gaps_sorted = sorted(gaps, key=lambda g: g[0])
                        cur = min(p1x, p2x)
                        wall_end = max(p1x, p2x)
                        wy = p1y
                        for gs, ge in gaps_sorted:
                            if gs > cur:
                                draw_segment(cur, wy, gs, wy)
                            cur = ge
                        if cur < wall_end:
                            draw_segment(cur, wy, wall_end, wy)

                # Always draw the clean editable Visio walls. YOLO contours are
                # useful evidence, but replacing walls with raw mask outlines
                # loses door gaps, wall weights, and the professional visual style.
                draw_wall_segments(x1, y1, x1, y2, left_doors, is_vertical_wall=True, is_outer=left_is_outer)
                draw_wall_segments(x2, y1, x2, y2, right_doors, is_vertical_wall=True, is_outer=right_is_outer)
                draw_wall_segments(x1, y1, x2, y1, bottom_doors, is_vertical_wall=False, is_outer=bottom_is_outer)
                draw_wall_segments(x1, y2, x2, y2, top_doors, is_vertical_wall=False, is_outer=top_is_outer)

                # Optional diagnostic overlay for model mask contours. Disabled
                # by default because it is not suitable for production drawings.
                contour = room.get("contour") or []
                draw_model_contours = os.environ.get("VISIO_DRAW_MODEL_CONTOURS", "false").strip().lower() in {"1", "true", "yes", "on"}
                if draw_model_contours and contour and len(contour) >= 3:
                    try:
                        ofidx_for_room = room.get("_orig_floor_idx", room.get("floor_idx", 0)) if (page_use_grid and len(rooms) > 1) else int(room.get("floor_idx", 0) or 0)
                        pts = []
                        for pt in contour:
                            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                vx, vy = to_visio(float(pt[0]), float(pt[1]), ofidx_for_room)
                                pts.append((vx, vy))
                        if len(pts) >= 3:
                            for ci in range(len(pts)):
                                bx, by = pts[ci]
                                ex, ey = pts[(ci + 1) % len(pts)]
                                seg = page.DrawLine(bx, by, ex, ey)
                                seg.Cells("LineWeight").FormulaU = "0.25 pt"
                                seg.Cells("LineColor").FormulaU = "RGB(80,80,80)"
                                seg.Cells("LinePattern").FormulaU = "2"
                    except Exception as e:
                        print(f"[VISIO] Failed to draw diagnostic room contour for {label}: {e}")

                # Room label with name and dimensions (PropertyBox style)
                display_name = label if label else f"Room {room_num}"
                room_w_in = x2 - x1
                room_h_in = y2 - y1

                # Dimensions - prefer surveyor-written measurements over estimates.
                measured_w = room.get("measured_width_m")
                measured_h = room.get("measured_height_m")
                dim_source = room.get("dimension_source") or ("measured" if (measured_w or measured_h) else "estimated")
                if measured_w and measured_h:
                    width_m, height_m = float(measured_w), float(measured_h)
                else:
                    pixel_scale = detected.get("pixel_scale", 0)
                    bbox = room.get("bbox", [])
                    if pixel_scale > 0 and bbox and len(bbox) >= 4:
                        width_m = bbox[2] * pixel_scale
                        height_m = bbox[3] * pixel_scale
                    else:
                        SCALE_M_PER_IN = 1.2
                        width_m = room_w_in * SCALE_M_PER_IN
                        height_m = room_h_in * SCALE_M_PER_IN

                def _m_to_ft_in(m):
                    total_in = m * 39.3701
                    ft = int(total_in // 12)
                    inches = int(total_in % 12)
                    return f"{ft}' {inches}\""

                # Estimated dimensions get a prefix so readers can tell them apart.
                if dim_source == "measured":
                    dim_text = f"{width_m:.2f}m x {height_m:.2f}m\n({_m_to_ft_in(width_m)} x {_m_to_ft_in(height_m)})"
                else:
                    dim_text = f"(est. {width_m:.2f}m x {height_m:.2f}m)\n({_m_to_ft_in(width_m)} x {_m_to_ft_in(height_m)})"

                # Adaptive font size based on room area
                room_area_in = room_w_in * room_h_in
                if room_area_in > 8:
                    name_font = "10 pt"
                    num_font = "12 pt"
                    dim_font = "7 pt"
                elif room_area_in > 3:
                    name_font = "8 pt"
                    num_font = "10 pt"
                    dim_font = "6 pt"
                else:
                    name_font = "6 pt"
                    num_font = "8 pt"
                    dim_font = "5 pt"

                # Room number (large, bold, centered)
                ofidx_for_room = room.get("_orig_floor_idx", room.get("floor_idx", 0)) if (page_use_grid and len(rooms) > 1) else int(room.get("floor_idx", 0) or 0)
                if room.get("label_bbox") and has_bbox:
                    lbl_bbox = room["label_bbox"]
                    lvx1, lvy1 = to_visio(lbl_bbox[0], lbl_bbox[1], ofidx_for_room)
                    lvx2, lvy2 = to_visio(lbl_bbox[0] + lbl_bbox[2], lbl_bbox[1] + lbl_bbox[3], ofidx_for_room)
                    lbl_x1, lbl_y1 = min(lvx1, lvx2), min(lvy1, lvy2)
                    lbl_x2, lbl_y2 = max(lvx1, lvx2), max(lvy1, lvy2)
                else:
                    # Center a compact text mask rectangle inside the room
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    lbl_w = min(x2 - x1 - 0.08 * scale_mult, 1.8 * scale_mult)
                    lbl_h = min(y2 - y1 - 0.08 * scale_mult, 1.0 * scale_mult)
                    lbl_x1 = cx - lbl_w / 2
                    lbl_x2 = cx + lbl_w / 2
                    lbl_y1 = cy - lbl_h / 2
                    lbl_y2 = cy + lbl_h / 2
                draw_stair_symbol = os.environ.get("VISIO_DRAW_STAIR_SYMBOL", "true").strip().lower() in {"1", "true", "yes", "on"}
                stair_connector_room = bool(is_stairs and draw_stair_symbol and compact_connector and (connector_stair_label or adjacency_stair_connector))
                if not stair_connector_room:
                    num_shape = page.DrawRectangle(lbl_x1, lbl_y1, lbl_x2, lbl_y2)
                    exclude_sizes = os.environ.get("EXCLUDE_ROOM_SIZES", "true").lower() == "true"
                    if exclude_sizes:
                        num_shape.Text = f"{room_num}\n{display_name}"
                    else:
                        num_shape.Text = f"{room_num}\n{display_name}\n{dim_text}"
                    num_shape.Cells("Char.Size").FormulaU = name_font
                    num_shape.Cells("Char.Style").FormulaU = "1"  # Bold
                    num_shape.Cells("Char.Color").FormulaU = "RGB(30,30,30)"
                    num_shape.Cells("Para.HorzAlign").FormulaU = "1"  # Center
                    num_shape.Cells("VerticalAlign").FormulaU = "1"  # Middle
                    num_shape.Cells("LinePattern").FormulaU = "0"
                    num_shape.Cells("FillPattern").FormulaU = "0"

                if is_stairs and draw_stair_symbol:
                    stair_rect = None
                    stair_bbox = room.get("stairs_bbox")
                    if stair_bbox and len(stair_bbox) >= 4 and has_bbox:
                        svx1, svy1 = to_visio(stair_bbox[0], stair_bbox[1], ofidx_for_room)
                        svx2, svy2 = to_visio(stair_bbox[0] + stair_bbox[2], stair_bbox[1] + stair_bbox[3], ofidx_for_room)
                        stair_rect = (
                            min(svx1, svx2),
                            min(svy1, svy2),
                            max(svx1, svx2),
                            max(svy1, svy2),
                        )
                    if stair_rect is None:
                        # If the extractor only knows that this room contains stairs,
                        # keep the room geometry intact and draw a compact stair symbol
                        # inside it. Expanding the stair master to the whole room makes
                        # landings and bedrooms look like full staircases on real surveys.
                        room_w = max(x2 - x1, 0.01)
                        room_h = max(y2 - y1, 0.01)
                        inset = min(room_w, room_h) * (0.04 if stair_connector_room else 0.08)
                        if stair_connector_room:
                            stair_w = room_w - inset * 2
                            stair_h = room_h - inset * 2
                        elif room_h >= room_w:
                            stair_w = max(min(room_w * 0.58, 0.82 * scale_mult), room_w * 0.34)
                            stair_h = max(min(room_h * 0.42, 1.05 * scale_mult), room_h * 0.24)
                        else:
                            stair_w = max(min(room_w * 0.42, 1.05 * scale_mult), room_w * 0.24)
                            stair_h = max(min(room_h * 0.58, 0.82 * scale_mult), room_h * 0.34)
                        sx1 = x1 + (room_w - stair_w) / 2.0
                        sy1 = y1 + (room_h - stair_h) / 2.0
                        sx2 = sx1 + stair_w
                        sy2 = sy1 + stair_h
                        stair_rect = (
                            max(x1 + inset, sx1),
                            max(y1 + inset, sy1),
                            min(x2 - inset, sx2),
                            min(y2 - inset, sy2),
                        )
                    sx1, sy1, sx2, sy2 = stair_rect
                    # Feature detection: check if preloaded stair master is available
                    available_masters = [doc.Masters.Item(i).Name for i in range(1, doc.Masters.Count + 1)]
                    use_native_stairs = os.environ.get("VISIO_USE_NATIVE_STAIR_MASTER", "false").strip().lower() in {"1", "true", "yes", "on"}
                    if use_native_stairs and "Straight staircase.19" in available_masters:
                        try:
                            cx = (sx1 + sx2) / 2
                            cy = (sy1 + sy2) / 2
                            room_w = sx2 - sx1
                            room_h = sy2 - sy1
                            
                            # Determine orientation by aspect ratio and drop based on LocPinX=0 start corner
                            if room_h > room_w:
                                drop_x = cx
                                drop_y = sy1
                                shape = page.Drop(doc.Masters.Item("Straight staircase.19"), drop_x, drop_y)
                                shape.Cells("Width").ResultIU = room_h
                                shape.Cells("Height").ResultIU = room_w
                                shape.Cells("Angle").ResultIU = 1.57079  # 90 degrees
                                try:
                                    shape.Cells("Prop.FlightRun").ResultIU = room_h
                                except Exception:
                                    pass
                            else:
                                drop_x = sx1
                                drop_y = cy
                                shape = page.Drop(doc.Masters.Item("Straight staircase.19"), drop_x, drop_y)
                                shape.Cells("Width").ResultIU = room_w
                                shape.Cells("Height").ResultIU = room_h
                                shape.Cells("Angle").ResultIU = 0.0      # 0 degrees
                                try:
                                    shape.Cells("Prop.FlightRun").ResultIU = room_w
                                except Exception:
                                    pass
                            
                            # Configure properties to hide handrails and breaks for neat professional style
                            try:
                                shape.Cells("Prop.HideRail").FormulaU = "TRUE"
                                shape.Cells("Prop.HideBreak").FormulaU = "TRUE"
                                shape.Cells("Prop.TreadNumber").ResultIU = 12.0
                            except Exception:
                                pass
                            print(f"[VISIO] Dropped native Straight staircase.19 shape ID {shape.ID}")
                        except Exception as e:
                            print(f"[VISIO] Error dropping native stairs: {e}")
                    else:
                        # Use the same template-style editable stair symbol everywhere.
                        draw_survey_stair_symbol(sx1, sy1, sx2, sy2, direction="up")

                if is_loft:
                    pad = 0.05 * scale_mult
                    diag1 = page.DrawLine(x1 + pad, y1 + pad, x2 - pad, y2 - pad)
                    diag1.Cells("LineWeight").FormulaU = "0.5 pt"
                    diag1.Cells("LineColor").FormulaU = "RGB(0,0,0)"
                    diag2 = page.DrawLine(x2 - pad, y1 + pad, x1 + pad, y2 - pad)
                    diag2.Cells("LineWeight").FormulaU = "0.5 pt"
                    diag2.Cells("LineColor").FormulaU = "RGB(0,0,0)"
                    # Queue stair-access marker for main floor side nearest loft.
                    pending_stair_access.append((x1, (y1 + y2) / 2.0))


            # ---- INFERRED STAIR GAP SYMBOLS ----
            # Some surveys draw stairs as the narrow blank band between a wet/bath room
            # and a lobby/landing, not as a separately labelled room. Draw a stair symbol
            # in that gap when the surrounding room semantics make it clear.
            if os.environ.get("VISIO_INFER_STAIR_GAPS", "true").strip().lower() in {"1", "true", "yes", "on"}:
                drawn_gap_keys = set()
                def _is_wet_label(text):
                    return any(k in text for k in ["wet", "bath", "wc"])
                def _is_connector_label(text):
                    return any(k in text for k in ["lobby", "landing", "hall", "corridor"])
                for ai, aroom in enumerate(rooms):
                    alabel = str(aroom.get("label") or "").lower()
                    if not (_is_wet_label(alabel) or _is_connector_label(alabel)):
                        continue
                    ax1, ay1, ax2, ay2 = room_rects[ai]
                    for bi, broom in enumerate(rooms):
                        if bi <= ai:
                            continue
                        blabel = str(broom.get("label") or "").lower()
                        if not ((_is_wet_label(alabel) and _is_connector_label(blabel)) or (_is_connector_label(alabel) and _is_wet_label(blabel))):
                            continue
                        bx1, by1, bx2, by2 = room_rects[bi]
                        overlap_x = min(ax2, bx2) - max(ax1, bx1)
                        if overlap_x <= min(ax2 - ax1, bx2 - bx1) * 0.45:
                            continue
                        if ay1 >= by2:
                            gap_y1, gap_y2 = by2, ay1
                        elif by1 >= ay2:
                            gap_y1, gap_y2 = ay2, by1
                        else:
                            continue
                        gap_h = gap_y2 - gap_y1
                        if not (0.08 * scale_mult <= gap_h <= 0.95 * scale_mult):
                            continue
                        sx1 = max(ax1, bx1) + 0.04 * scale_mult
                        sx2 = min(ax2, bx2) - 0.04 * scale_mult
                        sy1 = gap_y1 + 0.04 * scale_mult
                        sy2 = gap_y2 - 0.04 * scale_mult
                        if sx2 <= sx1 or sy2 <= sy1:
                            continue
                        key = (round(sx1, 2), round(sy1, 2), round(sx2, 2), round(sy2, 2))
                        if key in drawn_gap_keys:
                            continue
                        drawn_gap_keys.add(key)
                        draw_survey_stair_symbol(sx1, sy1, sx2, sy2)

            # Draw deferred stair-access markers on main floor nearest to loft.
            for loft_x1, loft_mid_y in pending_stair_access:
                target_idx = None
                best_score = -1e9
                for idx, room in enumerate(rooms):
                    label_lower = (room.get("label") or "").lower()
                    if any(kw in label_lower for kw in ["loft", "attic", "roof space"]):
                        continue
                    rx1, ry1, rx2, ry2 = room_rects[idx]
                    effective_room_num = str(room.get("room_number") or room_numbers.get(idx, "")).strip()
                    has_corridor_hint = any(
                        kw in label_lower for kw in ["corridor", "hall", "hallway", "landing", "office", "conduit"]
                    )
                    y_overlap = 1 if (ry1 <= loft_mid_y <= ry2) else 0
                    left_of_loft = 1 if rx2 <= loft_x1 + 0.2 else 0
                    # Priority order:
                    # 1) explicit room 001 (main floor anchor)
                    # 2) corridor/hall/office labels
                    # 3) geometry near loft boundary with Y-overlap
                    # 4) closeness as tie-breaker
                    score = 0.0
                    if effective_room_num == "001":
                        score += 10000.0
                    if has_corridor_hint:
                        score += 4000.0
                    score += y_overlap * 1500.0
                    score += left_of_loft * 800.0
                    score -= abs(((rx1 + rx2) / 2.0) - loft_x1)
                    if score > best_score:
                        best_score = score
                        target_idx = idx

                if target_idx is None:
                    continue

                tx1, ty1, tx2, ty2 = room_rects[target_idx]
                room_w = max(tx2 - tx1, 0.01)
                room_h = max(ty2 - ty1, 0.01)
                stair_w = min(max(room_w * 0.34, 0.55 * scale_mult), room_w * 0.72)
                stair_h = min(max(room_h * 0.22, 0.32 * scale_mult), room_h * 0.48)
                sx2 = tx2 - 0.06 * scale_mult
                sx1 = max(tx1 + 0.06 * scale_mult, sx2 - stair_w)
                sy1 = ty1 + room_h * 0.28
                sy2 = min(ty2 - 0.06 * scale_mult, sy1 + stair_h)
                if sx2 > sx1 and sy2 > sy1:
                    draw_survey_stair_symbol(sx1, sy1, sx2, sy2, direction="up")

            # ---- DOOR ARCS (native Door SmartShape or polyline fallback) ----
            # First, check if there are preloaded Door masters in the document
            available_masters = [doc.Masters.Item(i).Name for i in range(1, doc.Masters.Count + 1)]
            door_master_name = None
            for name in ["Door.20", "Door.9", "Door"]:
                if name in available_masters:
                    door_master_name = name
                    break
            
            if door_master_name:
                print(f"[VISIO] Using native Visio master '{door_master_name}' for doors")
                for arc_x, arc_y, arc_r, arc_type, door_rgb in door_arcs:
                    try:
                        # Determine position and rotation based on the center of the gap (since LocPin is Width*0.5, Height*0.5)
                        if arc_type == "vertical_right":
                            px = arc_x
                            py = arc_y + arc_r / 2
                            angle = 1.57079  # 90 degrees
                        else:
                            px = arc_x - arc_r / 2
                            py = arc_y
                            angle = 0.0      # 0 degrees
                        
                        shape = page.Drop(doc.Masters.Item(door_master_name), px, py)
                        # Set size (width = door gap, height = wall thickness)
                        shape.Cells("Width").ResultIU = arc_r
                        shape.Cells("Height").ResultIU = 0.03937 * scale_mult
                        shape.Cells("Angle").ResultIU = angle
                        # Try to configure custom properties for beautiful display
                        try:
                            # 50% open percentage for clean visual swing line matching manual templates
                            shape.Cells("Prop.VisDoorOpenPercent").ResultIU = 50
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[VISIO] Error dropping native door: {e}")
            else:
                # Fallback to manual line segments drawing (original behavior)
                import math as _math
                ARC_SEGMENTS = 12
                for arc_x, arc_y, arc_r, arc_type, door_rgb in door_arcs:
                    try:
                        pts = []
                        if arc_type == "vertical_right":
                            for seg in range(ARC_SEGMENTS + 1):
                                angle = _math.pi / 2 * (1 - seg / ARC_SEGMENTS)
                                px = arc_x + arc_r * _math.cos(angle)
                                py = arc_y + arc_r * _math.sin(angle)
                                pts.append((px, py))
                        elif arc_type == "horizontal_up":
                            for seg in range(ARC_SEGMENTS + 1):
                                angle = _math.pi / 2 + _math.pi / 2 * (1 - seg / ARC_SEGMENTS)
                                px = arc_x + arc_r * _math.cos(angle)
                                py = arc_y + arc_r * _math.sin(angle)
                                pts.append((px, py))
                        for k in range(len(pts) - 1):
                            seg = page.DrawLine(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1])
                            seg.Cells("LineWeight").FormulaU = "0.65 pt"
                            seg.Cells("LineColor").FormulaU = door_rgb
                    except Exception as e:
                        print(f"[VISIO] Door arc error: {e}")

            # ---- MODEL-DETECTED DOORS (supplement heuristic shared-wall doors) ----
            # If the pipeline passed detected door positions from the model, draw extra
            # door arcs at those locations (only if not already near a heuristic door).
            model_doors = detected.get("doors") or []
            if model_doors and has_bbox:
                existing_door_centers = []
                for arc_x, arc_y, arc_r, arc_type, door_rgb in door_arcs:
                    existing_door_centers.append((arc_x, arc_y))

                for md in model_doors:
                    md_bbox = md.get("bbox")
                    if not md_bbox:
                        continue
                    # Convert pixel center to Visio coords
                    dx_px = md_bbox[0] + md_bbox[2] // 2
                    dy_px = md_bbox[1] + md_bbox[3] // 2
                    dvx, dvy = to_visio(dx_px, dy_px)

                    # Skip if too close to an existing heuristic door
                    too_close = False
                    for ex, ey in existing_door_centers:
                        if ((dvx - ex) ** 2 + (dvy - ey) ** 2) ** 0.5 < 0.5 * scale_mult:
                            too_close = True
                            break
                    if too_close:
                        continue

                    # Drop native door if available, else fallback to manual arc
                    if door_master_name:
                        try:
                            # Guess orientation from bbox aspect ratio
                            if md_bbox[2] > md_bbox[3]:
                                px = dvx
                                py = dvy
                                angle = 0.0
                            else:
                                px = dvx
                                py = dvy
                                angle = 1.57079
                            
                            shape = page.Drop(doc.Masters.Item(door_master_name), px, py)
                            # Set size (width = standard gap size, height = wall thickness)
                            shape.Cells("Width").ResultIU = 0.3543 * scale_mult
                            shape.Cells("Height").ResultIU = 0.03937 * scale_mult
                            shape.Cells("Angle").ResultIU = angle
                            try:
                                shape.Cells("Prop.VisDoorOpenPercent").ResultIU = 50
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"[VISIO] Error dropping native model door: {e}")
                    else:
                        # Draw a small door arc at this location
                        try:
                            pts = []
                            for seg in range(ARC_SEGMENTS + 1):
                                angle = _math.pi / 2 * (1 - seg / ARC_SEGMENTS)
                                px = dvx + ARC_R * 0.7 * _math.cos(angle)
                                py = dvy + ARC_R * 0.7 * _math.sin(angle)
                                pts.append((px, py))
                            for k in range(len(pts) - 1):
                                seg = page.DrawLine(pts[k][0], pts[k][1],
                                                    pts[k + 1][0], pts[k + 1][1])
                                seg.Cells("LineWeight").FormulaU = "0.5 pt"
                                seg.Cells("LineColor").FormulaU = DOOR_DEFAULT_RGB
                        except Exception:
                            pass

            # ---- MODEL-DETECTED WINDOWS (double parallel lines on exterior walls) ----
            model_windows = detected.get("windows") or []
            if model_windows and has_bbox:
                for mw in model_windows:
                    mw_bbox = mw.get("bbox")
                    if not mw_bbox:
                        continue
                    wx_px = mw_bbox[0] + mw_bbox[2] // 2
                    wy_px = mw_bbox[1] + mw_bbox[3] // 2
                    wvx, wvy = to_visio(wx_px, wy_px)
                    win_half = 0.25 * scale_mult  # half-width of window marker
                    # Determine orientation from aspect ratio
                    if mw_bbox[2] > mw_bbox[3]:
                        # Horizontal window
                        try:
                            for offset in [-0.02 * scale_mult, 0.02 * scale_mult]:
                                line = page.DrawLine(wvx - win_half, wvy + offset,
                                                     wvx + win_half, wvy + offset)
                                line.Cells("LineWeight").FormulaU = "0.7 pt"
                                line.Cells("LineColor").FormulaU = "RGB(0,120,200)"
                        except Exception:
                            pass
                    else:
                        # Vertical window
                        try:
                            for offset in [-0.02 * scale_mult, 0.02 * scale_mult]:
                                line = page.DrawLine(wvx + offset, wvy - win_half,
                                                     wvx + offset, wvy + win_half)
                                line.Cells("LineWeight").FormulaU = "0.7 pt"
                                line.Cells("LineColor").FormulaU = "RGB(0,120,200)"
                        except Exception:
                            pass

            # ---- Bounding box of main rooms (exclude detached for centering) ----
            # Detached rooms (like Loft) are far to the right and shouldn't skew
            # the plan center used for sample annotation placement
            main_rects = []
            for i, r in enumerate(room_rects):
                fl = rooms[i].get("floor") or ""
                if len(floor_set) > 1 and fl in detached_floors:
                    continue
                main_rects.append(r)
            if not main_rects:
                main_rects = list(room_rects)  # fallback to all
            all_x1 = min(r[0] for r in main_rects)
            all_y1 = min(r[1] for r in main_rects)
            all_x2 = max(r[2] for r in main_rects)
            all_y2 = max(r[3] for r in main_rects)
            plan_cx = (all_x1 + all_x2) / 2

            # Build room number to Visio rect/center mapping for sample arrows.
            # Sample callouts should sit close to their target room. Using the
            # whole-plan edge creates long red lines across fragmented layouts.
            room_targets = {}
            for i, room in enumerate(rooms):
                if i < len(room_rects):
                    rx1, ry1, rx2, ry2 = room_rects[i]
                    num_raw = room.get("number")
                    num_clean = str(num_raw or "").strip().lstrip('0')
                    if num_clean:
                        room_targets[num_clean] = {
                            "rect": (rx1, ry1, rx2, ry2),
                            "center": ((rx1 + rx2) / 2, (ry1 + ry2) / 2),
                        }

            # ---- SAMPLE ANNOTATIONS ----
            if has_bbox and sample_details:
                slots_left = []
                slots_right = []
                SLOT_H = 0.50 * scale_mult

                for sd in sample_details:
                    if not sd.get("location"):
                        continue
                    target_num = str(sd.get("target_room_number") or "").strip().lstrip('0')
                    target_rect = None
                    if target_num in room_targets:
                        target_rect = room_targets[target_num]["rect"]
                        sx, sy = room_targets[target_num]["center"]
                    else:
                        sx, sy = to_visio(sd["location"][0], sd["location"][1], sd.get("target_floor_idx"))
                        sx, sy = _clamp_point_to_bounds(
                            sx, sy, (all_x1, all_y1, all_x2, all_y2)
                        )
                    sid = sd["id"]
                    material = sd.get("material") or ""

                    # Conditional label: "Ref S001" for cross-references, "S001" for originals.
                    # Real Acorn plans convey a confirmed-positive result purely through
                    # colour (callout text + arrow in red) -- they don't append a "+" to
                    # the label text itself, so we match that rather than decorating it.
                    label = f"Ref {sid}" if sd.get("is_reference") else sid
                    ann_lines = [label]
                    if material:
                        words = material.split()
                        wline = ""
                        for w in words:
                            if len(wline) + len(w) > 22:
                                ann_lines.append(wline.strip())
                                wline = w + " "
                            else:
                                wline += w + " "
                        if wline.strip():
                            ann_lines.append(wline.strip())

                    ann_text = "\n".join(ann_lines)
                    ann_h = max(0.3 * scale_mult, len(ann_lines) * 0.15 * scale_mult)
                    ann_w = 1.8 * scale_mult

                    if target_rect is not None:
                        rx1, ry1, rx2, ry2 = target_rect
                        pad = 0.10 * scale_mult
                        offset = 0.18 * scale_mult
                        place_left = sx < plan_cx
                        if place_left:
                            ann_x = rx1 - ann_w - offset
                            arrow_sx = ann_x + ann_w
                        else:
                            ann_x = rx2 + offset
                            arrow_sx = ann_x
                        ann_y = max(ry1 + pad, min(sy - ann_h / 2, ry2 - ann_h - pad))
                        if ann_x < MARGIN_L or ann_x + ann_w > PAGE_W - MARGIN_R:
                            ann_x = max(rx1 + pad, min(sx - ann_w / 2, rx2 - ann_w - pad))
                            if sy >= (all_y1 + all_y2) / 2:
                                ann_y = min(ry2 + offset, PAGE_H - MARGIN_T - ann_h)
                                arrow_sy = ann_y
                            else:
                                ann_y = max(ry1 - ann_h - offset, MARGIN_B)
                                arrow_sy = ann_y + ann_h
                            arrow_sx = ann_x + ann_w / 2
                        else:
                            ann_y = max(MARGIN_B, min(ann_y, PAGE_H - MARGIN_T - ann_h))
                            arrow_sy = ann_y + ann_h / 2
                        if arrow_sx <= rx1:
                            sx = rx1
                            sy = max(ry1 + pad, min(arrow_sy, ry2 - pad))
                        elif arrow_sx >= rx2:
                            sx = rx2
                            sy = max(ry1 + pad, min(arrow_sy, ry2 - pad))
                        elif arrow_sy >= ry2:
                            sx = max(rx1 + pad, min(arrow_sx, rx2 - pad))
                            sy = ry2
                        else:
                            sx = max(rx1 + pad, min(arrow_sx, rx2 - pad))
                            sy = ry1
                    elif sx < plan_cx:
                        ann_x = max(0.3 * scale_mult, all_x1 - ann_w - 0.3 * scale_mult)
                        ann_y = sy - ann_h / 2
                        for uy in slots_left:
                            if abs(ann_y - uy) < SLOT_H:
                                ann_y = uy + SLOT_H
                        ann_y = max(MARGIN_B, min(ann_y, PAGE_H - MARGIN_T - ann_h))
                        slots_left.append(ann_y)
                        arrow_sx = ann_x + ann_w
                        arrow_sy = ann_y + ann_h / 2
                    else:
                        ann_x = min(all_x2 + 0.3 * scale_mult, PAGE_W - ann_w - 0.3 * scale_mult)
                        ann_y = sy - ann_h / 2
                        for uy in slots_right:
                            if abs(ann_y - uy) < SLOT_H:
                                ann_y = uy + SLOT_H
                        ann_y = max(MARGIN_B, min(ann_y, PAGE_H - MARGIN_T - ann_h))
                        slots_right.append(ann_y)
                        arrow_sx = ann_x
                        arrow_sy = ann_y + ann_h / 2

                    # Colour reflects the lab result, matching real Acorn plans:
                    # a confirmed-positive sample's callout + arrow render in red;
                    # everything else (negative / not yet analysed) is plain black.
                    callout_color = "RGB(204,0,0)" if sd.get("acm_positive") else "RGB(20,20,20)"

                    ann_shape = page.DrawRectangle(ann_x, ann_y, ann_x + ann_w, ann_y + ann_h)
                    ann_shape.Text = ann_text
                    ann_shape.Cells("Char.Size").FormulaU = "7 pt"
                    ann_shape.Cells("Char.Style").FormulaU = "1"  # Bold
                    ann_shape.Cells("Char.Color").FormulaU = callout_color
                    ann_shape.Cells("LinePattern").FormulaU = "0"
                    ann_shape.Cells("FillPattern").FormulaU = "0"
                    ann_shape.Cells("Para.HorzAlign").FormulaU = "0"

                    arrow = page.DrawLine(arrow_sx, arrow_sy, sx, sy)
                    arrow.Cells("LineColor").FormulaU = callout_color
                    arrow.Cells("LineWeight").FormulaU = "0.5 pt"
                    arrow.Cells("EndArrow").FormulaU = "13"
                    arrow.Cells("EndArrowSize").FormulaU = "2"

            # ---- UTILITY MARKERS (positioned inside containing room) ----
            marker_defs = []
            if hints.get("detect_atm", True):
                marker_defs.append(("ATM", detected.get("atm_location"), "RGB(51,51,51)", "RGB(255,255,255)"))
                marker_defs.append(("DB", detected.get("db_location"), "RGB(0,100,0)", "RGB(255,255,255)"))
            if hints.get("detect_gas_meter", True):
                marker_defs.append(("GAS", detected.get("gas_meter"), "RGB(204,102,0)", "RGB(255,255,255)"))
            if hints.get("detect_water_stop_tap", True):
                marker_defs.append(("WATER", detected.get("water_stop_tap"), "RGB(0,0,180)", "RGB(255,255,255)"))
            ew, eh = 0.45 * scale_mult, 0.20 * scale_mult  # marker oval size in inches
            for mlabel, mpos, mfill, mtext_color in marker_defs:
                if not mpos:
                    continue
                if has_bbox and isinstance(mpos, (list, tuple)) and len(mpos) == 2:
                    # Find which room contains this marker (pixel coords)
                    mpx, mpy = mpos[0], mpos[1]
                    containing_rect = None
                    for rx1, ry1, rx2, ry2 in room_rects:
                        if rx1 <= to_visio(mpx, mpy)[0] <= rx2 and ry1 <= to_visio(mpx, mpy)[1] <= ry2:
                            containing_rect = (rx1, ry1, rx2, ry2)
                            break
                    if containing_rect:
                        # Place marker in bottom-left corner of containing room (avoids label overlap)
                        cr_x1, cr_y1, cr_x2, cr_y2 = containing_rect
                        mx = cr_x1 + 0.15 * scale_mult + ew / 2
                        my = cr_y1 + 0.15 * scale_mult + eh / 2
                    else:
                        # No containing room - use raw converted position
                        mx, my = to_visio(mpx, mpy)
                    m_shape = page.DrawOval(mx - ew/2, my - eh/2, mx + ew/2, my + eh/2)
                else:
                    mx = MARGIN_L + 0.15 * scale_mult
                    my = 0.3 * scale_mult
                    m_shape = page.DrawOval(mx, my, mx + ew, my + eh)
                m_shape.Text = mlabel
                m_shape.Cells("Char.Size").FormulaU = "7 pt"
                m_shape.Cells("Char.Style").FormulaU = "1"  # Bold
                m_shape.Cells("Char.Color").FormulaU = mtext_color
                m_shape.Cells("LineWeight").FormulaU = "0.5 pt"
                m_shape.Cells("LineColor").FormulaU = "RGB(0,0,0)"
                m_shape.Cells("FillForegnd").FormulaU = mfill
                m_shape.Cells("FillPattern").FormulaU = "1"
                print(f"[VISIO] Marker {mlabel} at ({mx:.2f},{my:.2f})")

            # ---- CAVEAT TEXT ANNOTATIONS ----
            caveat_list = detected.get("caveats") or []
            if has_bbox and caveat_list:
                caveat_slot_y = all_y1 - 0.1 * scale_mult  # Start above plan
                for ci, cav in enumerate(caveat_list):
                    cav_text = cav.get("text", "")
                    if not cav_text:
                        continue
                    cav_loc = cav.get("location")
                    # Position caveat text box above the plan, stacked
                    cav_w = min(3.0 * scale_mult, len(cav_text) * 0.06 * scale_mult + 0.4 * scale_mult)
                    cav_h = 0.25 * scale_mult
                    cav_x = all_x1 + ci * (cav_w + 0.3 * scale_mult)
                    cav_y = caveat_slot_y + ci * 0.35 * scale_mult
                    cav_y = min(cav_y, PAGE_H - MARGIN_T - cav_h)

                    cav_shape = page.DrawRectangle(cav_x, cav_y, cav_x + cav_w, cav_y + cav_h)
                    cav_shape.Text = cav_text
                    cav_shape.Cells("Char.Size").FormulaU = "6 pt"
                    cav_shape.Cells("Char.Style").FormulaU = "18"  # Bold + Italic
                    cav_shape.Cells("Char.Color").FormulaU = "RGB(0,0,153)"  # Dark blue
                    cav_shape.Cells("LinePattern").FormulaU = "0"
                    cav_shape.Cells("FillPattern").FormulaU = "0"
                    cav_shape.Cells("Para.HorzAlign").FormulaU = "0"

                    # Leader arrow from text to approximate location
                    if cav_loc and has_bbox:
                        cx, cy = to_visio(cav_loc[0], cav_loc[1])
                        arrow = page.DrawLine(cav_x + cav_w / 2, cav_y, cx, cy)
                        arrow.Cells("LineColor").FormulaU = "RGB(0,0,153)"
                        arrow.Cells("LineWeight").FormulaU = "0.3 pt"
                        arrow.Cells("EndArrow").FormulaU = "13"
                        arrow.Cells("EndArrowSize").FormulaU = "2"
                        arrow.Cells("LinePattern").FormulaU = "2"  # Dashed

                    print(f"[VISIO] Caveat: {cav_text[:40]}...")

            # ---- LEGEND ----
            # Real Acorn manual plans never show a colour-swatch legend (ACM
            # status is conveyed entirely through the sample text callouts
            # drawn with arrows to each wall/ceiling/floor). Off by default;
            # set VISIO_SHOW_LEGEND=true to restore it for a client who wants it.
            show_legend = _env_flag("VISIO_SHOW_LEGEND", False)
            legend_x = all_x2 - 2.5 * scale_mult if has_bbox else PAGE_W - MARGIN_R - 2.5 * scale_mult
            legend_y = 0.6 * scale_mult
            legend_items = [
                (FILL_ACM, "ACM Positive"),
                (FILL_NO_ACCESS, "No Access"),
                (FILL_CLEAR, "Clear / NAD"),
            ]
            if show_legend:
                legend_rows = len(legend_items) + (1 if (hints.get("detect_cable_route", True) and detected.get("has_cable_route")) else 0)
                legend_h = max(0.95 * scale_mult, legend_rows * 0.28 * scale_mult + 0.24 * scale_mult)
                legend_bg = page.DrawRectangle(legend_x - 0.08 * scale_mult, legend_y + 0.23 * scale_mult, legend_x + 1.58 * scale_mult, legend_y - legend_h)
                legend_bg.Cells("FillForegnd").FormulaU = "RGB(255,255,255)"
                legend_bg.Cells("FillPattern").FormulaU = "1"
                legend_bg.Cells("LineColor").FormulaU = "RGB(180,180,180)"
                legend_bg.Cells("LineWeight").FormulaU = "0.36 pt"
                legend_title = page.DrawRectangle(legend_x, legend_y + 0.08 * scale_mult, legend_x + 1.45 * scale_mult, legend_y + 0.22 * scale_mult)
                legend_title.Text = "Legend"
                legend_title.Cells("Char.Size").FormulaU = "8 pt"
                legend_title.Cells("Char.Style").FormulaU = "1"
                legend_title.Cells("Char.Color").FormulaU = "RGB(30,30,30)"
                legend_title.Cells("LinePattern").FormulaU = "0"
                legend_title.Cells("FillPattern").FormulaU = "0"
                legend_title.Cells("Para.HorzAlign").FormulaU = "0"
                for li, (lfill, llabel) in enumerate(legend_items):
                    lx = legend_x
                    ly = legend_y - li * 0.28 * scale_mult
                    swatch = page.DrawRectangle(lx, ly, lx + 0.2 * scale_mult, ly + 0.15 * scale_mult)
                    swatch.Cells("LineColor").FormulaU = "RGB(0,0,0)"
                    swatch.Cells("LineWeight").FormulaU = "0.36 pt"
                    swatch.Cells("FillBkgnd").FormulaU = "RGB(255,255,255)"
                    swatch.Cells("FillForegnd").FormulaU = lfill
                    swatch.Cells("FillPattern").FormulaU = "1"
                    lbl = page.DrawRectangle(lx + 0.25 * scale_mult, ly, lx + 1.5 * scale_mult, ly + 0.15 * scale_mult)
                    lbl.Text = llabel
                    lbl.Cells("Char.Size").FormulaU = "7.5 pt"
                    lbl.Cells("Char.Color").FormulaU = "RGB(25,25,25)"
                    lbl.Cells("LinePattern").FormulaU = "0"
                    lbl.Cells("FillPattern").FormulaU = "0"
                    lbl.Cells("Para.HorzAlign").FormulaU = "0"

            # ---- CABLE ROUTE (green dashed line from DB to ATM) ----
            cable_color_rgb = f"RGB({cable_rgb[0]},{cable_rgb[1]},{cable_rgb[2]})"
            if hints.get("detect_cable_route", True) and detected.get("has_cable_route"):
                atm_loc = detected.get("atm_location")
                db_loc = detected.get("db_location")
                if (has_bbox and atm_loc and db_loc
                        and isinstance(atm_loc, (list, tuple)) and len(atm_loc) >= 2
                        and isinstance(db_loc, (list, tuple)) and len(db_loc) >= 2):
                    # Draw green dashed line from DB to ATM
                    db_vx, db_vy = to_visio(db_loc[0], db_loc[1])
                    atm_vx, atm_vy = to_visio(atm_loc[0], atm_loc[1])
                    cable_line = page.DrawLine(db_vx, db_vy, atm_vx, atm_vy)
                    cable_line.Cells("LineColor").FormulaU = cable_color_rgb
                    cable_line.Cells("LineWeight").FormulaU = "1.5 pt"
                    cable_line.Cells("LinePattern").FormulaU = "2"  # Dashed
                    cable_line.Cells("EndArrow").FormulaU = "13"   # Filled triangle
                    cable_line.Cells("EndArrowSize").FormulaU = "2"
                    # Label at midpoint
                    mid_vx = (db_vx + atm_vx) / 2
                    mid_vy = (db_vy + atm_vy) / 2
                    cr_label = page.DrawRectangle(mid_vx - 0.5 * scale_mult, mid_vy - 0.15 * scale_mult, mid_vx + 0.5 * scale_mult, mid_vy + 0.1 * scale_mult)
                    cr_label.Text = "Cable Route"
                    cr_label.Cells("Char.Size").FormulaU = "6 pt"
                    cr_label.Cells("Char.Style").FormulaU = "18"  # Italic
                    cr_label.Cells("Char.Color").FormulaU = cable_color_rgb
                    cr_label.Cells("LinePattern").FormulaU = "0"
                    cr_label.Cells("FillPattern").FormulaU = "0"
                    print(f"[VISIO] Cable route: DB({db_vx:.2f},{db_vy:.2f}) -> ATM({atm_vx:.2f},{atm_vy:.2f})")

                # Legend entry for cable route
                if show_legend:
                    cr_legend = page.DrawRectangle(legend_x, legend_y - len(legend_items) * 0.28 * scale_mult,
                                                   legend_x + 1.5 * scale_mult, legend_y - len(legend_items) * 0.28 * scale_mult + 0.15 * scale_mult)
                    cr_legend.Text = "Cable Route (DB -> ATM)"
                    cr_legend.Cells("Char.Size").FormulaU = "7 pt"
                    cr_legend.Cells("Char.Style").FormulaU = "18"
                    cr_legend.Cells("Char.Color").FormulaU = cable_color_rgb
                    cr_legend.Cells("LinePattern").FormulaU = "0"
                    cr_legend.Cells("FillPattern").FormulaU = "0"

        print(f"[VISIO] Multi-page doc: {sum(len(v) for v in floor_groups.values())} rooms across {len(floor_order)} page(s)")

        # Save to a temporary file first. The document must be closed before
        # Python moves the VSDX; Visio keeps the saved file handle open until
        # Close/Quit completes, which otherwise raises WinError 32.
        final_output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(final_output), exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="acorn_visio_save_")
        temp_output = os.path.join(temp_dir, os.path.basename(final_output))
        try:
            doc.SaveAs(temp_output)
            print(f"[VISIO] Saved to temp: {temp_output}")

            try:
                doc.Close()
            except Exception as e:
                print(f"[VISIO] doc.Close() warning: {e}")
            try:
                visio.Quit()
            except Exception as e:
                print(f"[VISIO] visio.Quit() warning: {e}")
            doc = None
            visio = None
            gc.collect()

            # Visio can keep a brief file lock on the saved doc even after
            # Quit() returns; retry the move so a fast box doesn't hit
            # WinError 32 ("file in use by another process").
            last_err = None
            for _attempt in range(10):
                try:
                    shutil.move(temp_output, final_output)
                    last_err = None
                    break
                except OSError as e:
                    last_err = e
                    time.sleep(0.5)
            if last_err is not None:
                raise last_err

            print(f"[VISIO] Saved: {final_output}")
            return final_output
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"[VISIO] Professional Visio export failed: {e}")
        return None
    finally:
        try:
            if visio is not None:
                visio.Quit()
        except Exception:
            pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass
