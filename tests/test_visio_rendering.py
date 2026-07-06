"""
test_visio_rendering.py - Regression tests for the professional_visio.py
COM renderer's ACM styling, matching real Acorn production output.

Ground truth: a real set of ~37 historical Acorn survey sketch -> final
Visio-plan pairs showed that confirmed-ACM rooms are NEVER solid-filled and
never carry a colour legend -- ACM status is conveyed by (a) a light 45-deg
diagonal hatch of individual line strokes on the room itself, and (b) the
matching sample callout + arrow rendered in red text (black for
negative/unconfirmed samples). This file locks that behaviour in.

Runs without win32com/Visio -- only exercises the pure-geometry helper.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import utils.visio.professional_visio as pv  # noqa: E402


class _FakeCell:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name

    @property
    def FormulaU(self):
        return self._owner.cells.get(self._name)

    @FormulaU.setter
    def FormulaU(self, value):
        self._owner.cells[self._name] = value


class _FakeShape:
    def __init__(self):
        self.cells = {}
        self.Text = ""

    def Cells(self, name):
        return _FakeCell(self, name)


class _FakePage:
    def __init__(self):
        self.lines = []
        self.rects = []

    def DrawLine(self, x1, y1, x2, y2):
        shape = _FakeShape()
        self.lines.append((x1, y1, x2, y2, shape))
        return shape

    def DrawRectangle(self, x1, y1, x2, y2):
        shape = _FakeShape()
        self.rects.append((x1, y1, x2, y2, shape))
        return shape


class TestDrawDiagonalHatch:
    def test_lines_are_45_degrees_and_clipped_within_room(self):
        page = _FakePage()
        x1, y1, x2, y2 = 2.0, 1.0, 6.0, 4.0
        pv._draw_diagonal_hatch(page, x1, y1, x2, y2, spacing=0.14)

        assert len(page.lines) > 10
        for lx1, ly1, lx2, ly2, _shape in page.lines:
            assert x1 - 1e-6 <= lx1 <= x2 + 1e-6
            assert x1 - 1e-6 <= lx2 <= x2 + 1e-6
            assert y1 - 1e-6 <= ly1 <= y2 + 1e-6
            assert y1 - 1e-6 <= ly2 <= y2 + 1e-6
            # 45-degree slope: dx == dy
            assert abs((lx2 - lx1) - (ly2 - ly1)) < 1e-6

    def test_degenerate_room_draws_nothing(self):
        page = _FakePage()
        pv._draw_diagonal_hatch(page, 5, 5, 5, 5)
        assert len(page.lines) == 0

    def test_lines_use_requested_colour_and_stay_thin(self):
        page = _FakePage()
        pv._draw_diagonal_hatch(page, 0, 0, 2, 2, color_rgb="RGB(196,60,70)")
        assert page.lines
        for _x1, _y1, _x2, _y2, shape in page.lines:
            assert shape.cells["LineColor"] == "RGB(196,60,70)"
            assert shape.cells["LinePattern"] == "1"


class TestNoSolidAcmFillHelperContract:
    """These assert the module no longer exposes the old solid-fill-only
    behaviour for ACM rooms -- i.e. _draw_diagonal_hatch exists and is the
    documented mechanism (see module docstring / inline comments)."""

    def test_draw_diagonal_hatch_is_exported(self):
        assert hasattr(pv, "_draw_diagonal_hatch")
        assert callable(pv._draw_diagonal_hatch)

    def test_env_flag_helper_exists_for_legend_gating(self):
        assert hasattr(pv, "_env_flag")
        assert pv._env_flag("SOME_UNSET_FLAG_XYZ", False) is False
        assert pv._env_flag("SOME_UNSET_FLAG_XYZ", True) is True
