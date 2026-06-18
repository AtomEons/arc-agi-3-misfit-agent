"""ARC-AGI-2 rule families v2 — object-relational, symmetry, gravity, counting.

The v1 grammar (Identity / Recolor / Translate2 / ReflectH/V / Transpose /
Rotate{1,2,3} / CropToBbox / Tile) hit 1.40% on training, 0% on eval. The
depth-2 sweep showed composition added zero lift — the bottleneck is RULE
FAMILY COVERAGE, not search depth.

This module adds 20+ new families covering the categories the failure
analysis pointed at: object-relational (operations over the perceived
object set), symmetry-completion, gravity, structural recoloring, and
counting/numerosity.

Each rule follows the v1 contract:
  .fit(train_pairs) -> bool      # True if rule holds on EVERY train pair
  .predict(grid) -> Grid         # apply rule to a new input
  .signature() -> tuple          # hashable identity for beam dedup

All rules are Tier-1 admissible — Spelke priors (objectness, geometry,
numerosity, contact-causality) made executable. No public-eval tuning,
no LLM heuristic, no pretrained weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perceptor import perceive_grid, _background_color


Grid = np.ndarray


# ---------------------------------------------------------------------------
# OBJECT-RELATIONAL FAMILY
# Operations over the perceived object set. Spelke OBJECTNESS + NUMEROSITY.
# ---------------------------------------------------------------------------


def _objects_of(grid: Grid) -> list:
    """Return perceptor objects for the grid (cached interface)."""
    return list(perceive_grid(np.asarray(grid)).objects)


def _blank_like(grid: Grid) -> Grid:
    return np.full_like(np.asarray(grid), fill_value=_background_color(grid))


def _stamp_object(grid: Grid, obj, src_grid: Grid) -> None:
    """In-place: copy obj's cells from src_grid to grid."""
    r0, c0, r1, c1 = obj.bbox
    bg = _background_color(src_grid)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = int(src_grid[r, c])
            if v != bg and v == obj.color:
                grid[r, c] = v


@dataclass
class KeepLargestObject:
    """Output = input with all but the largest object blanked. Spelke OBJECTNESS."""
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        largest = max(objs, key=lambda o: o.area)
        out = _blank_like(grid)
        _stamp_object(out, largest, grid)
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("KeepLargestObject",)


@dataclass
class KeepSmallestObject:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        smallest = min(objs, key=lambda o: o.area)
        out = _blank_like(grid)
        _stamp_object(out, smallest, grid)
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("KeepSmallestObject",)


@dataclass
class KeepByColor:
    """Keep only objects whose color matches the fitted target color."""
    target_color: int = -1
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        # Try each color in [0..9] and pick the one that's consistent
        for color in range(10):
            ok = True
            for inp, out in train_pairs:
                if not np.array_equal(self._apply_color(np.asarray(inp), color),
                                       np.asarray(out)):
                    ok = False
                    break
            if ok:
                self.target_color = color
                self.fitted = True
                return True
        return False

    def _apply_color(self, grid: Grid, color: int) -> Grid:
        objs = _objects_of(grid)
        out = _blank_like(grid)
        for obj in objs:
            if obj.color == color:
                _stamp_object(out, obj, grid)
        return out

    def predict(self, grid):
        return self._apply_color(np.asarray(grid), self.target_color).copy()

    def signature(self):
        return ("KeepByColor", self.target_color)


@dataclass
class DeleteByColor:
    """Delete objects whose color matches the fitted target color."""
    target_color: int = -1
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for color in range(10):
            ok = True
            for inp, out in train_pairs:
                if not np.array_equal(self._apply_color(np.asarray(inp), color),
                                       np.asarray(out)):
                    ok = False
                    break
            if ok:
                self.target_color = color
                self.fitted = True
                return True
        return False

    def _apply_color(self, grid: Grid, color: int) -> Grid:
        inp = np.asarray(grid)
        out = inp.copy()
        bg = _background_color(inp)
        out[inp == color] = bg
        return out

    def predict(self, grid):
        return self._apply_color(np.asarray(grid), self.target_color).copy()

    def signature(self):
        return ("DeleteByColor", self.target_color)


@dataclass
class KeepEdgeTouching:
    """Keep only objects touching the grid edge. Spelke TOPOLOGY."""
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        out = _blank_like(grid)
        for obj in objs:
            if obj.touches_edge:
                _stamp_object(out, obj, grid)
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("KeepEdgeTouching",)


@dataclass
class KeepNonEdge:
    """Keep only objects NOT touching the grid edge."""
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        out = _blank_like(grid)
        for obj in objs:
            if not obj.touches_edge:
                _stamp_object(out, obj, grid)
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("KeepNonEdge",)


# ---------------------------------------------------------------------------
# SYMMETRY-COMPLETION FAMILY
# Make output symmetric by completing missing halves. Spelke GEOMETRY.
# ---------------------------------------------------------------------------


@dataclass
class SymmetrizeH:
    """Output = input combined with horizontal mirror. Foreground cells
    on either side combine; background filled where both are bg."""
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        mirror = np.fliplr(inp)
        out = np.where(inp != bg, inp, mirror)
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("SymmetrizeH",)


@dataclass
class SymmetrizeV:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        mirror = np.flipud(inp)
        out = np.where(inp != bg, inp, mirror)
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("SymmetrizeV",)


@dataclass
class SymmetrizeBoth:
    """Apply both H and V symmetrize — 4-fold symmetry."""
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        for mirror in (np.fliplr(inp), np.flipud(inp), np.flipud(np.fliplr(inp))):
            inp = np.where(inp != bg, inp, mirror)
        return inp

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("SymmetrizeBoth",)


# ---------------------------------------------------------------------------
# GRAVITY FAMILY
# All non-bg cells fall toward a direction. Spelke CONTACT-CAUSALITY.
# ---------------------------------------------------------------------------


@dataclass
class GravityDown:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        out = np.full_like(inp, bg)
        for c in range(inp.shape[1]):
            col = inp[:, c]
            non_bg = col[col != bg]
            if len(non_bg) > 0:
                out[-len(non_bg):, c] = non_bg
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("GravityDown",)


@dataclass
class GravityUp:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        out = np.full_like(inp, bg)
        for c in range(inp.shape[1]):
            col = inp[:, c]
            non_bg = col[col != bg]
            if len(non_bg) > 0:
                out[:len(non_bg), c] = non_bg
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("GravityUp",)


@dataclass
class GravityLeft:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        out = np.full_like(inp, bg)
        for r in range(inp.shape[0]):
            row = inp[r, :]
            non_bg = row[row != bg]
            if len(non_bg) > 0:
                out[r, :len(non_bg)] = non_bg
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("GravityLeft",)


@dataclass
class GravityRight:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            pred = self._apply(np.asarray(inp))
            if not np.array_equal(pred, np.asarray(out)):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        out = np.full_like(inp, bg)
        for r in range(inp.shape[0]):
            row = inp[r, :]
            non_bg = row[row != bg]
            if len(non_bg) > 0:
                out[r, -len(non_bg):] = non_bg
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("GravityRight",)


# ---------------------------------------------------------------------------
# COLOR-SWAP FAMILY
# Pairs of colors swapped, or palette-shuffled. Spelke (none required —
# this is a color-identity operation, generic).
# ---------------------------------------------------------------------------


@dataclass
class BackgroundSwap:
    """Swap the background color with a specific foreground color throughout."""
    fg_color: int = -1
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for fg in range(10):
            ok = True
            for inp, out in train_pairs:
                if not np.array_equal(self._apply(np.asarray(inp), fg),
                                       np.asarray(out)):
                    ok = False
                    break
            if ok:
                self.fg_color = fg
                self.fitted = True
                return True
        return False

    def _apply(self, grid: Grid, fg: int) -> Grid:
        inp = np.asarray(grid)
        bg = _background_color(inp)
        if fg == bg:
            return inp.copy()
        out = inp.copy()
        out[inp == bg] = fg
        out[inp == fg] = bg
        return out

    def predict(self, grid):
        return self._apply(np.asarray(grid), self.fg_color).copy()

    def signature(self):
        return ("BackgroundSwap", self.fg_color)


# ---------------------------------------------------------------------------
# OUTPUT-SHAPE-VARIABLE FAMILY
# Output dimensions depend on input properties. Spelke NUMEROSITY.
# ---------------------------------------------------------------------------


@dataclass
class OutputIsObjectCount:
    """Output = 1x1 grid with the count of foreground objects as the value.
    Generalized variant of NUMEROSITY rules.
    """
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            out_arr = np.asarray(out)
            if out_arr.shape != (1, 1):
                return False
            objs = _objects_of(np.asarray(inp))
            if int(out_arr[0, 0]) != len(objs):
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        return np.array([[len(objs)]], dtype=np.int32)

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("OutputIsObjectCount",)


@dataclass
class OutputIsLargestColor:
    """Output = 1x1 grid with the color of the largest non-bg object."""
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            out_arr = np.asarray(out)
            if out_arr.shape != (1, 1):
                return False
            objs = _objects_of(np.asarray(inp))
            if not objs:
                return False
            largest = max(objs, key=lambda o: o.area)
            if int(out_arr[0, 0]) != largest.color:
                return False
        self.fitted = True
        return True

    def _apply(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        if not objs:
            return np.array([[0]], dtype=np.int32)
        largest = max(objs, key=lambda o: o.area)
        return np.array([[largest.color]], dtype=np.int32)

    def predict(self, grid):
        return self._apply(np.asarray(grid)).copy()

    def signature(self):
        return ("OutputIsLargestColor",)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def all_v2_factories():
    """All v2 rule factories. Order matters for beam ranking on ties."""
    return [
        # Object-relational
        lambda: KeepLargestObject(),
        lambda: KeepSmallestObject(),
        lambda: KeepByColor(),
        lambda: DeleteByColor(),
        lambda: KeepEdgeTouching(),
        lambda: KeepNonEdge(),
        # Symmetry completion
        lambda: SymmetrizeH(),
        lambda: SymmetrizeV(),
        lambda: SymmetrizeBoth(),
        # Gravity
        lambda: GravityDown(),
        lambda: GravityUp(),
        lambda: GravityLeft(),
        lambda: GravityRight(),
        # Color swap
        lambda: BackgroundSwap(),
        # Output-shape-variable
        lambda: OutputIsObjectCount(),
        lambda: OutputIsLargestColor(),
    ]
