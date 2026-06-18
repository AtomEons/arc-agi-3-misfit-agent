"""Wave 6 — cell-level color mapping + content cropping.

Diagnostic on 12 random Wave-5-unsolved tasks showed:
  Class B (33%): new color appears in output (need ColorMap inference)
  Class D (25%): shape change with bg removal (need CropToContent)
  Class A (33%): same shape/colors, small diff (need pattern completion — future wave)

Wave 6 attacks Classes B and D with deterministic enumeration.
Tier-1 strict throughout: no LLM, no pretrained weights, no learned params at eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..perceptor import _background_color


Grid = np.ndarray


# ---------------------------------------------------------------------------
# ColorMap — universal color permutation inferred from train pairs.
# ---------------------------------------------------------------------------

@dataclass
class ColorMap:
    """Apply a fixed color-to-color mapping inferred from train pairs.
    Each input color c maps to exactly one output color m[c].
    Tier-1 strict: the mapping is deterministically inferred by checking
    consistency across all train pairs; rejected if any input color maps
    to two different output colors.
    """
    mapping: tuple[int, ...] = field(default_factory=lambda: tuple(range(10)))
    fitted: bool = False

    def _infer(self, train_pairs: list[tuple[Grid, Grid]]) -> Optional[tuple[int, ...]]:
        m: dict[int, int] = {}
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return None
            for c_in, c_out in zip(inp.flatten().tolist(), out.flatten().tolist()):
                if c_in in m and m[c_in] != c_out:
                    return None
                m[c_in] = c_out
        result = [m.get(i, i) for i in range(10)]
        if result == list(range(10)):
            return None
        return tuple(result)

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        m = self._infer(train_pairs)
        if m is None:
            return False
        self.mapping = m
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        out = grid.copy()
        for c in range(10):
            if self.mapping[c] != c:
                out[grid == c] = self.mapping[c]
        return out

    def signature(self) -> tuple:
        return ("ColorMap", self.mapping)


# ---------------------------------------------------------------------------
# ColorReplace — single (src, dst) pair, no need for full permutation.
# ---------------------------------------------------------------------------

@dataclass
class ColorReplace:
    """Replace a single source color with a single destination color.
    Inferred deterministically by checking all (src, dst) pairs.
    """
    src: int = 0
    dst: int = 1
    fitted: bool = False

    def _apply(self, grid: Grid, src: int, dst: int) -> Grid:
        grid = np.asarray(grid)
        out = grid.copy()
        out[grid == src] = dst
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for src in range(10):
            for dst in range(10):
                if src == dst:
                    continue
                ok = True
                for inp, out in train_pairs:
                    inp = np.asarray(inp); out = np.asarray(out)
                    if inp.shape != out.shape:
                        ok = False
                        break
                    if not np.array_equal(self._apply(inp, src, dst), out):
                        ok = False
                        break
                if ok:
                    self.src, self.dst = src, dst
                    self.fitted = True
                    return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.src, self.dst)

    def signature(self) -> tuple:
        return ("ColorReplace", self.src, self.dst)


# ---------------------------------------------------------------------------
# CropToContent — extract bbox of non-bg cells.
# ---------------------------------------------------------------------------

@dataclass
class CropToContent:
    """Crop input to the bounding box of all non-background cells."""
    fitted: bool = False

    def _apply(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        mask = grid != bg
        if not mask.any():
            return grid.copy()
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        return grid[r0:r1+1, c0:c1+1].copy()

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            cropped = self._apply(inp)
            if cropped.shape != out.shape:
                return False
            if not np.array_equal(cropped, out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid)

    def signature(self) -> tuple:
        return ("CropToContent",)


# ---------------------------------------------------------------------------
# CropToColor — extract bbox of cells matching a target color.
# ---------------------------------------------------------------------------

@dataclass
class CropToColor:
    """Crop to bounding box of cells matching a target color (inferred)."""
    target: int = 1
    fitted: bool = False

    def _apply(self, grid: Grid, target: int) -> Grid:
        grid = np.asarray(grid)
        mask = grid == target
        if not mask.any():
            return grid.copy()
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        return grid[r0:r1+1, c0:c1+1].copy()

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for target in range(10):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                cropped = self._apply(inp, target)
                if cropped.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(cropped, out):
                    ok = False
                    break
            if ok:
                self.target = target
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.target)

    def signature(self) -> tuple:
        return ("CropToColor", self.target)


# ---------------------------------------------------------------------------
# KeepOnlyColor — keep cells matching target color, replace rest with bg.
# ---------------------------------------------------------------------------

@dataclass
class KeepOnlyColor:
    """Replace all cells NOT equal to target color with bg color."""
    target: int = 1
    fitted: bool = False

    def _apply(self, grid: Grid, target: int) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        out = np.full_like(grid, bg)
        out[grid == target] = target
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for target in range(10):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(self._apply(inp, target), out):
                    ok = False
                    break
            if ok:
                self.target = target
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.target)

    def signature(self) -> tuple:
        return ("KeepOnlyColor", self.target)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

ALL_WAVE6_RULES = [
    ColorMap,
    ColorReplace,
    CropToContent,
    CropToColor,
    KeepOnlyColor,
]
