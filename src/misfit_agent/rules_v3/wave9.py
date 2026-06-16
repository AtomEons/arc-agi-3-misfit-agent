"""Wave 9 — neighbor-aware and containment-aware conditional recolor.

Targets the 30% same_shape_recolor_global unsolved category — tasks where a
cell's recolor depends on its NEIGHBORHOOD or CONTAINMENT relation, not just
on a fixed color value.

Tier-1 strict throughout: deterministic enumeration over (src, neighbor, dst)
color tuples; rule accepted only if it applies identically on all train pairs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..perceptor import _background_color
from .wave4 import _objects_of


Grid = np.ndarray


# ---------------------------------------------------------------------------
# NeighborAwareRecolor — cell of color SRC adjacent to color NEIGH → DST.
# ---------------------------------------------------------------------------

@dataclass
class NeighborAwareRecolor:
    """For each cell of color SRC that has a 4-connected neighbor of color NEIGH,
    recolor that cell to DST. SRC, NEIGH, DST inferred from train pairs.
    """
    src: int = 0
    neigh: int = 1
    dst: int = 2
    fitted: bool = False

    def _apply(self, grid: Grid, src: int, neigh: int, dst: int) -> Grid:
        grid = np.asarray(grid)
        rows, cols = grid.shape
        out = grid.copy()
        # 4-connected neighbor mask: shift the neighbor color in 4 directions
        neigh_mask = np.zeros_like(grid, dtype=bool)
        neigh_grid = (grid == neigh)
        if rows > 1:
            neigh_mask[:-1, :] |= neigh_grid[1:, :]
            neigh_mask[1:, :]  |= neigh_grid[:-1, :]
        if cols > 1:
            neigh_mask[:, :-1] |= neigh_grid[:, 1:]
            neigh_mask[:, 1:]  |= neigh_grid[:, :-1]
        mask = (grid == src) & neigh_mask
        out[mask] = dst
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        # Restrict src,neigh to colors that ACTUALLY appear in train inputs;
        # restrict dst to colors that appear in train outputs.
        in_colors = set()
        out_colors = set()
        for inp, out in train_pairs:
            in_colors.update(np.asarray(inp).flatten().tolist())
            out_colors.update(np.asarray(out).flatten().tolist())
        for src in sorted(in_colors):
            for neigh in sorted(in_colors):
                if src == neigh:
                    continue
                for dst in sorted(out_colors):
                    if dst == src:
                        continue
                    ok = True
                    actually_changed = False
                    for inp, out in train_pairs:
                        inp = np.asarray(inp); out = np.asarray(out)
                        if inp.shape != out.shape:
                            ok = False; break
                        pred = self._apply(inp, src, neigh, dst)
                        if not np.array_equal(pred, out):
                            ok = False; break
                        if not np.array_equal(pred, inp):
                            actually_changed = True
                    if ok and actually_changed:
                        self.src, self.neigh, self.dst = src, neigh, dst
                        self.fitted = True
                        return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.src, self.neigh, self.dst)

    def signature(self) -> tuple:
        return ("NeighborAwareRecolor", self.src, self.neigh, self.dst)


# ---------------------------------------------------------------------------
# RecolorEnclosedByColor — cells of bg color topologically enclosed by
# color BORDER get recolored to color FILL.
# ---------------------------------------------------------------------------

@dataclass
class RecolorEnclosedByColor:
    """Bg cells that are NOT connected to the grid boundary (via bg-cell
    4-connectivity) get recolored to FILL. Optionally restricted to cells
    enclosed by a specific BORDER color (here: anything non-bg)."""
    fill: int = 1
    fitted: bool = False

    def _apply(self, grid: Grid, fill: int) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        rows, cols = grid.shape
        if rows < 2 or cols < 2:
            return grid.copy()
        # Flood-fill bg from all boundary bg cells to find reachable bg
        reachable = np.zeros_like(grid, dtype=bool)
        stack = []
        for r in range(rows):
            for c in (0, cols-1):
                if grid[r, c] == bg and not reachable[r, c]:
                    stack.append((r, c)); reachable[r, c] = True
        for c in range(cols):
            for r in (0, rows-1):
                if grid[r, c] == bg and not reachable[r, c]:
                    stack.append((r, c)); reachable[r, c] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == bg and not reachable[nr, nc]:
                        reachable[nr, nc] = True
                        stack.append((nr, nc))
        enclosed = (grid == bg) & (~reachable)
        out = grid.copy()
        out[enclosed] = fill
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for fill in range(10):
            ok = True
            changed = False
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False; break
                pred = self._apply(inp, fill)
                if not np.array_equal(pred, out):
                    ok = False; break
                if not np.array_equal(pred, inp):
                    changed = True
            if ok and changed:
                self.fill = fill
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.fill)

    def signature(self) -> tuple:
        return ("RecolorEnclosedByColor", self.fill)


# ---------------------------------------------------------------------------
# ConnectedComponentColorCount — recolor each object based on its component size.
# ---------------------------------------------------------------------------

@dataclass
class RecolorObjectBySizeRank:
    """For each non-bg object, repaint it based on its area rank:
    rank-0 (largest) → color_at_rank_0, rank-1 → color_at_rank_1, etc.
    Colors inferred from training pairs."""
    colors: tuple[int, ...] = (0, 0, 0)
    fitted: bool = False

    def _apply(self, grid: Grid, colors: tuple[int, ...]) -> Grid:
        grid = np.asarray(grid)
        out = grid.copy()
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        for i, o in enumerate(objs):
            if i < len(colors):
                out[o["mask"]] = colors[i]
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        # Restrict colors to those in train outputs.
        out_colors = set()
        for _, o in train_pairs:
            out_colors.update(np.asarray(o).flatten().tolist())
        out_colors = sorted(out_colors)
        for c0 in out_colors:
            for c1 in out_colors:
                for c2 in out_colors:
                    cfg = (c0, c1, c2)
                    ok = True
                    changed = False
                    for inp, out in train_pairs:
                        inp = np.asarray(inp); out = np.asarray(out)
                        if inp.shape != out.shape:
                            ok = False; break
                        pred = self._apply(inp, cfg)
                        if not np.array_equal(pred, out):
                            ok = False; break
                        if not np.array_equal(pred, inp):
                            changed = True
                    if ok and changed:
                        self.colors = cfg
                        self.fitted = True
                        return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.colors)

    def signature(self) -> tuple:
        return ("RecolorObjectBySizeRank", self.colors)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

ALL_WAVE9_RULES = [
    NeighborAwareRecolor,
    RecolorEnclosedByColor,
    RecolorObjectBySizeRank,
]
