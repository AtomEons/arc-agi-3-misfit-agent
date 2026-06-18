"""Wave 15 — regional pattern completion + multi-region grids.

Many ARC-AGI-2 eval tasks contain MULTIPLE pattern regions separated by
frames or background. Each region has its own periodicity and noise. The
output is the SAME grid with each region repaired against its own period.

Wave 15 primitives:

  * RegionalPatternComplete — detect rectangular regions (by frame color
    or bg separation), apply periodic-orbit-fill per region, paste back.
  * RowPeriodicFix — detect per-row periodicity; fix outlier cells per row.
  * ColPeriodicFix — per-column periodicity.
  * RegionalSymmetryFill — per region, apply symmetry-orbit fill.
  * SubgridReplaceByOutlier — for grids that look like "tile arrangements"
    where most tiles agree and a few are outliers; replace outliers with
    the consensus tile.

All Tier-1 strict: deterministic enumeration, no learned params at eval.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..perceptor import _background_color
from .wave11 import _orbit_majority_periodic, _best_period


Grid = np.ndarray


# ---------------------------------------------------------------------------
# Region detection.
# A "region" is a maximal rectangular block of cells inside the grid that
# is bounded by either (a) the grid edge or (b) a single-color "frame"
# (all cells of one color forming a closed border).
# ---------------------------------------------------------------------------


def _detect_horizontal_strips(grid: Grid, frame_color: int) -> list[tuple[int, int]]:
    """Return list of (row_start, row_end) inclusive ranges of strips
    between frame-color rows. A row is "frame" if it's all frame_color.
    """
    rows, cols = grid.shape
    frame_rows = []
    for r in range(rows):
        if all(grid[r, c] == frame_color for c in range(cols)):
            frame_rows.append(r)
    if not frame_rows:
        return [(0, rows - 1)]
    strips = []
    prev = -1
    for fr in frame_rows:
        if fr > prev + 1:
            strips.append((prev + 1, fr - 1))
        prev = fr
    if prev < rows - 1:
        strips.append((prev + 1, rows - 1))
    return strips


def _detect_vertical_strips(grid: Grid, frame_color: int) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    frame_cols = []
    for c in range(cols):
        if all(grid[r, c] == frame_color for r in range(rows)):
            frame_cols.append(c)
    if not frame_cols:
        return [(0, cols - 1)]
    strips = []
    prev = -1
    for fc in frame_cols:
        if fc > prev + 1:
            strips.append((prev + 1, fc - 1))
        prev = fc
    if prev < cols - 1:
        strips.append((prev + 1, cols - 1))
    return strips


def _detect_grid_frame_colors(grid: Grid) -> list[int]:
    """Colors that COULD be a frame color: appear in border row/col."""
    rows, cols = grid.shape
    candidates = set()
    if rows > 0:
        candidates.update(grid[0].tolist())
        candidates.update(grid[-1].tolist())
    if cols > 0:
        candidates.update(grid[:, 0].tolist())
        candidates.update(grid[:, -1].tolist())
    return sorted(candidates)


def _region_periodic_repair(grid: Grid) -> Grid:
    """Try frame-color-defined strips; per strip, periodic fill."""
    rows, cols = grid.shape
    bg = _background_color(grid)
    out = grid.copy()
    candidates = _detect_grid_frame_colors(grid)
    # Try each candidate frame color; prefer the one that produces strips
    # with non-trivial periodic structure.
    for fc in candidates:
        for orientation in ("horizontal", "vertical"):
            if orientation == "horizontal":
                strips = _detect_horizontal_strips(grid, fc)
                slices = [(slice(r0, r1+1), slice(0, cols)) for r0, r1 in strips]
            else:
                strips = _detect_vertical_strips(grid, fc)
                slices = [(slice(0, rows), slice(c0, c1+1)) for c0, c1 in strips]
            if len(strips) < 2:
                continue
            test = grid.copy()
            for sr, sc in slices:
                region = grid[sr, sc]
                if region.size <= 1:
                    continue
                period = _best_period(region)
                if period is None:
                    continue
                ph, pw = period
                repaired = _orbit_majority_periodic(region, ph, pw)
                test[sr, sc] = repaired
            if not np.array_equal(test, grid):
                return test
    return out


@dataclass
class RegionalPatternComplete:
    """Detect frame-color separation, apply periodic repair per region."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        any_change = False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            repaired = _region_periodic_repair(inp)
            if not np.array_equal(repaired, out):
                return False
            if not np.array_equal(inp, out):
                any_change = True
        if not any_change:
            return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _region_periodic_repair(np.asarray(grid))

    def signature(self) -> tuple:
        return ("RegionalPatternComplete",)


# ---------------------------------------------------------------------------
# RowPeriodicFix / ColPeriodicFix — per-row / per-column periodicity repair.
# Each row independently detects its own period and fills.
# ---------------------------------------------------------------------------


def _row_periodic_fix(grid: Grid) -> Grid:
    rows, cols = grid.shape
    out = grid.copy()
    for r in range(rows):
        row = grid[r]
        if cols < 2:
            continue
        # find best period for this row
        best_p = None
        best_score = -1
        for p in range(1, cols // 2 + 1):
            # candidate fill: bucket by index % p, take majority
            filled = np.zeros_like(row)
            buckets = {}
            for c in range(cols):
                buckets.setdefault(c % p, []).append(int(row[c]))
            top = {}
            for k, vals in buckets.items():
                cnt = Counter(vals)
                top[k] = cnt.most_common(1)[0][0]
            for c in range(cols):
                filled[c] = top[c % p]
            agree = int((filled == row).sum())
            if agree > best_score:
                best_score = agree
                best_p = p
                best_filled = filled
        if best_p is not None and best_score < cols:
            out[r] = best_filled
    return out


def _col_periodic_fix(grid: Grid) -> Grid:
    return _row_periodic_fix(grid.T).T


@dataclass
class RowPeriodicFix:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        any_change = False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            applied = _row_periodic_fix(inp)
            if not np.array_equal(applied, out):
                return False
            if not np.array_equal(inp, out):
                any_change = True
        if not any_change:
            return False
        self.fitted = True
        return True

    def predict(self, grid):
        return _row_periodic_fix(np.asarray(grid))

    def signature(self):
        return ("RowPeriodicFix",)


@dataclass
class ColPeriodicFix:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        any_change = False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            applied = _col_periodic_fix(inp)
            if not np.array_equal(applied, out):
                return False
            if not np.array_equal(inp, out):
                any_change = True
        if not any_change:
            return False
        self.fitted = True
        return True

    def predict(self, grid):
        return _col_periodic_fix(np.asarray(grid))

    def signature(self):
        return ("ColPeriodicFix",)


# ---------------------------------------------------------------------------
# SubgridReplaceByOutlier — for grids that are arrangements of equally-sized
# tiles where most agree and a few are outliers; replace outliers with the
# canonical (most-frequent) tile.
# ---------------------------------------------------------------------------


def _subgrid_consensus_fix(grid: Grid) -> Optional[Grid]:
    """Try (nr, nc) tile counts; pick the most-frequent (mode) tile;
    paint all positions with it."""
    rows, cols = grid.shape
    best = None
    best_agree = 0
    for nr in range(2, 7):
        if rows % nr != 0:
            continue
        for nc in range(2, 7):
            if cols % nc != 0:
                continue
            th, tw = rows // nr, cols // nc
            tiles = []
            for r in range(nr):
                for c in range(nc):
                    t = grid[r*th:(r+1)*th, c*tw:(c+1)*tw]
                    tiles.append(t.tobytes())
            cnt = Counter(tiles)
            mode_bytes, mode_n = cnt.most_common(1)[0]
            if mode_n < 2:
                continue
            # consensus has clear winner if mode_n > others
            mode_tile = np.frombuffer(mode_bytes, dtype=grid.dtype).reshape((th, tw))
            out = grid.copy()
            for r in range(nr):
                for c in range(nc):
                    out[r*th:(r+1)*th, c*tw:(c+1)*tw] = mode_tile
            agree = int((out == grid).sum())
            if agree > best_agree:
                best_agree = agree
                best = out
    return best


@dataclass
class SubgridReplaceByOutlier:
    fitted: bool = False

    def fit(self, train_pairs):
        if not train_pairs:
            return False
        any_change = False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            res = _subgrid_consensus_fix(inp)
            if res is None or not np.array_equal(res, out):
                return False
            if not np.array_equal(inp, out):
                any_change = True
        if not any_change:
            return False
        self.fitted = True
        return True

    def predict(self, grid):
        grid = np.asarray(grid)
        res = _subgrid_consensus_fix(grid)
        return res if res is not None else grid.copy()

    def signature(self):
        return ("SubgridReplaceByOutlier",)


ALL_WAVE15_RULES = [
    RegionalPatternComplete,
    RowPeriodicFix,
    ColPeriodicFix,
    SubgridReplaceByOutlier,
]
