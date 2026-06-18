"""Wave 11 — pattern-completion families.

Failure-mode fingerprint shows 27% of unsolved ARC-AGI-2 tasks are
pattern_completion: input has a partial / noisy pattern, output completes it.

Three Tier-1 strict primitives, deterministic enumeration, no learned params:

  * PatternCompleteByPeriodicity — detect (ph, pw) modular period;
    fill each modular orbit with the majority non-bg color of the orbit.
  * PatternCompleteBySymmetry — detect H / V / D / AD axis (or combos);
    fill each symmetry-orbit with the majority non-bg color.
  * PatternCompleteByTile — detect (n_rows, n_cols) tile structure;
    pick the canonical (most-frequent-across-tiles) tile and replicate.

Each rule fits if the relation holds on ALL train pairs (output == completion
of input under the detected structure). Parameters are bound at predict time
from the test input's own structure — property-bound contract (Wave 7
doctrine).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..perceptor import _background_color


Grid = np.ndarray


# ---------------------------------------------------------------------------
# Periodicity-based pattern completion.
#
# Strategy:
#   1. Enumerate candidate periods (ph, pw) where ph divides rows OR rows
#      mod ph yields a clean repeat AND pw similar for cols. Allow ph in
#      [1, rows//2], pw in [1, cols//2].
#   2. For each candidate (ph, pw), bucket cells by (r % ph, c % pw).
#      Compute the majority NON-BG color per bucket (ties → most common
#      overall, then bg).
#   3. Build the "completed" grid by filling each cell with its bucket's
#      majority color.
#   4. Pick the (ph, pw) that minimizes disagreement with the input on
#      non-bg cells (so we're "fixing the noise", not reinventing it).
# ---------------------------------------------------------------------------


def _orbit_majority_periodic(grid: Grid, ph: int, pw: int) -> Grid:
    """Bucket cells by (r % ph, c % pw) and fill each bucket with the
    majority NON-BG color of that bucket. Bg fallback if a bucket is
    all-bg or empty.
    """
    rows, cols = grid.shape
    bg = _background_color(grid)
    out = np.full_like(grid, bg)
    # Build buckets
    buckets: dict[tuple[int, int], list[int]] = {}
    for r in range(rows):
        for c in range(cols):
            key = (r % ph, c % pw)
            buckets.setdefault(key, []).append(int(grid[r, c]))
    bucket_fill: dict[tuple[int, int], int] = {}
    for key, vals in buckets.items():
        non_bg = [v for v in vals if v != bg]
        if non_bg:
            c = Counter(non_bg)
            top, _ = c.most_common(1)[0]
            bucket_fill[key] = top
        else:
            bucket_fill[key] = bg
    for r in range(rows):
        for c in range(cols):
            out[r, c] = bucket_fill[(r % ph, c % pw)]
    return out


def _best_period(grid: Grid) -> Optional[tuple[int, int]]:
    """Return (ph, pw) that minimizes disagreement with grid under
    periodic-orbit fill. None if grid is too small.
    """
    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        return None
    best = None
    best_score = (-1, 0)  # (consistency_count, neg-area-penalty)
    for ph in range(1, max(2, rows // 2 + 1)):
        for pw in range(1, max(2, cols // 2 + 1)):
            try:
                completed = _orbit_majority_periodic(grid, ph, pw)
            except Exception:
                continue
            agree = int(np.sum(completed == grid))
            score = (agree, -(ph * pw))
            if score > best_score:
                best_score = score
                best = (ph, pw)
    return best


@dataclass
class PatternCompleteByPeriodicity:
    """Fit if every train pair has output == periodic-orbit-fill of input
    under a per-pair-discovered period. Predict via discovering the period
    of the test input.
    """
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        # All pairs must be same-shape input/output AND output must equal
        # the periodic-fill of input under SOME period.
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            best = _best_period(inp)
            if best is None:
                return False
            ph, pw = best
            completed = _orbit_majority_periodic(inp, ph, pw)
            if not np.array_equal(completed, out):
                # Try the next-best non-trivial period
                rows, cols = inp.shape
                found = False
                for ph2 in range(1, max(2, rows // 2 + 1)):
                    for pw2 in range(1, max(2, cols // 2 + 1)):
                        if (ph2, pw2) == best:
                            continue
                        completed2 = _orbit_majority_periodic(inp, ph2, pw2)
                        if np.array_equal(completed2, out):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        best = _best_period(grid)
        if best is None:
            return grid.copy()
        ph, pw = best
        return _orbit_majority_periodic(grid, ph, pw)

    def signature(self) -> tuple:
        return ("PatternCompleteByPeriodicity",)


# ---------------------------------------------------------------------------
# Symmetry-based pattern completion.
#
# Strategy:
#   - For each axis (H, V, diag, antidiag, all-of-them), bucket cells into
#     symmetry orbits. Fill each orbit with its majority non-bg color.
#   - Fit if there exists an axis under which output == symmetry-fill(input).
# ---------------------------------------------------------------------------


def _orbit_indices_h(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    orbits = {}
    for r in range(rows):
        for c in range(cols):
            key = (min(r, rows - 1 - r), c)
            orbits.setdefault(key, []).append((r, c))
    return orbits


def _orbit_indices_v(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    orbits = {}
    for r in range(rows):
        for c in range(cols):
            key = (r, min(c, cols - 1 - c))
            orbits.setdefault(key, []).append((r, c))
    return orbits


def _orbit_indices_d(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    if rows != cols:
        return {}
    orbits = {}
    for r in range(rows):
        for c in range(cols):
            key = tuple(sorted((r, c)))
            orbits.setdefault(key, []).append((r, c))
    return orbits


def _orbit_indices_ad(rows: int, cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    if rows != cols:
        return {}
    n = rows
    orbits = {}
    for r in range(rows):
        for c in range(cols):
            key = tuple(sorted((r, n - 1 - c)))
            orbits.setdefault(key, []).append((r, c))
    return orbits


def _orbit_indices_hv(rows: int, cols: int) -> dict:
    orbits = {}
    for r in range(rows):
        for c in range(cols):
            key = (min(r, rows - 1 - r), min(c, cols - 1 - c))
            orbits.setdefault(key, []).append((r, c))
    return orbits


def _orbit_indices_d4(rows: int, cols: int) -> dict:
    """All 4 reflections + rotations — only for square grids."""
    if rows != cols:
        return {}
    n = rows
    orbits = {}
    for r in range(n):
        for c in range(n):
            rotations = [
                (r, c),
                (c, n - 1 - r),
                (n - 1 - r, n - 1 - c),
                (n - 1 - c, r),
                (r, n - 1 - c),
                (n - 1 - r, c),
                (c, r),
                (n - 1 - c, n - 1 - r),
            ]
            key = min(rotations)
            orbits.setdefault(key, []).append((r, c))
    return orbits


_AXIS_FNS = {
    "H": _orbit_indices_h,
    "V": _orbit_indices_v,
    "D": _orbit_indices_d,
    "AD": _orbit_indices_ad,
    "HV": _orbit_indices_hv,
    "D4": _orbit_indices_d4,
}


def _orbit_majority_symmetric(grid: Grid, axis: str) -> Grid:
    rows, cols = grid.shape
    bg = _background_color(grid)
    orbits = _AXIS_FNS[axis](rows, cols)
    if not orbits:
        return grid.copy()
    out = np.full_like(grid, bg)
    for key, cells in orbits.items():
        vals = [int(grid[r, c]) for r, c in cells]
        non_bg = [v for v in vals if v != bg]
        if non_bg:
            c = Counter(non_bg)
            top, _ = c.most_common(1)[0]
        else:
            top = bg
        for r, ccol in cells:
            out[r, ccol] = top
    return out


@dataclass
class PatternCompleteBySymmetry:
    """Try each symmetry axis (H, V, D, AD, HV, D4). Fit if there is an
    axis under which output == symmetry-fill(input) on EVERY train pair.
    The axis is the same for all train pairs of a task (locked at fit).
    """
    axis: str = "H"
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates = set(_AXIS_FNS.keys())
        # Require same-shape input/output
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            local = set()
            for axis in list(candidates):
                completed = _orbit_majority_symmetric(inp, axis)
                if np.array_equal(completed, out):
                    local.add(axis)
            candidates &= local
            if not candidates:
                return False
        # Prefer simpler axes first
        for pref in ("H", "V", "D", "AD", "HV", "D4"):
            if pref in candidates:
                self.axis = pref
                break
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _orbit_majority_symmetric(np.asarray(grid), self.axis)

    def signature(self) -> tuple:
        return ("PatternCompleteBySymmetry", self.axis)


# ---------------------------------------------------------------------------
# Tile-based pattern completion.
#
# Strategy:
#   - For tile counts (nr, nc) where rows % nr == 0 and cols % nc == 0
#     (with nr, nc in 2..6), split into tiles. Pick the canonical tile as
#     the most-frequent tile, OR the cell-wise majority across tiles.
#   - Build the completed grid by replicating the canonical tile across
#     all positions.
# ---------------------------------------------------------------------------


def _tile_split(grid: Grid, nr: int, nc: int) -> Optional[list[list[Grid]]]:
    rows, cols = grid.shape
    if rows % nr != 0 or cols % nc != 0:
        return None
    th, tw = rows // nr, cols // nc
    tiles = []
    for r in range(nr):
        row = []
        for c in range(nc):
            tile = grid[r*th:(r+1)*th, c*tw:(c+1)*tw]
            row.append(tile)
        tiles.append(row)
    return tiles


def _canonical_tile(tiles: list[list[Grid]]) -> Grid:
    """Cell-wise majority across all tiles. Bg fallback when all-bg."""
    if not tiles or not tiles[0]:
        return np.zeros((1, 1), dtype=np.int32)
    th, tw = tiles[0][0].shape
    all_tiles = [tiles[r][c] for r in range(len(tiles)) for c in range(len(tiles[0]))]
    # Bg = global bg of the underlying tile set
    flat = np.concatenate([t.flatten() for t in all_tiles])
    if (flat == 0).any():
        bg = 0
    else:
        bg = int(np.bincount(flat, minlength=10).argmax())
    canon = np.full((th, tw), bg, dtype=np.int32)
    for r in range(th):
        for c in range(tw):
            vals = [int(t[r, c]) for t in all_tiles]
            non_bg = [v for v in vals if v != bg]
            if non_bg:
                cnt = Counter(non_bg)
                top, _ = cnt.most_common(1)[0]
                canon[r, c] = top
    return canon


def _tile_fill(grid: Grid, nr: int, nc: int) -> Optional[Grid]:
    tiles = _tile_split(grid, nr, nc)
    if tiles is None:
        return None
    canon = _canonical_tile(tiles)
    rows, cols = grid.shape
    out = np.zeros_like(grid)
    th, tw = canon.shape
    for r in range(nr):
        for c in range(nc):
            out[r*th:(r+1)*th, c*tw:(c+1)*tw] = canon
    return out


@dataclass
class PatternCompleteByTile:
    """Fit if there is a (nr, nc) tile count for which output ==
    tile-fill(input, nr, nc) on EVERY train pair. (nr, nc) locked across
    pairs of the task.
    """
    nr: int = 2
    nc: int = 2
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates: Optional[set] = None
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            local: set = set()
            rows, cols = inp.shape
            for nr in range(2, 7):
                if rows % nr != 0:
                    continue
                for nc in range(2, 7):
                    if cols % nc != 0:
                        continue
                    filled = _tile_fill(inp, nr, nc)
                    if filled is not None and np.array_equal(filled, out):
                        local.add((nr, nc))
            if not local:
                return False
            candidates = local if candidates is None else (candidates & local)
            if not candidates:
                return False
        if not candidates:
            return False
        # Smallest tile count (Occam)
        self.nr, self.nc = min(candidates, key=lambda nrc: nrc[0] * nrc[1])
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        filled = _tile_fill(grid, self.nr, self.nc)
        if filled is None:
            return grid.copy()
        return filled

    def signature(self) -> tuple:
        return ("PatternCompleteByTile", self.nr, self.nc)


# ---------------------------------------------------------------------------
# BgHoleFill — bg cells where the periodic / symmetric extension reveals a
# non-bg color. Apply to the cell. Used when the input is a "punctured"
# version of a pattern (bg holes), output is the unpunctured pattern.
# ---------------------------------------------------------------------------


@dataclass
class BgHoleFillByOrbit:
    """For each cell of color BG, look at its periodic-orbit majority
    (auto-detect period). If majority non-bg, recolor the cell to it.
    Non-bg cells unchanged.
    """
    fitted: bool = False

    def _apply(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        best = _best_period(grid)
        if best is None:
            return grid.copy()
        ph, pw = best
        rows, cols = grid.shape
        out = grid.copy()
        buckets: dict[tuple[int, int], list[int]] = {}
        for r in range(rows):
            for c in range(cols):
                buckets.setdefault((r % ph, c % pw), []).append(int(grid[r, c]))
        bucket_top: dict[tuple[int, int], int] = {}
        for key, vals in buckets.items():
            non_bg = [v for v in vals if v != bg]
            if non_bg:
                cnt = Counter(non_bg)
                top, _ = cnt.most_common(1)[0]
                bucket_top[key] = top
        for r in range(rows):
            for c in range(cols):
                if out[r, c] == bg:
                    key = (r % ph, c % pw)
                    if key in bucket_top:
                        out[r, c] = bucket_top[key]
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        had_change = False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            applied = self._apply(inp)
            if not np.array_equal(applied, out):
                return False
            if not np.array_equal(inp, out):
                had_change = True
        if not had_change:
            return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return self._apply(np.asarray(grid))

    def signature(self) -> tuple:
        return ("BgHoleFillByOrbit",)


# ---------------------------------------------------------------------------
# MirrorFillByAxis — for each non-bg cell, mirror it across the detected
# axis to fill that mirror position if it is currently bg. Best on tasks
# where half the pattern is given and the other half must be inferred.
# ---------------------------------------------------------------------------


def _mirror_fill_axis(grid: Grid, axis: str) -> Grid:
    grid = np.asarray(grid)
    bg = _background_color(grid)
    rows, cols = grid.shape
    out = grid.copy()
    orbits_fn = _AXIS_FNS.get(axis)
    if orbits_fn is None:
        return out
    orbits = orbits_fn(rows, cols)
    if not orbits:
        return out
    for key, cells in orbits.items():
        vals = [int(out[r, c]) for r, c in cells]
        non_bg = [v for v in vals if v != bg]
        if not non_bg:
            continue
        cnt = Counter(non_bg)
        top, _ = cnt.most_common(1)[0]
        # Only fill the bg cells in the orbit; do not overwrite non-bg.
        for r, c in cells:
            if out[r, c] == bg:
                out[r, c] = top
    return out


@dataclass
class MirrorFillByAxis:
    """For each axis (H/V/D/AD/HV/D4), fill bg cells with the orbit's
    non-bg majority. Non-bg cells unchanged. Fit if some axis works for
    all train pairs.
    """
    axis: str = "H"
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates = set(_AXIS_FNS.keys())
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            local = set()
            for axis in list(candidates):
                filled = _mirror_fill_axis(inp, axis)
                if np.array_equal(filled, out):
                    local.add(axis)
            candidates &= local
            if not candidates:
                return False
        for pref in ("H", "V", "D", "AD", "HV", "D4"):
            if pref in candidates:
                self.axis = pref
                break
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _mirror_fill_axis(np.asarray(grid), self.axis)

    def signature(self) -> tuple:
        return ("MirrorFillByAxis", self.axis)


ALL_WAVE11_RULES = [
    PatternCompleteByPeriodicity,
    PatternCompleteBySymmetry,
    PatternCompleteByTile,
    BgHoleFillByOrbit,
    MirrorFillByAxis,
]
