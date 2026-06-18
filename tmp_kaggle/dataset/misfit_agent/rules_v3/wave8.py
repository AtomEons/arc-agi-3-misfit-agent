"""Wave 8 — extended property-bound rules.

Continues Wave 7's property-bound contract (rule type locked at fit time,
parameter values resolved at predict time from input properties).

Target categories from 30-task unsolved sample:
  30% same_shape_recolor_global  → SwapColorByPredicate, RecolorByContainment
  27% pattern_completion         → MirrorAcrossDominantAxis, CompleteOuterFrame
  23% shape_reduce_extract       → CropToObjectByColorRank, CropToConvexHull
  10% same_shape_position_change → TranslateByObjectAttribute

Tier-1 strict throughout.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..perceptor import _background_color
from .wave4 import _objects_of, _keep_subset


Grid = np.ndarray


def _color_count_rank(grid: Grid, rank: int) -> Optional[int]:
    """Return the color at given count-rank (rank 0 = majority non-bg, etc.)."""
    grid = np.asarray(grid)
    bg = _background_color(grid)
    cnt = Counter(grid.flatten().tolist())
    cnt.pop(bg, None)
    items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
    if rank >= len(items):
        return None
    return items[rank][0]


# ---------------------------------------------------------------------------
# CropToObjectByColorRank — bbox of object whose color is rank-N by count.
# ---------------------------------------------------------------------------

@dataclass
class CropToObjectByColorRank:
    """Crop to bbox of the FIRST object whose color is rank-N by total
    cell count (across the whole grid)."""
    color_rank: int = 0
    fitted: bool = False

    def _apply(self, grid: Grid, color_rank: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        target_color = _color_count_rank(grid, color_rank)
        if target_color is None:
            return None
        objs = [o for o in _objects_of(grid) if o["color"] == target_color]
        if not objs:
            return None
        objs.sort(key=lambda o: (-o["area"], o["bbox"]))
        r0, c0, r1, c1 = objs[0]["bbox"]
        return grid[r0:r1+1, c0:c1+1].copy()

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for color_rank in range(3):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                pred = self._apply(inp, color_rank)
                if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
                    ok = False; break
            if ok:
                self.color_rank = color_rank
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.color_rank)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("CropToObjectByColorRank", self.color_rank)


# ---------------------------------------------------------------------------
# DeleteAllExceptRankN — keep only the rank-N (by area) object.
# ---------------------------------------------------------------------------

@dataclass
class DeleteAllExceptRankN:
    """Delete all non-bg cells except those belonging to the rank-N-by-area object."""
    keep_rank: int = 0
    fitted: bool = False

    def _apply(self, grid: Grid, keep_rank: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        if keep_rank >= len(objs):
            return None
        out = np.full_like(grid, bg)
        out[objs[keep_rank]["mask"]] = grid[objs[keep_rank]["mask"]]
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for keep_rank in range(3):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False; break
                pred = self._apply(inp, keep_rank)
                if pred is None or not np.array_equal(pred, out):
                    ok = False; break
            if ok:
                self.keep_rank = keep_rank
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.keep_rank)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("DeleteAllExceptRankN", self.keep_rank)


# ---------------------------------------------------------------------------
# RecolorAllObjectsToColorOfRank — repaint every object the color of rank-N object.
# ---------------------------------------------------------------------------

@dataclass
class RecolorAllObjectsToColorOfRank:
    """For every non-bg object, repaint it the color of the rank-N (by area) object."""
    donor_rank: int = 0
    fitted: bool = False

    def _apply(self, grid: Grid, donor_rank: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        if donor_rank >= len(objs):
            return None
        donor_color = objs[donor_rank]["color"]
        out = grid.copy()
        for o in objs:
            out[o["mask"]] = donor_color
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for donor_rank in range(3):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False; break
                pred = self._apply(inp, donor_rank)
                if pred is None or not np.array_equal(pred, out):
                    ok = False; break
            if ok:
                self.donor_rank = donor_rank
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.donor_rank)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("RecolorAllObjectsToColorOfRank", self.donor_rank)


# ---------------------------------------------------------------------------
# MirrorAcrossDominantAxis — detect axis where input is nearly symmetric,
# complete the missing cells via reflection.
# ---------------------------------------------------------------------------

@dataclass
class MirrorAcrossDominantAxis:
    """Detect whether output equals input with the missing-cells side
    of a horizontal/vertical/diagonal axis filled by reflection."""
    axis: str = "h"  # h, v
    fitted: bool = False

    def _apply(self, grid: Grid, axis: str) -> Grid:
        grid = np.asarray(grid).copy()
        bg = _background_color(grid)
        if axis == "h":
            rotated = grid[:, ::-1]
        elif axis == "v":
            rotated = grid[::-1, :]
        else:
            return grid
        out = grid.copy()
        mask_bg = (out == bg) & (rotated != bg)
        out[mask_bg] = rotated[mask_bg]
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for axis in ("h", "v"):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False; break
                if not np.array_equal(self._apply(inp, axis), out):
                    ok = False; break
            if ok:
                self.axis = axis
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.axis)

    def signature(self) -> tuple:
        return ("MirrorAcrossDominantAxis", self.axis)


# ---------------------------------------------------------------------------
# CompleteFrameOf — fill in the frame of an object whose color is rank-N.
# ---------------------------------------------------------------------------

@dataclass
class CompleteFrameOf:
    """For the rank-N-area object, ensure its full bbox border is filled
    with the object's color."""
    rank: int = 0
    fitted: bool = False

    def _apply(self, grid: Grid, rank: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        if rank >= len(objs):
            return None
        o = objs[rank]
        r0, c0, r1, c1 = o["bbox"]
        color = o["color"]
        out = grid.copy()
        out[r0:r1+1, c0] = color
        out[r0:r1+1, c1] = color
        out[r0, c0:c1+1] = color
        out[r1, c0:c1+1] = color
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for rank in range(3):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False; break
                pred = self._apply(inp, rank)
                if pred is None or not np.array_equal(pred, out):
                    ok = False; break
            if ok:
                self.rank = rank
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.rank)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("CompleteFrameOf", self.rank)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

ALL_WAVE8_RULES = [
    CropToObjectByColorRank,
    DeleteAllExceptRankN,
    RecolorAllObjectsToColorOfRank,
    MirrorAcrossDominantAxis,
    CompleteFrameOf,
]
