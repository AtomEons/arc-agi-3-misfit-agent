"""Wave 7 — property-bound rules: bind parameters at PREDICT time, not fit time.

Foundational diagnostic on the 1000 ARC-AGI-2 training tasks revealed that
980/1000 tasks have ZERO rules in the existing 52-rule grammar that fit
train pairs. The bottleneck is NOT vocabulary size — it is that prior rule
templates lock a global parameter (specific color, specific predicate) at
fit time, but real ARC tasks frequently require per-test-input parameter
binding (e.g., "recolor the MINORITY color" — whose identity differs per pair).

Wave 7 introduces rules whose .fit() validates the RELATION across train
pairs (does the rule's relation hold uniformly?) but whose .predict()
LOOKS UP the parameter at runtime from the test input's properties.

Tier-1 strict: no LLM, no pretrained weights, no learned values at eval.
Binding rules are deterministic functions of input properties.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..perceptor import _background_color
from .wave4 import _objects_of, _keep_subset


Grid = np.ndarray


def _nonbg_color_counts(grid: Grid) -> list[tuple[int, int]]:
    """Return [(color, count), ...] sorted by count desc, bg excluded."""
    grid = np.asarray(grid)
    bg = _background_color(grid)
    cnt = Counter(grid.flatten().tolist())
    cnt.pop(bg, None)
    return sorted(cnt.items(), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# RecolorByCountRank — recolor the rank-N-by-count non-bg color → target rule.
# ---------------------------------------------------------------------------

@dataclass
class RecolorByCountRank:
    """At predict time, find the rank-N non-bg color by pixel count in the
    input, and recolor those cells to the rank-M color (also looked up
    in the input). N and M are the locked rule params; the COLORS are
    per-input.
    """
    src_rank: int = 0   # 0 = majority, 1 = next, etc.
    dst_rank: int = 1
    use_bg_as_dst: bool = False
    fitted: bool = False

    def _resolve_colors(self, grid: Grid) -> Optional[tuple[int, int]]:
        counts = _nonbg_color_counts(grid)
        if self.use_bg_as_dst:
            if self.src_rank >= len(counts):
                return None
            src = counts[self.src_rank][0]
            dst = _background_color(grid)
            return (src, dst)
        if self.src_rank >= len(counts) or self.dst_rank >= len(counts):
            return None
        return (counts[self.src_rank][0], counts[self.dst_rank][0])

    def _apply(self, grid: Grid, src_rank: int, dst_rank: int, use_bg: bool) -> Optional[Grid]:
        self.src_rank, self.dst_rank, self.use_bg_as_dst = src_rank, dst_rank, use_bg
        colors = self._resolve_colors(grid)
        if colors is None:
            return None
        src, dst = colors
        if src == dst:
            return None
        out = np.asarray(grid).copy()
        out[out == src] = dst
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        # Try: rank pairs and bg-as-dst variants
        configs = []
        for src_rank in range(3):
            configs.append((src_rank, -1, True))
            for dst_rank in range(3):
                if src_rank != dst_rank:
                    configs.append((src_rank, dst_rank, False))
        for cfg in configs:
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False; break
                pred = self._apply(inp, *cfg)
                if pred is None or not np.array_equal(pred, out):
                    ok = False; break
            if ok:
                self.src_rank, self.dst_rank, self.use_bg_as_dst = cfg
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.src_rank, self.dst_rank, self.use_bg_as_dst)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("RecolorByCountRank", self.src_rank, self.dst_rank, self.use_bg_as_dst)


# ---------------------------------------------------------------------------
# SwapTwoNonBgColors — at predict, find the two most-common non-bg colors,
# and swap them. Binds at predict time.
# ---------------------------------------------------------------------------

@dataclass
class SwapTwoNonBgColors:
    """Find the two top-count non-bg colors at predict time and swap them."""
    fitted: bool = False

    def _apply(self, grid: Grid) -> Optional[Grid]:
        counts = _nonbg_color_counts(grid)
        if len(counts) < 2:
            return None
        a, b = counts[0][0], counts[1][0]
        out = np.asarray(grid).copy()
        ma = out == a
        mb = out == b
        out[ma] = b
        out[mb] = a
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            pred = self._apply(inp)
            if pred is None or not np.array_equal(pred, out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("SwapTwoNonBgColors",)


# ---------------------------------------------------------------------------
# RecolorByAreaRank — recolor cells of the rank-N-area object's color.
# Differs from RecolorByCountRank in that it picks per OBJECT, not per cell.
# ---------------------------------------------------------------------------

@dataclass
class RecolorByAreaRank:
    """For the rank-N (by area) object, recolor its mask to bg or to another
    color (rank-M object's color). Rule type locked at fit; values per input.
    """
    src_rank: int = 0  # 0 = largest area
    dst_rank: int = -1  # -1 = bg
    fitted: bool = False

    def _apply(self, grid: Grid, src_rank: int, dst_rank: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        if not objs or src_rank >= len(objs):
            return None
        if dst_rank == -1:
            dst = _background_color(grid)
        else:
            if dst_rank >= len(objs):
                return None
            dst = objs[dst_rank]["color"]
        out = grid.copy()
        out[objs[src_rank]["mask"]] = dst
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for src_rank in range(3):
            for dst_rank in [-1, 0, 1, 2]:
                if dst_rank == src_rank:
                    continue
                ok = True
                for inp, out in train_pairs:
                    inp = np.asarray(inp); out = np.asarray(out)
                    if inp.shape != out.shape:
                        ok = False; break
                    pred = self._apply(inp, src_rank, dst_rank)
                    if pred is None or not np.array_equal(pred, out):
                        ok = False; break
                if ok:
                    self.src_rank, self.dst_rank = src_rank, dst_rank
                    self.fitted = True
                    return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.src_rank, self.dst_rank)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("RecolorByAreaRank", self.src_rank, self.dst_rank)


# ---------------------------------------------------------------------------
# CropToObjectByAreaRank — extract bbox of rank-N-area object.
# ---------------------------------------------------------------------------

@dataclass
class CropToObjectByAreaRank:
    """Crop to the bounding box of the rank-N-area object."""
    rank: int = 0
    fitted: bool = False

    def _apply(self, grid: Grid, rank: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        if not objs or rank >= len(objs):
            return None
        r0, c0, r1, c1 = objs[rank]["bbox"]
        return grid[r0:r1+1, c0:c1+1].copy()

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for rank in range(3):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                pred = self._apply(inp, rank)
                if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
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
        return ("CropToObjectByAreaRank", self.rank)


# ---------------------------------------------------------------------------
# OverlayByObjectRank — paint smaller objects on top of larger ones.
# ---------------------------------------------------------------------------

@dataclass
class PaintObjectByRankWithColorOfRank:
    """Take the rank-S object, paint it the color of the rank-T object.
    Both ranks locked at fit; colors looked up at predict.
    """
    src_rank: int = 0
    color_donor_rank: int = 1
    fitted: bool = False

    def _apply(self, grid: Grid, sr: int, cr: int) -> Optional[Grid]:
        grid = np.asarray(grid)
        objs = sorted(_objects_of(grid), key=lambda o: (-o["area"], o["color"]))
        if sr >= len(objs) or cr >= len(objs):
            return None
        out = grid.copy()
        out[objs[sr]["mask"]] = objs[cr]["color"]
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for sr in range(3):
            for cr in range(3):
                if sr == cr:
                    continue
                ok = True
                for inp, out in train_pairs:
                    inp = np.asarray(inp); out = np.asarray(out)
                    if inp.shape != out.shape:
                        ok = False; break
                    pred = self._apply(inp, sr, cr)
                    if pred is None or not np.array_equal(pred, out):
                        ok = False; break
                if ok:
                    self.src_rank, self.color_donor_rank = sr, cr
                    self.fitted = True
                    return True
        return False

    def predict(self, grid: Grid) -> Grid:
        pred = self._apply(grid, self.src_rank, self.color_donor_rank)
        return pred if pred is not None else np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("PaintObjectByRankWithColorOfRank", self.src_rank, self.color_donor_rank)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

ALL_WAVE7_RULES = [
    RecolorByCountRank,
    SwapTwoNonBgColors,
    RecolorByAreaRank,
    CropToObjectByAreaRank,
    PaintObjectByRankWithColorOfRank,
]
