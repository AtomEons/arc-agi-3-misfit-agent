"""Wave 5 — per-object rule dispatch and parameter inference.

ForEachObject wraps any inner Wave-4 rule and applies it PER OBJECT
rather than globally. KeepByInferredPredicate auto-infers the keep
criterion from train pairs (which object survived: largest? smallest?
matching a particular color? touching the edge?).

Tier-1 strict: no LLM, no pretrained weights. The "inference" is
deterministic — try each predicate, accept the one that fits all
train pairs, reject otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..perceptor import _background_color
from .wave4 import (
    _objects_of, _keep_subset,
    GravityUp, GravityDown, GravityLeft, GravityRight,
    SymmetrizeH, SymmetrizeV,
)


Grid = np.ndarray


# ---------------------------------------------------------------------------
# Predicate-inferred keep / delete
# ---------------------------------------------------------------------------

PREDICATES = {
    "largest_area": lambda objs: max(objs, key=lambda o: o["area"]),
    "smallest_area": lambda objs: min(objs, key=lambda o: o["area"]),
    "max_color": lambda objs: max(objs, key=lambda o: o["color"]),
    "min_color": lambda objs: min(objs, key=lambda o: o["color"]),
    "touches_edge": None,   # handled specially: filter, not select
    "interior": None,       # not edge-touching
    "unique_color": None,   # only object of its color
    "duplicated_color": None,
}


def _touches_edge(grid_shape, mask: np.ndarray) -> bool:
    rows, cols = grid_shape
    return bool(mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any())


@dataclass
class KeepByPredicate:
    """Keep objects matching one of the inferred predicates."""
    predicate: str = "largest_area"
    fitted: bool = False

    def _apply(self, grid: Grid, predicate: str) -> Grid:
        grid = np.asarray(grid)
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        if predicate in ("largest_area", "smallest_area", "max_color", "min_color"):
            chosen = PREDICATES[predicate](objs)
            return _keep_subset(grid, chosen["mask"])
        if predicate == "touches_edge":
            mask = np.zeros_like(grid, dtype=bool)
            for o in objs:
                if _touches_edge(grid.shape, o["mask"]):
                    mask |= o["mask"]
            return _keep_subset(grid, mask)
        if predicate == "interior":
            mask = np.zeros_like(grid, dtype=bool)
            for o in objs:
                if not _touches_edge(grid.shape, o["mask"]):
                    mask |= o["mask"]
            return _keep_subset(grid, mask)
        if predicate == "unique_color":
            from collections import Counter
            counts = Counter(o["color"] for o in objs)
            mask = np.zeros_like(grid, dtype=bool)
            for o in objs:
                if counts[o["color"]] == 1:
                    mask |= o["mask"]
            return _keep_subset(grid, mask)
        if predicate == "duplicated_color":
            from collections import Counter
            counts = Counter(o["color"] for o in objs)
            mask = np.zeros_like(grid, dtype=bool)
            for o in objs:
                if counts[o["color"]] > 1:
                    mask |= o["mask"]
            return _keep_subset(grid, mask)
        return grid.copy()

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for predicate in PREDICATES.keys():
            self.predicate = predicate
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(self._apply(inp, predicate), out):
                    ok = False
                    break
            if ok:
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.predicate)

    def signature(self) -> tuple:
        return ("KeepByPredicate", self.predicate)


@dataclass
class DeleteByPredicate:
    """Delete objects matching one of the inferred predicates."""
    predicate: str = "largest_area"
    fitted: bool = False

    def _apply(self, grid: Grid, predicate: str) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        out = grid.copy()
        if predicate in ("largest_area", "smallest_area", "max_color", "min_color"):
            chosen = PREDICATES[predicate](objs)
            out[chosen["mask"]] = bg
            return out
        if predicate == "touches_edge":
            for o in objs:
                if _touches_edge(grid.shape, o["mask"]):
                    out[o["mask"]] = bg
            return out
        if predicate == "interior":
            for o in objs:
                if not _touches_edge(grid.shape, o["mask"]):
                    out[o["mask"]] = bg
            return out
        if predicate == "unique_color":
            from collections import Counter
            counts = Counter(o["color"] for o in objs)
            for o in objs:
                if counts[o["color"]] == 1:
                    out[o["mask"]] = bg
            return out
        if predicate == "duplicated_color":
            from collections import Counter
            counts = Counter(o["color"] for o in objs)
            for o in objs:
                if counts[o["color"]] > 1:
                    out[o["mask"]] = bg
            return out
        return grid.copy()

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for predicate in PREDICATES.keys():
            self.predicate = predicate
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(self._apply(inp, predicate), out):
                    ok = False
                    break
            if ok:
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.predicate)

    def signature(self) -> tuple:
        return ("DeleteByPredicate", self.predicate)


# ---------------------------------------------------------------------------
# Per-object color-change (Recolor PER OBJECT)
# ---------------------------------------------------------------------------

@dataclass
class RecolorByPredicate:
    """For each object, if it matches the predicate, recolor to target.
    Both predicate and target color inferred from train pairs.
    """
    predicate: str = "largest_area"
    target_color: int = 1
    fitted: bool = False

    def _apply(self, grid: Grid, predicate: str, target: int) -> Grid:
        grid = np.asarray(grid)
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        out = grid.copy()
        # Determine which objects pass the predicate
        chosen_masks = []
        if predicate in ("largest_area", "smallest_area", "max_color", "min_color"):
            chosen_masks = [PREDICATES[predicate](objs)["mask"]]
        elif predicate == "touches_edge":
            chosen_masks = [o["mask"] for o in objs
                            if _touches_edge(grid.shape, o["mask"])]
        elif predicate == "interior":
            chosen_masks = [o["mask"] for o in objs
                            if not _touches_edge(grid.shape, o["mask"])]
        elif predicate == "unique_color":
            from collections import Counter
            counts = Counter(o["color"] for o in objs)
            chosen_masks = [o["mask"] for o in objs if counts[o["color"]] == 1]
        elif predicate == "duplicated_color":
            from collections import Counter
            counts = Counter(o["color"] for o in objs)
            chosen_masks = [o["mask"] for o in objs if counts[o["color"]] > 1]
        for m in chosen_masks:
            out[m] = target
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for predicate in PREDICATES.keys():
            for target in range(10):
                ok = True
                for inp, out in train_pairs:
                    inp = np.asarray(inp); out = np.asarray(out)
                    if inp.shape != out.shape:
                        ok = False
                        break
                    if not np.array_equal(self._apply(inp, predicate, target), out):
                        ok = False
                        break
                if ok:
                    # reject trivial (no change to grid)
                    sample = np.asarray(train_pairs[0][0])
                    sample_out = self._apply(sample, predicate, target)
                    if np.array_equal(sample, sample_out):
                        continue
                    self.predicate = predicate
                    self.target_color = target
                    self.fitted = True
                    return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.predicate, self.target_color)

    def signature(self) -> tuple:
        return ("RecolorByPredicate", self.predicate, self.target_color)


# ---------------------------------------------------------------------------
# ForEachObject wrapper — apply a sub-transformation per object's bbox
# ---------------------------------------------------------------------------

@dataclass
class ForEachObjectGravity:
    """For each non-bg object, apply gravity within its own bounding box.
    Direction inferred from train pairs.
    """
    direction: str = "up"
    fitted: bool = False

    def _apply(self, grid: Grid, direction: str) -> Grid:
        from .wave4 import _gravity
        grid = np.asarray(grid)
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        out = grid.copy()
        for o in objs:
            r0, c0, r1, c1 = o["bbox"]
            sub = out[r0:r1+1, c0:c1+1].copy()
            sub_out = _gravity(sub, direction)
            out[r0:r1+1, c0:c1+1] = sub_out
        return out

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for direction in ("up", "down", "left", "right"):
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(self._apply(inp, direction), out):
                    ok = False
                    break
            if ok:
                self.direction = direction
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply(grid, self.direction)

    def signature(self) -> tuple:
        return ("ForEachObjectGravity", self.direction)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

ALL_WAVE5_RULES = [
    KeepByPredicate, DeleteByPredicate,
    RecolorByPredicate,
    ForEachObjectGravity,
]
