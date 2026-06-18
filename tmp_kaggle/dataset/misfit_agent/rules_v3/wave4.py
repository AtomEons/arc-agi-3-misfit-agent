"""Wave 4 rule families — climb the score.

Six families, all Spelke priors only:
  - ObjectKeep: KeepLargest, KeepSmallest, KeepByMaxColor, KeepByMinColor
  - ObjectDelete: DeleteLargest, DeleteSmallest
  - Symmetrize: SymmetrizeH, SymmetrizeV, SymmetrizeDiag
  - Gravity: GravityUp, GravityDown, GravityLeft, GravityRight
  - BorderAndFill: DrawBorder, FillInterior
  - FixedPoint: ApplyUntilStable

Each rule has standard interface: fit(train_pairs)->bool, predict(grid)->grid,
signature()->tuple. Tier-1 honest: no LLM, no pretrained weights, no learned
parameters. Pure observation-driven induction from train pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..perceptor import perceive_grid, _background_color


Grid = np.ndarray


# ---------------------------------------------------------------------------
# Object-relational helpers
# ---------------------------------------------------------------------------

def _objects_of(grid: Grid) -> list[dict]:
    """Return list of object dicts: {color, area, bbox, mask}."""
    grid = np.asarray(grid)
    bg = _background_color(grid)
    scene = perceive_grid(grid)
    out = []
    for obj in scene.objects:
        r0, c0, r1, c1 = obj.bbox
        mask = np.zeros_like(grid, dtype=bool)
        # rebuild precise mask via 4-conn flood
        mask[r0:r1+1, c0:c1+1] = (grid[r0:r1+1, c0:c1+1] == obj.color)
        out.append({
            "color": int(obj.color),
            "area": int(obj.area),
            "bbox": tuple(obj.bbox),
            "mask": mask,
        })
    return out


def _keep_subset(grid: Grid, keep_mask: np.ndarray) -> Grid:
    """Return grid with only keep_mask cells preserved; rest set to background."""
    grid = np.asarray(grid)
    bg = _background_color(grid)
    out = np.full_like(grid, fill_value=bg)
    out[keep_mask] = grid[keep_mask]
    return out


# ---------------------------------------------------------------------------
# ObjectKeep family
# ---------------------------------------------------------------------------

@dataclass
class KeepLargest:
    """Output = input with only the largest non-bg object preserved."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            pred = self.predict(inp)
            if not np.array_equal(pred, out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        if not objs:
            return np.asarray(grid).copy()
        largest = max(objs, key=lambda o: o["area"])
        return _keep_subset(grid, largest["mask"])

    def signature(self) -> tuple:
        return ("KeepLargest",)


@dataclass
class KeepSmallest:
    """Output = input with only the smallest non-bg object preserved."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        objs = _objects_of(grid)
        if not objs:
            return np.asarray(grid).copy()
        smallest = min(objs, key=lambda o: o["area"])
        return _keep_subset(grid, smallest["mask"])

    def signature(self) -> tuple:
        return ("KeepSmallest",)


@dataclass
class KeepByMaxColor:
    """Output = input with only the highest-numeric-color object class preserved."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        fg_colors = [c for c in range(10) if (grid == c).any() and c != bg]
        if not fg_colors:
            return grid.copy()
        target = max(fg_colors)
        keep = grid == target
        return _keep_subset(grid, keep)

    def signature(self) -> tuple:
        return ("KeepByMaxColor",)


@dataclass
class KeepByMinColor:
    """Output = input with only the lowest-numeric-fg color object class preserved."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        fg_colors = [c for c in range(10) if (grid == c).any() and c != bg]
        if not fg_colors:
            return grid.copy()
        target = min(fg_colors)
        keep = grid == target
        return _keep_subset(grid, keep)

    def signature(self) -> tuple:
        return ("KeepByMinColor",)


# ---------------------------------------------------------------------------
# ObjectDelete family
# ---------------------------------------------------------------------------

@dataclass
class DeleteLargest:
    """Output = input with the largest non-bg object removed."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        largest = max(objs, key=lambda o: o["area"])
        out = grid.copy()
        out[largest["mask"]] = bg
        return out

    def signature(self) -> tuple:
        return ("DeleteLargest",)


@dataclass
class DeleteSmallest:
    """Output = input with the smallest non-bg object removed."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        objs = _objects_of(grid)
        if not objs:
            return grid.copy()
        smallest = min(objs, key=lambda o: o["area"])
        out = grid.copy()
        out[smallest["mask"]] = bg
        return out

    def signature(self) -> tuple:
        return ("DeleteSmallest",)


# ---------------------------------------------------------------------------
# Symmetrize family — OR-completion across axes
# ---------------------------------------------------------------------------

@dataclass
class SymmetrizeH:
    """Output = input OR-completed to be horizontally symmetric (fliplr-equal)."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        flipped = np.fliplr(grid)
        out = grid.copy()
        # OR-completion: where original is bg, use flipped value
        out = np.where(out == bg, flipped, out)
        return out

    def signature(self) -> tuple:
        return ("SymmetrizeH",)


@dataclass
class SymmetrizeV:
    """Output = input OR-completed to be vertically symmetric (flipud-equal)."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        flipped = np.flipud(grid)
        out = grid.copy()
        out = np.where(out == bg, flipped, out)
        return out

    def signature(self) -> tuple:
        return ("SymmetrizeV",)


@dataclass
class SymmetrizeDiag:
    """Output = input OR-completed to be diagonally symmetric (transpose-equal).
    Requires square input.
    """
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape[0] != inp.shape[1] or inp.shape != out.shape:
                return False
            if not np.array_equal(self.predict(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        if grid.shape[0] != grid.shape[1]:
            return grid.copy()
        bg = _background_color(grid)
        transposed = grid.T
        out = grid.copy()
        out = np.where(out == bg, transposed, out)
        return out

    def signature(self) -> tuple:
        return ("SymmetrizeDiag",)


# ---------------------------------------------------------------------------
# Gravity family — non-bg cells fall toward an edge
# ---------------------------------------------------------------------------

def _gravity(grid: Grid, direction: str) -> Grid:
    """Apply gravity. Direction in {'up','down','left','right'}."""
    grid = np.asarray(grid)
    bg = _background_color(grid)
    out = np.full_like(grid, fill_value=bg)
    rows, cols = grid.shape
    if direction in ("up", "down"):
        for c in range(cols):
            col = grid[:, c]
            nonbg = col[col != bg]
            if direction == "up":
                out[:len(nonbg), c] = nonbg
            else:
                out[rows - len(nonbg):, c] = nonbg
    else:
        for r in range(rows):
            row = grid[r, :]
            nonbg = row[row != bg]
            if direction == "left":
                out[r, :len(nonbg)] = nonbg
            else:
                out[r, cols - len(nonbg):] = nonbg
    return out


@dataclass
class GravityUp:
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(_gravity(inp, "up"), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _gravity(grid, "up")

    def signature(self) -> tuple:
        return ("GravityUp",)


@dataclass
class GravityDown:
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(_gravity(inp, "down"), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _gravity(grid, "down")

    def signature(self) -> tuple:
        return ("GravityDown",)


@dataclass
class GravityLeft:
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(_gravity(inp, "left"), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _gravity(grid, "left")

    def signature(self) -> tuple:
        return ("GravityLeft",)


@dataclass
class GravityRight:
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(_gravity(inp, "right"), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return _gravity(grid, "right")

    def signature(self) -> tuple:
        return ("GravityRight",)


# ---------------------------------------------------------------------------
# BorderAndFill family
# ---------------------------------------------------------------------------

@dataclass
class DrawBorder:
    """Output = input with outer border drawn in a consistent color (inferred)."""
    color: int = 0
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates: Optional[set] = None
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            # discover which color filled the border in each pair
            border_colors = set()
            rows, cols = out.shape
            top = out[0, :].tolist()
            bot = out[-1, :].tolist()
            left = out[:, 0].tolist()
            right = out[:, -1].tolist()
            border = set(top + bot + left + right)
            # the border should differ from input borders to be a meaningful rule
            inp_top = inp[0, :].tolist()
            inp_bot = inp[-1, :].tolist()
            inp_left = inp[:, 0].tolist()
            inp_right = inp[:, -1].tolist()
            inp_border = set(inp_top + inp_bot + inp_left + inp_right)
            if border == inp_border:
                return False
            if len(border) != 1:
                return False
            c = border.pop()
            if candidates is None:
                candidates = {c}
            else:
                candidates &= {c}
            if not candidates:
                return False
            # check that interior is preserved
            inp_interior = inp[1:-1, 1:-1]
            out_interior = out[1:-1, 1:-1]
            if not np.array_equal(inp_interior, out_interior):
                return False
        if not candidates:
            return False
        self.color = candidates.pop()
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        out = grid.copy()
        out[0, :] = self.color
        out[-1, :] = self.color
        out[:, 0] = self.color
        out[:, -1] = self.color
        return out

    def signature(self) -> tuple:
        return ("DrawBorder", self.color)


@dataclass
class FillInterior:
    """Output = input with all bg cells replaced by an inferred color."""
    fill_color: int = 0
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates: Optional[set] = None
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            bg = _background_color(inp)
            bg_mask = inp == bg
            if not bg_mask.any():
                return False
            # at bg positions, output should be a consistent single color
            filled_colors = set(out[bg_mask].tolist())
            if len(filled_colors) != 1:
                return False
            c = filled_colors.pop()
            if c == bg:
                return False
            if candidates is None:
                candidates = {c}
            else:
                candidates &= {c}
            if not candidates:
                return False
            # non-bg positions should be preserved
            non_bg = ~bg_mask
            if not np.array_equal(inp[non_bg], out[non_bg]):
                return False
        if not candidates:
            return False
        self.fill_color = candidates.pop()
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        out = grid.copy()
        out[grid == bg] = self.fill_color
        return out

    def signature(self) -> tuple:
        return ("FillInterior", self.fill_color)


# ---------------------------------------------------------------------------
# Fixed-point family — apply until stable
# ---------------------------------------------------------------------------

@dataclass
class ApplyUntilStable:
    """Apply a child rule repeatedly until output stops changing or max_iter hit.
    The child is fitted from train pairs as one of: Symmetrize{H,V,Diag}.
    """
    child_name: str = "SymmetrizeH"
    max_iter: int = 8
    fitted: bool = False

    def _child(self):
        m = {
            "SymmetrizeH": SymmetrizeH(),
            "SymmetrizeV": SymmetrizeV(),
            "SymmetrizeDiag": SymmetrizeDiag(),
        }
        return m[self.child_name]

    def _apply_until_stable(self, grid: Grid, child) -> Grid:
        cur = np.asarray(grid).copy()
        for _ in range(self.max_iter):
            nxt = child.predict(cur)
            if np.array_equal(nxt, cur):
                break
            cur = nxt
        return cur

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for child_name in ("SymmetrizeH", "SymmetrizeV", "SymmetrizeDiag"):
            self.child_name = child_name
            child = self._child()
            ok = True
            for inp, out in train_pairs:
                inp = np.asarray(inp); out = np.asarray(out)
                if inp.shape != out.shape:
                    ok = False
                    break
                if not np.array_equal(self._apply_until_stable(inp, child), out):
                    ok = False
                    break
            if ok:
                # reject trivial (single-step is already covered by base rule)
                first_pair_inp = np.asarray(train_pairs[0][0])
                if np.array_equal(child.predict(first_pair_inp),
                                  self._apply_until_stable(first_pair_inp, child)):
                    continue
                self.fitted = True
                return True
        return False

    def predict(self, grid: Grid) -> Grid:
        return self._apply_until_stable(grid, self._child())

    def signature(self) -> tuple:
        return ("ApplyUntilStable", self.child_name, self.max_iter)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

ALL_WAVE4_RULES = [
    KeepLargest, KeepSmallest, KeepByMaxColor, KeepByMinColor,
    DeleteLargest, DeleteSmallest,
    SymmetrizeH, SymmetrizeV, SymmetrizeDiag,
    GravityUp, GravityDown, GravityLeft, GravityRight,
    DrawBorder, FillInterior,
    ApplyUntilStable,
]
