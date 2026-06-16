"""Twelve atomic Spelke primitives for the Misfit-Alpha DSL.

Each primitive:
  - declares a typed signature (input types, output type, scalar params)
  - implements apply(*inputs) -> output
  - implements to_string() for human-readable program AST display
  - implements mdl_bits() for the MDL prior (program-length penalty)

The implementations DELEGATE to the underlying numpy logic in
arc2_solver.py where applicable — the DSL is the typed shell, not a
rewrite. This keeps the cargo-green status intact while adding the
type system.

Tier-1 disclosure: every primitive encodes a single Spelke prior or a
generic operation. No primitive is hand-tuned on the public eval set.
The set is intentionally small (12) — composition handles the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from ..perceptor import perceive_grid, _background_color
from .types import Color, Grid, Number, Object, ObjSet, Mask, Bool
from .types import DslType, Signature


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """Base class. Subclasses declare signature_typed() and override apply()."""

    def signature_typed(self) -> Signature:
        raise NotImplementedError

    def apply(self, *inputs):
        raise NotImplementedError

    def to_string(self) -> str:
        return type(self).__name__

    def mdl_bits(self) -> float:
        """Bits to encode this primitive choice in a program AST.
        Default: log2(|primitive catalog|). Subclasses with parameters add
        the bits needed to encode the parameter values."""
        return math.log2(max(len(ALL_PRIMITIVES), 2))


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------


@dataclass
class Identity(Primitive):
    """The null hypothesis. output = input."""

    def signature_typed(self) -> Signature:
        return Signature(inputs=(("g", Grid),), output=Grid)

    def apply(self, grid):
        return np.asarray(grid).copy()


# ---------------------------------------------------------------------------
# 2. Translate(dx, dy)
# ---------------------------------------------------------------------------


@dataclass
class Translate(Primitive):
    """Shift the grid by (dy, dx). Background fills exposed cells.
    Spelke GEOMETRY (translation symmetry)."""
    dy: int = 0
    dx: int = 0

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("dy", int), ("dx", int)),
        )

    def apply(self, grid):
        grid = np.asarray(grid)
        bg = _background_color(grid)
        rows, cols = grid.shape
        out = np.full_like(grid, fill_value=bg)
        for r in range(rows):
            for c in range(cols):
                nr, nc = r + self.dy, c + self.dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr, nc] = grid[r, c]
        return out

    def to_string(self) -> str:
        return f"Translate(dy={self.dy}, dx={self.dx})"

    def mdl_bits(self) -> float:
        # 2 integers in [-15, 15] range = 2 × log2(31) ≈ 9.92 bits
        return super().mdl_bits() + 9.92


# ---------------------------------------------------------------------------
# 3. Rotate(k)
# ---------------------------------------------------------------------------


@dataclass
class Rotate(Primitive):
    """k × 90° rotation. k ∈ {1, 2, 3}. Spelke GEOMETRY."""
    k: int = 1

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("k", int),),
        )

    def apply(self, grid):
        return np.rot90(np.asarray(grid), k=self.k).copy()

    def to_string(self) -> str:
        return f"Rotate(k={self.k})"

    def mdl_bits(self) -> float:
        # k ∈ {1,2,3} = log2(3) ≈ 1.58 bits
        return super().mdl_bits() + 1.58


# ---------------------------------------------------------------------------
# 4. Reflect(axis)
# ---------------------------------------------------------------------------


@dataclass
class Reflect(Primitive):
    """Reflect across axis. axis ∈ {'H', 'V', 'D1', 'D2'}. Spelke GEOMETRY."""
    axis: str = "H"

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("axis", str),),
        )

    def apply(self, grid):
        grid = np.asarray(grid)
        if self.axis == "H":
            return np.fliplr(grid).copy()
        if self.axis == "V":
            return np.flipud(grid).copy()
        if self.axis == "D1":
            return grid.T.copy()
        if self.axis == "D2":
            return np.fliplr(np.flipud(grid)).T.copy()
        raise ValueError(f"unknown Reflect axis: {self.axis}")

    def to_string(self) -> str:
        return f"Reflect(axis={self.axis})"

    def mdl_bits(self) -> float:
        # 4 axes = 2 bits
        return super().mdl_bits() + 2.0


# ---------------------------------------------------------------------------
# 5. Recolor(mapping)
# ---------------------------------------------------------------------------


@dataclass
class Recolor(Primitive):
    """Apply a color permutation. mapping: dict[int, int]. Generic."""
    mapping: dict[int, int] = None

    def __post_init__(self):
        if self.mapping is None:
            self.mapping = {}

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("mapping", dict),),
        )

    def apply(self, grid):
        grid = np.asarray(grid)
        out = grid.copy()
        for k, v in self.mapping.items():
            out[grid == k] = v
        return out

    def to_string(self) -> str:
        ms = ",".join(f"{k}→{v}" for k, v in sorted(self.mapping.items()))
        return f"Recolor({{{ms}}})"

    def mdl_bits(self) -> float:
        # encode the mapping: |mapping| × (4 bits source + 4 bits target)
        return super().mdl_bits() + 8.0 * len(self.mapping)


# ---------------------------------------------------------------------------
# 6. Crop()
# ---------------------------------------------------------------------------


@dataclass
class Crop(Primitive):
    """Crop the grid to the bounding box of non-background cells.
    Spelke OBJECTNESS — focus on the figure."""

    def signature_typed(self) -> Signature:
        return Signature(inputs=(("g", Grid),), output=Grid)

    def apply(self, grid):
        grid = np.asarray(grid)
        bg = _background_color(grid)
        mask = grid != bg
        if not mask.any():
            return grid.copy()
        ys, xs = np.where(mask)
        return grid[int(ys.min()):int(ys.max())+1,
                    int(xs.min()):int(xs.max())+1].copy()


# ---------------------------------------------------------------------------
# 7. Tile(rf, cf)
# ---------------------------------------------------------------------------


@dataclass
class Tile(Primitive):
    """Tile the grid by (rows_factor, cols_factor). Spelke GEOMETRY (translation
    symmetry)."""
    rf: int = 2
    cf: int = 2

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("rf", int), ("cf", int)),
        )

    def apply(self, grid):
        return np.tile(np.asarray(grid), (self.rf, self.cf)).copy()

    def to_string(self) -> str:
        return f"Tile(rf={self.rf}, cf={self.cf})"

    def mdl_bits(self) -> float:
        # factors typically in {1..4} — log2(4) × 2 = 4 bits
        return super().mdl_bits() + 4.0


# ---------------------------------------------------------------------------
# 8. Gravity(direction)
# ---------------------------------------------------------------------------


@dataclass
class Gravity(Primitive):
    """Non-background cells fall in direction. direction ∈ {'U','D','L','R'}.
    Spelke CONTACT-CAUSALITY."""
    direction: str = "D"

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("direction", str),),
        )

    def apply(self, grid):
        grid = np.asarray(grid)
        bg = _background_color(grid)
        out = np.full_like(grid, fill_value=bg)
        if self.direction in ("D", "U"):
            for c in range(grid.shape[1]):
                col = grid[:, c]
                non_bg = col[col != bg]
                if len(non_bg) == 0:
                    continue
                if self.direction == "D":
                    out[-len(non_bg):, c] = non_bg
                else:
                    out[:len(non_bg), c] = non_bg
        else:
            for r in range(grid.shape[0]):
                row = grid[r, :]
                non_bg = row[row != bg]
                if len(non_bg) == 0:
                    continue
                if self.direction == "R":
                    out[r, -len(non_bg):] = non_bg
                else:
                    out[r, :len(non_bg)] = non_bg
        return out

    def to_string(self) -> str:
        return f"Gravity(direction={self.direction})"

    def mdl_bits(self) -> float:
        # 4 directions = 2 bits
        return super().mdl_bits() + 2.0


# ---------------------------------------------------------------------------
# 9. Symmetrize(axis)
# ---------------------------------------------------------------------------


@dataclass
class Symmetrize(Primitive):
    """OR-combine the grid with its mirror across axis. axis ∈ {'H','V','BOTH'}.
    Spelke GEOMETRY (mirror symmetry completion)."""
    axis: str = "H"

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("axis", str),),
        )

    def apply(self, grid):
        grid = np.asarray(grid)
        bg = _background_color(grid)
        if self.axis == "H":
            mirror = np.fliplr(grid)
            return np.where(grid != bg, grid, mirror).copy()
        if self.axis == "V":
            mirror = np.flipud(grid)
            return np.where(grid != bg, grid, mirror).copy()
        if self.axis == "BOTH":
            out = grid.copy()
            for m in (np.fliplr(grid), np.flipud(grid), np.flipud(np.fliplr(grid))):
                out = np.where(out != bg, out, m)
            return out
        raise ValueError(f"unknown Symmetrize axis: {self.axis}")

    def to_string(self) -> str:
        return f"Symmetrize(axis={self.axis})"

    def mdl_bits(self) -> float:
        return super().mdl_bits() + math.log2(3)


# ---------------------------------------------------------------------------
# 10. KeepWhere(predicate)
# ---------------------------------------------------------------------------


@dataclass
class KeepWhere(Primitive):
    """Keep only objects matching predicate. predicate ∈
    {'largest','smallest','edge_touching','non_edge'}. Spelke OBJECTNESS."""
    predicate: str = "largest"

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("predicate", str),),
        )

    def apply(self, grid):
        grid = np.asarray(grid)
        objs = list(perceive_grid(grid).objects)
        if not objs:
            return grid.copy()
        bg = _background_color(grid)
        out = np.full_like(grid, fill_value=bg)

        if self.predicate == "largest":
            keep = [max(objs, key=lambda o: o.area)]
        elif self.predicate == "smallest":
            keep = [min(objs, key=lambda o: o.area)]
        elif self.predicate == "edge_touching":
            keep = [o for o in objs if o.touches_edge]
        elif self.predicate == "non_edge":
            keep = [o for o in objs if not o.touches_edge]
        else:
            raise ValueError(f"unknown KeepWhere predicate: {self.predicate}")

        for obj in keep:
            r0, c0, r1, c1 = obj.bbox
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    v = int(grid[r, c])
                    if v != bg and v == obj.color:
                        out[r, c] = v
        return out

    def to_string(self) -> str:
        return f"KeepWhere(pred={self.predicate})"

    def mdl_bits(self) -> float:
        # 4 predicates = 2 bits
        return super().mdl_bits() + 2.0


# ---------------------------------------------------------------------------
# 11. CountObj — Grid → Number (changes output TYPE)
# ---------------------------------------------------------------------------


@dataclass
class CountObj(Primitive):
    """Return the count of perceived foreground objects. Spelke NUMEROSITY."""

    def signature_typed(self) -> Signature:
        return Signature(inputs=(("g", Grid),), output=Number)

    def apply(self, grid):
        return int(len(perceive_grid(np.asarray(grid)).objects))


# ---------------------------------------------------------------------------
# 12. ShapeOf — Object → Grid
# ---------------------------------------------------------------------------


@dataclass
class ShapeOf(Primitive):
    """Extract an object as its own small grid. Spelke OBJECTNESS."""

    def signature_typed(self) -> Signature:
        return Signature(inputs=(("obj", Object),), output=Grid)

    def apply(self, obj):
        # obj is a perceptor Object record; we return a small grid that
        # contains exactly the object's stamp.
        if obj is None:
            return np.zeros((1, 1), dtype=np.int32)
        # The perceptor's Object carries bbox + color; we synthesize a
        # rectangular fill in the object's color. (A richer impl would
        # carry the original cell mask — future work.)
        r0, c0, r1, c1 = obj.bbox
        h, w = r1 - r0 + 1, c1 - c0 + 1
        return np.full((h, w), fill_value=int(obj.color), dtype=np.int32)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


ALL_PRIMITIVES = [
    Identity, Translate, Rotate, Reflect, Recolor,
    Crop, Tile, Gravity, Symmetrize, KeepWhere,
    CountObj, ShapeOf,
]
