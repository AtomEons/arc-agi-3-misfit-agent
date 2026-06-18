"""Wave 13 — per-test relational fallback for novel-color and novel-shape tasks.

Targets the gap in ColorMap-class rules where train output uses colors A->B
but the test input has a COLOR UNSEEN IN TRAINING (task aabf363d-style).

Two primitives:

  * NovelColorRecolor — if test input contains a color not in any train input,
    map it to the color the train output uses in the analogous role.
    Inferred relation: "the foreground color of the test input plays the same
    role as the foreground color of the train inputs."

  * PaletteBijection — fit a per-pair bijection between INPUT colors and
    OUTPUT colors, then apply per-test-input by composing the discovered
    bijections (extend with identity for novel colors).

Tier-1 strict: no LLM, no learned params at eval. Inference is per-test-input
deterministic enumeration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..perceptor import _background_color


Grid = np.ndarray


# ---------------------------------------------------------------------------
# NovelColorRecolor
#
# Strategy:
#   - For each train pair, find the "dominant non-bg color" of the input.
#     This is the FG color. Find the "dominant non-bg color" of the output.
#     Record the (fg_in, fg_out) pair.
#   - If all train pairs share a SINGLE (fg_in, fg_out) → that's a literal
#     ColorMap (handled elsewhere). Skip.
#   - If train pairs all share a SINGLE fg_out (universal target) → ANY
#     non-bg color in test input maps to that target.
#   - If train pairs each have their own (fg_in, fg_out) but the relation
#     is RELATIONAL ("fg_out = fg_in")  → identity.
#   - Otherwise, infer relation: fg_out = f(fg_in, palette). Try a small
#     family of functions:
#       * constant -> fg_out always K
#       * permutation by index in a sorted palette of input colors
# ---------------------------------------------------------------------------


def _dominant_non_bg(grid: Grid) -> Optional[int]:
    bg = _background_color(grid)
    g = np.asarray(grid)
    counts = np.bincount(g.ravel(), minlength=10).copy()
    counts[bg] = 0
    if counts.sum() == 0:
        return None
    return int(counts.argmax())


@dataclass
class NovelColorRecolor:
    """If all train outputs paint the dominant non-bg color of the input
    with a single target color, apply that mapping at test time even if
    the test input's dominant non-bg color is novel.
    """
    target: int = -1
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        targets = set()
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            bg_in = _background_color(inp)
            bg_out = _background_color(out)
            fg_in = _dominant_non_bg(inp)
            fg_out = _dominant_non_bg(out)
            if fg_in is None or fg_out is None:
                return False
            # Output must equal: input with bg unchanged and fg_in replaced by fg_out
            expected = inp.copy()
            mask = (inp != bg_in)
            expected[mask] = fg_out
            if not np.array_equal(expected, out):
                return False
            targets.add(fg_out)
        # Universal target — all train pairs map dominant non-bg to same color
        if len(targets) != 1:
            return False
        target = next(iter(targets))
        # Don't fit if it's a no-op recolor (train inputs already that color)
        if all(_dominant_non_bg(np.asarray(inp)) == target
               for inp, _ in train_pairs):
            return False
        self.target = target
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        out = grid.copy()
        out[grid != bg] = self.target
        return out

    def signature(self) -> tuple:
        return ("NovelColorRecolor", self.target)


# ---------------------------------------------------------------------------
# PaletteBijection — extend with identity for novel test colors.
#
# We try to fit ONE color mapping that explains all train pairs. Colors not
# in the train input palette but appearing in the test input default to
# IDENTITY (keep them).
# ---------------------------------------------------------------------------


@dataclass
class PaletteBijectionWithIdentityExtension:
    """Same as Recolor but extends with identity for novel test colors and
    accepts when the SAME bijection holds across all train pairs."""
    mapping: dict = field(default_factory=dict)
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        merged: dict[int, int] = {}
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            for c_in in range(10):
                mask = inp == c_in
                if not mask.any():
                    continue
                outs = np.unique(out[mask])
                if outs.size != 1:
                    return False
                target = int(outs[0])
                if c_in in merged and merged[c_in] != target:
                    return False
                merged[c_in] = target
        # Reject pure identity (covered by Identity rule)
        if all(k == v for k, v in merged.items()):
            return False
        # Reject if covered by plain Recolor (we want novel-color extension semantics)
        self.mapping = merged
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        out = grid.copy()
        for k, v in self.mapping.items():
            out[grid == k] = v
        # Novel colors -> identity (no remap)
        return out

    def signature(self) -> tuple:
        return ("PaletteBijectionWithIdentityExtension",
                tuple(sorted(self.mapping.items())))


# ---------------------------------------------------------------------------
# PaintBgWithMissingColor
#
# Some tasks: input has K colors, output has K-1 colors plus all bg cells
# painted with the "missing" color. Detect by:
#   - identify (in_palette, out_palette)
#   - if out_palette = (in_palette \ bg_in) ∪ {missing}, where missing is
#     a color absent from in_palette but present in out
# ---------------------------------------------------------------------------


@dataclass
class PaintBgWithMissingNonBgColor:
    """Predict: paint all bg cells with the color that train outputs introduced."""
    new_color: int = -1
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        introductions = set()
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            bg_in = _background_color(inp)
            in_palette = set(np.unique(inp).tolist())
            out_palette = set(np.unique(out).tolist())
            new = out_palette - in_palette
            if len(new) != 1:
                return False
            new_color = next(iter(new))
            # Check output is input with bg cells painted new_color
            expected = inp.copy()
            expected[inp == bg_in] = new_color
            if not np.array_equal(expected, out):
                return False
            introductions.add(new_color)
        if len(introductions) != 1:
            return False
        self.new_color = next(iter(introductions))
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        bg = _background_color(grid)
        out = grid.copy()
        out[grid == bg] = self.new_color
        return out

    def signature(self) -> tuple:
        return ("PaintBgWithMissingNonBgColor", self.new_color)


ALL_WAVE13_RULES = [
    NovelColorRecolor,
    PaletteBijectionWithIdentityExtension,
    PaintBgWithMissingNonBgColor,
]
