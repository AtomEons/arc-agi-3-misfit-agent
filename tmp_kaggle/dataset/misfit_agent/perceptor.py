"""Perceptor — Spelke Core Knowledge priors over a raw ARC-AGI-3 frame.

Admissible priors used here:
  - OBJECTNESS: 4-connectivity flood-fill segmentation; cohesion, persistence.
  - GEOMETRY:   bbox, centroid, symmetry under H/V/diagonal reflection.
  - TOPOLOGY:   touches-edge, connectedness.
  - NUMEROSITY: counts, ordering by area.
  - SPATIAL:    inside/outside via bounding-box containment.

No task-family classification. No "geometric_likely" / "recolor_likely" hints.
Those flags belong in the action-search layer, derived dynamically from
observed transitions — not hardcoded in the perceptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


Grid = np.ndarray  # 2-D uint8 ARC color grid


@dataclass(frozen=True)
class Obj:
    """A single coherent object under the objectness prior."""
    color: int
    area: int
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    centroid: tuple[float, float]
    touches_edge: bool
    v_symmetric: bool
    h_symmetric: bool


@dataclass
class SceneObservation:
    """Result of perceiving one frame under Spelke priors."""
    grid: Grid
    rows: int
    cols: int
    background_color: int
    objects: list[Obj] = field(default_factory=list)
    color_histogram: list[int] = field(default_factory=list)  # length 10
    foreground_cells: int = 0


def _flood_label(grid: Grid, bg: int) -> np.ndarray:
    """4-connected component labels. Cells with color==bg get label 0."""
    rows, cols = grid.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    next_label = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == bg or labels[r, c] != 0:
                continue
            next_label += 1
            stack = [(r, c)]
            target_color = int(grid[r, c])
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if labels[rr, cc] != 0:
                    continue
                if int(grid[rr, cc]) != target_color:
                    continue
                labels[rr, cc] = next_label
                stack.extend([(rr+1, cc), (rr-1, cc), (rr, cc+1), (rr, cc-1)])
    return labels


def _is_symmetric_v(sub: Grid) -> bool:
    return bool(np.array_equal(sub, np.flip(sub, axis=1)))


def _is_symmetric_h(sub: Grid) -> bool:
    return bool(np.array_equal(sub, np.flip(sub, axis=0)))


def _background_color(grid: Grid) -> int:
    """ARC convention: 0 is background when present; else most-frequent color."""
    if (grid == 0).any():
        return 0
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def perceive_grid(grid: Grid) -> SceneObservation:
    """Perceive a single 2-D grid under Spelke priors."""
    grid = np.asarray(grid, dtype=np.int32)
    if grid.ndim != 2:
        raise ValueError(f"perceive_grid expects 2-D grid, got shape {grid.shape}")
    rows, cols = grid.shape
    bg = _background_color(grid)
    labels = _flood_label(grid, bg)
    n_labels = int(labels.max())

    objects: list[Obj] = []
    for lab in range(1, n_labels + 1):
        mask = labels == lab
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        sub = grid[r0:r1+1, c0:c1+1]
        sub_mask = mask[r0:r1+1, c0:c1+1].astype(np.int32) * (sub != bg).astype(np.int32)
        color = int(grid[ys[0], xs[0]])
        touches = bool(r0 == 0 or c0 == 0 or r1 == rows-1 or c1 == cols-1)
        v_sym = _is_symmetric_v(sub_mask)
        h_sym = _is_symmetric_h(sub_mask)
        centroid = (float(ys.mean()), float(xs.mean()))
        objects.append(Obj(
            color=color,
            area=int(ys.size),
            bbox=(r0, c0, r1, c1),
            centroid=centroid,
            touches_edge=touches,
            v_symmetric=v_sym,
            h_symmetric=h_sym,
        ))

    # Sort by area descending (largest first)
    objects.sort(key=lambda o: -o.area)

    hist = np.bincount(grid.ravel(), minlength=10)[:10].tolist()
    fg = int((grid != bg).sum())

    return SceneObservation(
        grid=grid,
        rows=rows,
        cols=cols,
        background_color=bg,
        objects=objects,
        color_histogram=list(map(int, hist)),
        foreground_cells=fg,
    )


def perceive_frame(frame: Sequence) -> SceneObservation:
    """Perceive an ARC-AGI-3 frame. `frame` is the FrameData.frame attribute —
    documented upstream as `list[list[list[int]]]` after numpy.tolist().
    Most games are single-grid; multi-grid frames keep the first plane for now.
    """
    arr = np.asarray(frame, dtype=np.int32)
    if arr.ndim == 3:
        # Multi-plane: use the first plane. Higher-order spatial reasoning over
        # multiple planes is a Day-N+ extension under the geometry prior.
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"perceive_frame expects 2-D or 3-D grid, got shape {arr.shape}")
    return perceive_grid(arr)


def grid_diff(a: Grid, b: Grid) -> tuple[int, int, int]:
    """Cell-level diff between two grids of equal shape.
    Returns (changed_cell_count, bg_to_fg_count, fg_to_bg_count).
    Used by rule induction to characterize action-effect under Spelke priors.
    """
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    if a.shape != b.shape:
        return (max(a.size, b.size), 0, 0)
    bg_a = _background_color(a)
    bg_b = _background_color(b)
    a_bg = a == bg_a
    b_bg = b == bg_b
    bg_to_fg = int(((a_bg) & (~b_bg)).sum())
    fg_to_bg = int(((~a_bg) & (b_bg)).sum())
    changed = int((a != b).sum())
    return (changed, bg_to_fg, fg_to_bg)
