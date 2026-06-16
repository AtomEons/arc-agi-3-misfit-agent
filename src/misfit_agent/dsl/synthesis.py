"""Beam-search program synthesis over typed program ASTs.

The synthesis engine searches for a Program p such that, for every train
pair (x_i, y_i), evaluate(p, x_i) ≈ y_i. The space is otherwise vast, so
we keep it tractable with three levers:

  1. Typed expansion — a hole of expected_type=T can only be filled by a
     primitive whose output type is T. The type system in dsl/types.py
     does the heavy pruning at construction time.
  2. Beam search — at each depth we keep the top `beam_width` programs by
     score (train cell accuracy − λ × MDL bits).
  3. Time budget — the loop checks wall-clock time after each candidate
     and bails out cleanly.

This first version enumerates atomic primitives only (the combinator
integration team layers Seq/Parallel/Reduce/etc. on top via the same
expansion API). Even with atomics only, the parametric primitives
(Translate, Rotate, Reflect, Recolor, Tile, Gravity, Symmetrize,
KeepWhere) generate a sizeable parameter grid that we enumerate
explicitly here, with parameter choices derived from train-pair
inspection where helpful (e.g. Recolor draws color-overlap mappings).

Tier-1 honest by construction:
  - No learned parameters
  - No pretrained weights
  - Pure structural enumeration + brute-force scoring on the train pairs

Public surface:
  synthesize(train_pairs, max_depth, beam_width, time_budget_s) -> list[Program]

Scoring (higher is better):
  score(p) = mean_cell_accuracy(p, train_pairs) − λ × MDL_bits(p)
  where λ = MDL_LAMBDA = 0.01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import math
import time

import numpy as np

from .types import Grid, DslType
from .primitives import (
    Primitive,
    Identity, Translate, Rotate, Reflect, Recolor,
    Crop, Tile, Gravity, Symmetrize, KeepWhere,
)
from .ast import Program, PrimitiveNode, HoleNode, make_hole, make_program
from .interpreter import evaluate, IncompleteProgramError
from .walker import total_mdl_bits


# MDL penalty coefficient — tie-breaker that pushes shorter programs above
# longer programs when their train-fit is equal. Specified in the task brief.
MDL_LAMBDA: float = 0.01


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _cell_accuracy(predicted: Any, target: np.ndarray) -> float:
    """Per-cell accuracy of predicted grid against target.

    Mismatched shapes score 0.0. Non-array predictions (e.g. a Number from
    CountObj that landed at the root by mistake) also score 0.0. This
    keeps synthesis aimed at Grid-output programs even though the search
    surface formally supports type-changing primitives.
    """
    if not isinstance(predicted, np.ndarray):
        # Could be a Python int from CountObj or similar. Not a grid.
        return 0.0
    target = np.asarray(target)
    if predicted.shape != target.shape:
        return 0.0
    total = target.size
    if total == 0:
        return 0.0
    matches = int(np.sum(predicted == target))
    return matches / total


def _train_score(program: Program,
                 train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean per-pair cell accuracy. Programs that fail to evaluate score 0.0."""
    if not train_pairs:
        return 0.0
    acc_sum = 0.0
    for x, y in train_pairs:
        try:
            pred = evaluate(program, x)
        except (IncompleteProgramError, ValueError, IndexError,
                KeyError, TypeError, AttributeError):
            return 0.0
        acc_sum += _cell_accuracy(pred, y)
    return acc_sum / len(train_pairs)


def _final_score(program: Program,
                 train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Combined train fit + MDL penalty. Higher is better."""
    fit = _train_score(program, train_pairs)
    mdl = total_mdl_bits(program)
    return fit - MDL_LAMBDA * mdl


# ---------------------------------------------------------------------------
# Parameter-grid enumeration
# ---------------------------------------------------------------------------


def _color_overlap_mappings(
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    max_size: int = 3,
) -> list[dict[int, int]]:
    """Derive Recolor candidate mappings from train-pair color overlap.

    Strategy: for every train pair, find colors present in input but absent
    or different in output and propose a (input_color -> output_color)
    mapping. We collect unique mappings of size 1..max_size.
    """
    candidates: list[dict[int, int]] = []
    seen: set[tuple] = set()
    per_pair_pairs: list[set[tuple[int, int]]] = []

    for x, y in train_pairs:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            per_pair_pairs.append(set())
            continue
        pair_set: set[tuple[int, int]] = set()
        # Find cells whose color changed between input and output.
        diff_mask = x != y
        if diff_mask.any():
            in_colors = x[diff_mask]
            out_colors = y[diff_mask]
            for ic, oc in zip(in_colors.tolist(), out_colors.tolist()):
                pair_set.add((int(ic), int(oc)))
        per_pair_pairs.append(pair_set)

    # Single-color mappings: every (a -> b) seen in any pair.
    all_singles: set[tuple[int, int]] = set()
    for pair_set in per_pair_pairs:
        all_singles.update(pair_set)
    for a, b in sorted(all_singles):
        m = {a: b}
        key = tuple(sorted(m.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(m)

    if max_size >= 2:
        # Pairs of single mappings that share consistent direction across pairs.
        singles_list = sorted(all_singles)
        for i in range(len(singles_list)):
            for j in range(i + 1, len(singles_list)):
                a1, b1 = singles_list[i]
                a2, b2 = singles_list[j]
                if a1 == a2:
                    # Conflicting source — skip; can't map same color two ways.
                    continue
                m = {a1: b1, a2: b2}
                key = tuple(sorted(m.items()))
                if key not in seen:
                    seen.add(key)
                    candidates.append(m)

    if max_size >= 3:
        singles_list = sorted(all_singles)
        for i in range(len(singles_list)):
            for j in range(i + 1, len(singles_list)):
                for k in range(j + 1, len(singles_list)):
                    a1, b1 = singles_list[i]
                    a2, b2 = singles_list[j]
                    a3, b3 = singles_list[k]
                    srcs = {a1, a2, a3}
                    if len(srcs) != 3:
                        continue
                    m = {a1: b1, a2: b2, a3: b3}
                    key = tuple(sorted(m.items()))
                    if key not in seen:
                        seen.add(key)
                        candidates.append(m)

    return candidates


def _atomic_primitive_candidates(
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[Primitive]:
    """Enumerate every atomic primitive instance worth trying.

    Per the task brief:
      Identity: 1 instance
      Translate: (dy, dx) in [-3, 3]^2 — 49 instances
      Rotate: k in {1, 2, 3} — 3 instances
      Reflect: axis in {H, V, D1} — 3 instances
      Recolor: train-derived mappings of size 1..3
      Crop: 1 instance
      Tile: (rf, cf) in {(2,1),(1,2),(2,2),(3,1),(1,3)} — 5 instances
      Gravity: direction in {U, D, L, R} — 4 instances
      Symmetrize: axis in {H, V, BOTH} — 3 instances
      KeepWhere: predicate in {largest, smallest, edge_touching, non_edge} — 4

    All instances yield Grid -> Grid programs so they fit the root Grid hole.
    """
    out: list[Primitive] = [Identity(), Crop()]

    for dy in range(-3, 4):
        for dx in range(-3, 4):
            out.append(Translate(dy=dy, dx=dx))

    for k in (1, 2, 3):
        out.append(Rotate(k=k))

    for axis in ("H", "V", "D1"):
        out.append(Reflect(axis=axis))

    for mapping in _color_overlap_mappings(train_pairs):
        out.append(Recolor(mapping=dict(mapping)))

    for rf, cf in ((2, 1), (1, 2), (2, 2), (3, 1), (1, 3)):
        out.append(Tile(rf=rf, cf=cf))

    for direction in ("U", "D", "L", "R"):
        out.append(Gravity(direction=direction))

    for axis in ("H", "V", "BOTH"):
        out.append(Symmetrize(axis=axis))

    for predicate in ("largest", "smallest", "edge_touching", "non_edge"):
        out.append(KeepWhere(predicate=predicate))

    return out


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------


@dataclass
class _BeamEntry:
    """A candidate program plus its memoized score and hash for dedup."""
    program: Program
    score: float
    program_hash: str = ""

    def __post_init__(self):
        if not self.program_hash:
            self.program_hash = self.program.sha256_hash()


def _dedup_top_k(entries: Iterable[_BeamEntry], k: int) -> list[_BeamEntry]:
    """Keep the top-k entries by score, deduplicating by program hash."""
    seen: set[str] = set()
    unique: list[_BeamEntry] = []
    for e in sorted(entries, key=lambda e: e.score, reverse=True):
        if e.program_hash in seen:
            continue
        seen.add(e.program_hash)
        unique.append(e)
        if len(unique) >= k:
            break
    return unique


# ---------------------------------------------------------------------------
# Public synthesis entry point
# ---------------------------------------------------------------------------


def synthesize(
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    max_depth: int = 3,
    beam_width: int = 8,
    time_budget_s: float = 5.0,
) -> list[Program]:
    """Beam-search the program space; return top-K candidates.

    Args:
        train_pairs: list of (input_grid, output_grid) demonstrations
        max_depth: cap on program AST depth (reserved for combinator
            expansion — atomic-only synthesis is always depth 1)
        beam_width: number of candidates retained per generation; also
            the size of the returned list
        time_budget_s: soft wall-clock cap; the loop returns whatever it
            has when the budget is exhausted

    Returns:
        Sorted-descending list of up to `beam_width` Programs by score.
        Returns [] when train_pairs is empty.
    """
    if not train_pairs:
        return []

    # Normalize train pairs to numpy arrays once so we don't pay the
    # conversion cost inside the scoring inner loop.
    norm_pairs: list[tuple[np.ndarray, np.ndarray]] = [
        (np.asarray(x), np.asarray(y)) for x, y in train_pairs
    ]

    start = time.monotonic()
    deadline = start + max(0.0, float(time_budget_s))

    # Enumerate every atomic-primitive candidate. Each becomes a depth-1
    # program: prim(<input_hole:Grid>).
    candidate_prims = _atomic_primitive_candidates(norm_pairs)

    beam: list[_BeamEntry] = []
    for prim in candidate_prims:
        # Time-budget guard: bail out cleanly with what we have so far.
        if time.monotonic() > deadline:
            break
        try:
            prog = make_program(prim, make_hole(Grid))
        except Exception:
            # If construction (typed) ever rejects a candidate we silently
            # drop it — synthesis is a search, not an assertion.
            continue
        if prog.output_type() != Grid:
            continue
        if prog.depth() > max_depth:
            continue
        score = _final_score(prog, norm_pairs)
        beam.append(_BeamEntry(program=prog, score=score))

    top = _dedup_top_k(beam, beam_width)
    return [e.program for e in top]


__all__ = [
    "synthesize",
    "MDL_LAMBDA",
]
