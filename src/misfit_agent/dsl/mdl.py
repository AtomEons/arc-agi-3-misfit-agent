"""MDL prior scorer — Occam / Solomonoff penalty for the synthesis engine.

The synthesis engine searches over typed Program ASTs. Among programs that
fit the training pairs equally well, shorter programs generalize better
(MDL / Solomonoff induction). This module gives the engine a principled
bit-length estimate so it can trade fit against simplicity:

    score = train_cell_accuracy - mdl_lambda * encoding_bits

where ``encoding_bits`` is the arithmetic-coded length of the Program AST
under the grammar:

    PrimitiveNode  : bits to identify the primitive (log2|catalog|) plus
                     parameter-encoding bits — already encapsulated in
                     ``Primitive.mdl_bits()``.
    HoleNode       : bits the synthesizer will eventually pay to fill the
                     hole — modelled as ``log2|catalog|``.
    ConstNode      : bits to encode the literal value, ``log2`` of the
                     value-type domain size (10 for Color, 2 for Bool, 64
                     for Number, ``cells * log2(10)`` for Grid, etc.).

Tier-1 disclosure: pure arithmetic over the type system and the catalog.
No learned weights, no LLM, no priors that touch the eval set.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .ast import (
    ConstNode,
    HoleNode,
    PrimitiveNode,
    Program,
)
from .interpreter import IncompleteProgramError, evaluate
from .primitives import ALL_PRIMITIVES
from .types import DslType


# ---------------------------------------------------------------------------
# Domain-size table for ConstNode bit costs
# ---------------------------------------------------------------------------
#
# Each DslType has a notional "domain size" — the number of distinct values
# a literal of that type can take. The arithmetic-coded bit cost of a literal
# is ``log2(domain_size)``.
#
#   COLOR   — ARC palette is 10 colors (0..9)
#   NUMBER  — counts/offsets, conservatively bounded at 64 (6 bits)
#   BOOL    — exactly 2 values
#   GRID    — handled specially (per-cell cost based on actual shape)
#   OBJECT  — bbox(rows*cols<=900) + color(10) ≈ ~13 bits notional
#   OBJSET  — variable; modelled as 8 * Object cost (rare as a ConstNode)
#   MASK    — handled specially (per-cell, 1 bit each)

_DOMAIN_SIZE_DEFAULT: dict[DslType, int] = {
    DslType.COLOR: 10,
    DslType.NUMBER: 64,
    DslType.BOOL: 2,
    # OBJECT and OBJSET get notional defaults; the actual values rarely
    # appear as ConstNodes — the synthesizer prefers structural composition.
    DslType.OBJECT: 1 << 13,
    DslType.OBJSET: 1 << 16,
}

# Minimum bit cost for any node (an arithmetic coder cannot emit 0 bits for a
# real choice — even Identity costs a discrimination bit).
_MIN_BITS = 1.0


def _grid_bits(value) -> float:
    """Bits to encode a literal grid value: log2(10) per cell."""
    arr = np.asarray(value)
    return float(arr.size) * math.log2(10)


def _mask_bits(value) -> float:
    """Bits to encode a literal boolean mask: 1 bit per cell."""
    arr = np.asarray(value)
    return float(arr.size)


# ---------------------------------------------------------------------------
# Per-node bit cost
# ---------------------------------------------------------------------------


def _hole_bits() -> float:
    """Bit cost of an unfilled hole — the synthesizer must still pick a
    primitive, so the lower bound is ``log2(|catalog|)``."""
    catalog = max(len(ALL_PRIMITIVES), 2)
    return math.log2(catalog)


def _const_bits(node: ConstNode) -> float:
    """Bit cost of a typed literal under the per-type domain table."""
    t = node.value_type
    if t == DslType.GRID:
        return _grid_bits(node.value)
    if t == DslType.MASK:
        return _mask_bits(node.value)
    size = _DOMAIN_SIZE_DEFAULT.get(t)
    if size is None or size <= 1:
        return _MIN_BITS
    return math.log2(size)


def _node_bits(node) -> float:
    """Bit cost of a single node — primitive, hole, or constant."""
    if isinstance(node, PrimitiveNode):
        # primitive.mdl_bits() already includes catalog-discrimination cost
        # AND parameter-encoding cost (Translate adds ~9.92 bits, etc.)
        return float(node.primitive.mdl_bits())
    if isinstance(node, HoleNode):
        return _hole_bits()
    if isinstance(node, ConstNode):
        return _const_bits(node)
    # Unknown node kind — refuse silently to over-credit.
    return _MIN_BITS


def _iter_nodes(node) -> Iterable:
    """Pre-order walk over a single AST root (no Program wrapper required)."""
    yield node
    if isinstance(node, PrimitiveNode):
        for child in node.children:
            yield from _iter_nodes(child)


# ---------------------------------------------------------------------------
# Public API: encoding_bits
# ---------------------------------------------------------------------------


def encoding_bits(program: Program) -> float:
    """Sum per-node bit costs for a Program AST.

    Args:
        program: a typed Program (possibly with holes).

    Returns:
        Total arithmetic-coded bits to describe the program under the
        grammar. Strictly positive for non-empty programs.
    """
    if program is None or program.root is None:
        return 0.0
    return float(sum(_node_bits(n) for n in _iter_nodes(program.root)))


# ---------------------------------------------------------------------------
# Cell-accuracy helpers
# ---------------------------------------------------------------------------


def _cell_accuracy(pred, target) -> float:
    """Cell-level accuracy between two grids.

    Returns 0.0 if shapes mismatch (the program produced a structurally
    wrong output). Returns mean(pred == target) otherwise.
    """
    try:
        p = np.asarray(pred)
        t = np.asarray(target)
    except Exception:
        return 0.0
    if p.shape != t.shape:
        return 0.0
    if p.size == 0:
        return 0.0
    return float(np.mean(p == t))


def train_cell_accuracy(
    program: Program,
    train_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Mean cell-accuracy of ``program`` over the train pairs.

    Each pair is ``(input_grid, output_grid)``. The program is evaluated on
    ``input_grid`` and the prediction is compared to ``output_grid`` with
    ``_cell_accuracy``. Shape mismatches and evaluation errors contribute
    0.0 to the mean.

    Args:
        program: a typed Program — should be complete (no holes) for a
                 meaningful score; incomplete programs score 0.0.
        train_pairs: sequence of (input_grid, output_grid) tuples.

    Returns:
        Mean cell-accuracy in [0.0, 1.0]. Empty ``train_pairs`` returns 0.0.
    """
    if not train_pairs:
        return 0.0
    accs: list[float] = []
    for inp, out in train_pairs:
        try:
            pred = evaluate(program, inp)
        except IncompleteProgramError:
            accs.append(0.0)
            continue
        except Exception:
            # Any runtime error (shape, value, etc.) counts as a miss.
            accs.append(0.0)
            continue
        accs.append(_cell_accuracy(pred, out))
    return float(sum(accs) / len(accs))


# ---------------------------------------------------------------------------
# Public API: score
# ---------------------------------------------------------------------------


def score(
    program: Program,
    train_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    mdl_lambda: float = 0.01,
) -> float:
    """Synthesis score: train fit minus MDL penalty.

    ::

        score = train_cell_accuracy - mdl_lambda * encoding_bits

    With ``mdl_lambda = 0`` the score collapses to ``train_cell_accuracy``,
    matching the fit-only objective. Larger ``mdl_lambda`` penalizes longer
    programs more heavily — the Solomonoff prior.

    Args:
        program: candidate Program.
        train_pairs: sequence of (input_grid, output_grid) tuples.
        mdl_lambda: non-negative weight on the bit-length penalty.

    Returns:
        Real-valued score; higher is better.
    """
    acc = train_cell_accuracy(program, train_pairs)
    bits = encoding_bits(program)
    return float(acc) - float(mdl_lambda) * float(bits)


__all__ = [
    "encoding_bits",
    "train_cell_accuracy",
    "score",
]
