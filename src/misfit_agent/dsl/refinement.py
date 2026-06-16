"""HRM-style outer refinement loop — pure-structural error-driven editing.

Per the ARC Prize HRM analysis (arcprize.org/blog/hrm-analysis), the
"refinement" step gave +13pp from 0 → 1 iteration. The HRM paper uses
gradient updates; this module achieves the same outer-loop behaviour with
purely structural edits over the typed Program AST. No neural gradient,
no learned parameters, no pretrained weights — only deterministic
candidate enumeration scored against the train pairs.

The outer loop is the same shape as the synthesis beam-search inner loop
(propose candidate edits → score on train pairs → keep best), but it
operates on an *already-constructed* program rather than re-enumerating
from scratch. The point is that synthesis hands us a near-miss and the
refinement loop nudges it to a hit.

Outer loop sketch
-----------------
For at most `max_iters` rounds:

  1. Score the current program against the train pairs (cell accuracy
     averaged across pairs). If it is already perfect, stop.
  2. Inspect the per-pair predictions to characterise the dominant
     error type (color-wise off / shape-wise off / position-off).
  3. Generate structural-edit candidates biased by the dominant error
     type:
        - position-off  → mutate Translate(dy, dx) parameters
        - shape-wise off → swap or wrap with Rotate/Reflect/Crop/Tile
        - color-wise off → swap or wrap with Recolor / Identity
        - generic       → all three edits at every primitive node
  4. Score each candidate. If the best candidate scores strictly higher
     than the current best, accept it; otherwise stop early.

Public API
----------
  refine(program, train_pairs, max_iters=4) -> Program
  swap_primitive(program, target_idx, new_primitive) -> Program
  wrap_program(program, wrapping_primitive) -> Program
  mutate_param(program, target_idx, param_name, new_value) -> Program

Tier-1 honest by construction
-----------------------------
  - Pure structural edits — no learned parameters
  - No pretrained weights, no LLM in the loop
  - Edit candidates are enumerated deterministically from the same
    primitive catalog the synthesis engine uses
  - Scoring is the same cell-accuracy metric the synthesis engine uses
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable
import copy

import numpy as np

from .types import DslType, Grid, TypeMismatchError, type_signature
from .primitives import (
    Primitive,
    Identity, Translate, Rotate, Reflect, Recolor,
    Crop, Tile, Gravity, Symmetrize, KeepWhere,
)
from .ast import Program, PrimitiveNode, HoleNode, ConstNode, make_hole
from .interpreter import evaluate, IncompleteProgramError
from .walker import walk_preorder, find_holes


# Score-improvement floor below which we consider the loop converged
# and stop. This protects against floating-point noise nudging the loop
# forever on equivalent programs.
EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Scoring — same shape as the synthesis cell-accuracy metric so the outer
# loop and the inner loop agree about "better".
# ---------------------------------------------------------------------------


def _cell_accuracy(predicted: Any, target: np.ndarray) -> float:
    """Per-cell accuracy of predicted grid against target.

    Mismatched shapes score 0.0. Non-array predictions (e.g. a Python int
    from a type-changing primitive that landed at the root) score 0.0.
    Matches the synthesis scorer's contract.
    """
    if not isinstance(predicted, np.ndarray):
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


# ---------------------------------------------------------------------------
# Edit operations — every edit returns a *new* Program (the input is left
# untouched).  PrimitiveNode's __post_init__ re-runs the type-check, so an
# edit that breaks the type graph raises TypeMismatchError before the
# program ever runs.
# ---------------------------------------------------------------------------


def _clone_program(program: Program) -> Program:
    """Deep-copy a program so edits cannot mutate the caller's AST."""
    return Program(
        root=copy.deepcopy(program.root),
        desired_output=program.desired_output,
    )


def _collect_primitive_nodes(program: Program) -> list[PrimitiveNode]:
    """All PrimitiveNode instances in pre-order — the editable surface."""
    return [n for n in walk_preorder(program) if isinstance(n, PrimitiveNode)]


def swap_primitive(program: Program,
                   target_idx: int,
                   new_primitive: Primitive) -> Program:
    """Replace the primitive at `target_idx` (pre-order index of
    PrimitiveNodes) with `new_primitive`.

    Children are preserved. The substitution must respect the typed
    signature: if the new primitive's input arity or input types don't
    match the existing children, PrimitiveNode's type-check raises
    TypeMismatchError.

    The pre-order index counts PrimitiveNodes only — holes and consts
    are skipped, so callers can address every editable site by a simple
    integer.
    """
    if target_idx < 0:
        raise IndexError(f"swap_primitive: target_idx must be ≥ 0, "
                         f"got {target_idx}")

    new_program = _clone_program(program)
    prim_nodes = _collect_primitive_nodes(new_program)
    if target_idx >= len(prim_nodes):
        raise IndexError(
            f"swap_primitive: target_idx {target_idx} out of range "
            f"({len(prim_nodes)} primitive node(s) in program)"
        )

    target = prim_nodes[target_idx]
    # Re-construct the node so __post_init__ runs the type-check.
    replaced = PrimitiveNode(primitive=new_primitive,
                             children=target.children)

    # Splice `replaced` into the tree in place of `target`.
    if target is new_program.root:
        new_program.root = replaced
        new_program.desired_output = replaced.output_type
    else:
        _splice_in_tree(new_program.root, target, replaced)
    return new_program


def _splice_in_tree(node, old_child, new_child) -> bool:
    """Walk the tree from `node`; replace any direct child equal to
    `old_child` with `new_child`. Returns True on success."""
    if isinstance(node, PrimitiveNode):
        for i, c in enumerate(node.children):
            if c is old_child:
                node.children[i] = new_child
                return True
            if isinstance(c, PrimitiveNode):
                if _splice_in_tree(c, old_child, new_child):
                    return True
    return False


def wrap_program(program: Program,
                 wrapping_primitive: Primitive) -> Program:
    """Wrap the root in `wrapping_primitive`.

    The wrapper must declare exactly one input of type Grid, and its
    output type must be the same as the program's current root output
    (so the wrapped program remains valid against the desired output).

    Multi-input combinators (Seq, Parallel, …) are not eligible
    wrappers here — they require additional siblings, which is a
    different edit op (handled by the synthesis engine, not the
    refinement loop).
    """
    sig = type_signature(wrapping_primitive)
    if len(sig.inputs) != 1:
        raise TypeMismatchError(
            where=f"wrap_program({type(wrapping_primitive).__name__})",
            expected=DslType.GRID,  # symbolic — real error is arity
            actual=DslType.GRID,
        )
    new_program = _clone_program(program)
    expected_input_type = sig.inputs[0][1]
    if new_program.root.output_type != expected_input_type:
        raise TypeMismatchError(
            where=f"wrap_program({type(wrapping_primitive).__name__})",
            expected=expected_input_type,
            actual=new_program.root.output_type,
        )
    wrapped_root = PrimitiveNode(primitive=wrapping_primitive,
                                 children=[new_program.root])
    return Program(root=wrapped_root, desired_output=wrapped_root.output_type)


def mutate_param(program: Program,
                 target_idx: int,
                 param_name: str,
                 new_value: Any) -> Program:
    """Set the scalar parameter `param_name` on the primitive at
    `target_idx` (pre-order index of PrimitiveNodes) to `new_value`.

    The parameter must be declared in the primitive's signature.params.
    The original primitive instance is replaced by a fresh dataclass copy
    with the parameter swapped — leaving the children untouched.
    """
    new_program = _clone_program(program)
    prim_nodes = _collect_primitive_nodes(new_program)
    if target_idx < 0 or target_idx >= len(prim_nodes):
        raise IndexError(
            f"mutate_param: target_idx {target_idx} out of range "
            f"({len(prim_nodes)} primitive node(s) in program)"
        )

    target = prim_nodes[target_idx]
    sig = type_signature(target.primitive)
    declared = {n for n, _ in sig.params}
    if param_name not in declared:
        raise AttributeError(
            f"mutate_param: primitive "
            f"{type(target.primitive).__name__} has no parameter "
            f"{param_name!r}; declared params are {sorted(declared)}"
        )

    # Dataclass-aware copy: replace() only works on dataclass instances.
    # All Primitive subclasses are dataclasses (see primitives.py).
    new_primitive = replace(target.primitive, **{param_name: new_value})
    replaced = PrimitiveNode(primitive=new_primitive,
                             children=target.children)
    if target is new_program.root:
        new_program.root = replaced
        new_program.desired_output = replaced.output_type
    else:
        _splice_in_tree(new_program.root, target, replaced)
    return new_program


# ---------------------------------------------------------------------------
# Error analysis — classify the dominant error type per train pair so the
# candidate generator can bias its enumeration.
# ---------------------------------------------------------------------------


def _error_signature(predicted: Any, target: np.ndarray) -> str:
    """Classify the per-pair error as one of:
        "perfect"      — predicted == target
        "shape"        — shapes differ (rows or cols), needs Rotate/Tile/Crop
        "color"        — same shape, only colors differ at matching cells
        "position"     — same shape + same colorset, but cells shifted
        "mixed"        — same shape, but neither pure-color nor pure-position
        "incompatible" — non-array prediction (e.g. Number leaked to output)
    """
    if not isinstance(predicted, np.ndarray):
        return "incompatible"
    target = np.asarray(target)
    if predicted.shape != target.shape:
        return "shape"
    if np.array_equal(predicted, target):
        return "perfect"
    diff_mask = predicted != target

    # "Position-off": same multiset of values, but cells are misaligned.
    # Approximate this as "same color counts overall" → likely a shift.
    pred_counts = np.bincount(predicted.ravel().astype(int), minlength=10)
    targ_counts = np.bincount(target.ravel().astype(int), minlength=10)
    if np.array_equal(pred_counts, targ_counts):
        return "position"

    # If the changed cells use only colors that appear elsewhere (i.e.
    # the set of colors is the same), call it color-wise off.
    if set(np.unique(predicted).tolist()) == set(np.unique(target).tolist()):
        # Same color palette → guess color or position
        # If most differences fall on cells that ARE non-background in
        # both grids, lean color; otherwise lean position.
        if diff_mask.sum() / target.size <= 0.5:
            return "color"
        return "mixed"

    return "color"


def _aggregate_error_type(program: Program,
                          train_pairs: list[tuple[np.ndarray, np.ndarray]]
                          ) -> str:
    """Most-frequent per-pair error type — the dominant signal."""
    if not train_pairs:
        return "perfect"
    sigs: list[str] = []
    for x, y in train_pairs:
        try:
            pred = evaluate(program, x)
        except (IncompleteProgramError, ValueError, IndexError,
                KeyError, TypeError, AttributeError):
            sigs.append("incompatible")
            continue
        sigs.append(_error_signature(pred, y))
    if all(s == "perfect" for s in sigs):
        return "perfect"
    # Mode — pick the most common non-perfect signal.
    non_perfect = [s for s in sigs if s != "perfect"]
    if not non_perfect:
        return "perfect"
    return max(set(non_perfect), key=non_perfect.count)


# ---------------------------------------------------------------------------
# Candidate generation — biased by error type.
# ---------------------------------------------------------------------------


def _translate_grid() -> list[tuple[int, int]]:
    """All (dy, dx) candidates for Translate parameter mutation.

    Range matches the synthesis enumerator (-3..3)^2 — kept identical so
    refinement and synthesis cover the same parameter grid.
    """
    return [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)]


def _shape_changing_swaps() -> list[Primitive]:
    """Single-Grid-in / Grid-out primitives the refinement loop will try
    when the dominant error is shape-wise off."""
    out: list[Primitive] = [Identity(), Crop()]
    for k in (1, 2, 3):
        out.append(Rotate(k=k))
    for axis in ("H", "V", "D1"):
        out.append(Reflect(axis=axis))
    for rf, cf in ((2, 1), (1, 2), (2, 2)):
        out.append(Tile(rf=rf, cf=cf))
    return out


def _color_changing_swaps() -> list[Primitive]:
    """Candidates for color-wise error: Identity (do-nothing baseline)
    plus a small bank of typical Recolor swaps.

    We deliberately keep this list short — Recolor space is huge, but the
    synthesis engine already enumerates it from the train pairs.
    Refinement is the patch-up layer, not a re-enumeration.
    """
    out: list[Primitive] = [Identity()]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            out.append(Recolor(mapping={a: b}))
    return out


def _generic_swaps() -> list[Primitive]:
    """Last-resort enumeration: every single-Grid-in / Grid-out primitive.
    Used when the error signature is "mixed" or "incompatible"."""
    out: list[Primitive] = [Identity(), Crop()]
    for k in (1, 2, 3):
        out.append(Rotate(k=k))
    for axis in ("H", "V", "D1"):
        out.append(Reflect(axis=axis))
    for direction in ("U", "D", "L", "R"):
        out.append(Gravity(direction=direction))
    for axis in ("H", "V", "BOTH"):
        out.append(Symmetrize(axis=axis))
    for predicate in ("largest", "smallest", "edge_touching", "non_edge"):
        out.append(KeepWhere(predicate=predicate))
    return out


def _candidate_edits(program: Program,
                     error_type: str) -> Iterable[Program]:
    """Yield edited Programs to try this iteration.

    The candidate list is bounded — every iteration tries roughly
    `O(node_count × edit_bank_size)` programs, which for the small
    refinement budgets (<= 4 iters) keeps the loop cheap and finite.

    The candidates are yielded lazily; the caller scores each and tracks
    the best.
    """
    prim_nodes = _collect_primitive_nodes(program)

    # Pick the swap bank based on the dominant error.
    if error_type == "shape":
        swap_bank = _shape_changing_swaps()
        wrap_bank = _shape_changing_swaps()
    elif error_type == "color":
        swap_bank = _color_changing_swaps()
        wrap_bank = [Identity(), Recolor(mapping={})]
    elif error_type == "position":
        # Position-off → only mutate Translate dy/dx, no swap bank to spam.
        swap_bank = []
        wrap_bank = [Translate(dy=dy, dx=dx) for dy, dx in _translate_grid()
                     if not (dy == 0 and dx == 0)]
    else:
        # "mixed" / "incompatible" / unknown → broad sweep
        swap_bank = _generic_swaps()
        wrap_bank = _shape_changing_swaps()

    # 1. mutate_param: for every Translate node, sweep the (dy, dx) grid.
    for idx, node in enumerate(prim_nodes):
        prim = node.primitive
        if isinstance(prim, Translate):
            for dy, dx in _translate_grid():
                if dy == prim.dy and dx == prim.dx:
                    continue
                try:
                    # Two mutates: one for dy, one for dx — but we can do
                    # it in a single replace() via dataclass.
                    new_program = mutate_param(program, idx, "dy", dy)
                    new_program = mutate_param(new_program, idx, "dx", dx)
                    yield new_program
                except (TypeMismatchError, AttributeError, IndexError):
                    continue
        elif isinstance(prim, Rotate):
            for new_k in (1, 2, 3):
                if new_k == prim.k:
                    continue
                try:
                    yield mutate_param(program, idx, "k", new_k)
                except (TypeMismatchError, AttributeError, IndexError):
                    continue
        elif isinstance(prim, Reflect):
            for axis in ("H", "V", "D1"):
                if axis == prim.axis:
                    continue
                try:
                    yield mutate_param(program, idx, "axis", axis)
                except (TypeMismatchError, AttributeError, IndexError):
                    continue
        elif isinstance(prim, Tile):
            for rf, cf in ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)):
                if rf == prim.rf and cf == prim.cf:
                    continue
                try:
                    new_program = mutate_param(program, idx, "rf", rf)
                    new_program = mutate_param(new_program, idx, "cf", cf)
                    yield new_program
                except (TypeMismatchError, AttributeError, IndexError):
                    continue
        elif isinstance(prim, Gravity):
            for direction in ("U", "D", "L", "R"):
                if direction == prim.direction:
                    continue
                try:
                    yield mutate_param(program, idx, "direction", direction)
                except (TypeMismatchError, AttributeError, IndexError):
                    continue

    # 2. swap_primitive: try swapping each Grid-in / Grid-out primitive
    #    node with each candidate from the swap bank, respecting arity.
    for idx, node in enumerate(prim_nodes):
        sig = type_signature(node.primitive)
        # Only swap nodes that have exactly one Grid-typed input slot —
        # otherwise children won't fit.
        if len(sig.inputs) != 1 or sig.inputs[0][1] != Grid:
            continue
        if node.output_type != Grid:
            continue
        for candidate in swap_bank:
            cand_sig = type_signature(candidate)
            if len(cand_sig.inputs) != 1:
                continue
            if cand_sig.inputs[0][1] != Grid:
                continue
            if cand_sig.output != Grid:
                continue
            # Skip exact-same-primitive swaps to avoid no-op work.
            if type(candidate) is type(node.primitive):
                continue
            try:
                yield swap_primitive(program, idx, candidate)
            except (TypeMismatchError, IndexError):
                continue

    # 3. wrap_program: wrap the root with each candidate from the wrap bank.
    if program.root.output_type == Grid:
        for candidate in wrap_bank:
            try:
                yield wrap_program(program, candidate)
            except (TypeMismatchError, IndexError):
                continue


# ---------------------------------------------------------------------------
# Public outer loop.
# ---------------------------------------------------------------------------


def refine(program: Program,
           train_pairs: list[tuple[np.ndarray, np.ndarray]],
           max_iters: int = 4) -> Program:
    """Iteratively edit `program` to better fit `train_pairs`.

    Behaviour contract:
      - If the input program is already perfect (score 1.0), it is
        returned unchanged.
      - Otherwise, up to `max_iters` rounds of structural edits are
        proposed, each scored against the train pairs.
      - At each round, the highest-scoring candidate strictly above the
        current best (by more than EPSILON) is accepted as the new
        current program; otherwise the loop terminates early.
      - The function NEVER returns a program that scores lower than the
        input. If no improvement is ever found, the input is returned
        unchanged.

    Args:
        program: the typed Program to refine
        train_pairs: list of (input_grid, output_grid) demonstrations
        max_iters: maximum number of refinement rounds (default 4)

    Returns:
        A refined Program (possibly the original if no improvement found).
    """
    if max_iters < 0:
        raise ValueError(f"refine: max_iters must be ≥ 0, got {max_iters}")

    if not train_pairs:
        return program

    # Refinement assumes the program is structurally complete enough to
    # be evaluable on a Grid input — i.e. only Grid-typed holes remain
    # (the interpreter binds them to the initial input). Programs with
    # non-Grid holes are returned unchanged: refinement is a patch-up
    # layer, not a synthesis engine.
    open_holes = find_holes(program)
    if any(h.expected_type != Grid for h in open_holes):
        return program

    best_program = program
    best_score = _train_score(best_program, train_pairs)

    if best_score >= 1.0 - EPSILON:
        # Already perfect — nothing to refine.
        return best_program

    for _ in range(max_iters):
        error_type = _aggregate_error_type(best_program, train_pairs)
        if error_type == "perfect":
            break

        round_best: Program | None = None
        round_best_score = best_score

        for candidate in _candidate_edits(best_program, error_type):
            score = _train_score(candidate, train_pairs)
            if score > round_best_score + EPSILON:
                round_best = candidate
                round_best_score = score
                # Eager early-out: once we hit a perfect score there's
                # no point continuing this round.
                if score >= 1.0 - EPSILON:
                    break

        if round_best is None:
            # No edit improved the score this round — converged.
            break

        best_program = round_best
        best_score = round_best_score

        if best_score >= 1.0 - EPSILON:
            break

    return best_program


__all__ = [
    "refine",
    "swap_primitive",
    "wrap_program",
    "mutate_param",
    "EPSILON",
]
