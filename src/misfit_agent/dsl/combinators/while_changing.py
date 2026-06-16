"""WhileChanging(f, max_iter) — fixed-point iteration combinator.

Role
----
WhileChanging lifts a Grid→Grid child program f into a Grid→Grid combinator
that repeatedly applies f until the output stops changing (fixed point) or
until a hard `max_iter` cap is reached. This is the canonical combinator
for "settle" transformations where the right answer is "keep doing this
until nothing more happens":

    apply(grid):
        current = grid.copy()
        for _ in range(max_iter):
            next_grid = evaluate(child, current)
            if next_grid == current:    # fixed point reached
                return current
            current = next_grid
        return current                  # iteration cap hit

The canonical use case is Gravity(direction='D'): the first application
moves all non-background cells as far down as they can go in one pass.
Already-settled grids return after one iteration with no change. A
multi-cell column may require several passes if the underlying primitive
implementation only moves one row at a time — WhileChanging is the
combinator that makes "settle" first-class without baking iteration into
every primitive.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("child_program", object), ("max_iter", int))

Construction validation
-----------------------
The child program (or a PrimitiveNode used directly) must produce Grid.
Anything else raises TypeMismatchError at construction time — synthesis
never enumerates an ill-typed WhileChanging. A non-positive `max_iter` is
also rejected: an iteration cap of 0 or below would make the combinator
a noop and pollute the search space.

Tier-1 disclosure
-----------------
WhileChanging introduces no learned parameters, no pretrained weights,
no LLM calls, and no third-party imports beyond numpy (already a
substrate dependency). The fixed-point test is a structural equality
check (`np.array_equal`); there is no learned convergence criterion.
The child program f is a typed AST node from the same hand-authored
grammar, and its MDL cost is included in the combinator's own
`mdl_bits()` so the synthesis prior correctly penalizes deep iterated
programs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

import numpy as np

from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Signature, TypeMismatchError, DslType
from ..ast import Program, PrimitiveNode
from ..interpreter import evaluate


# A child program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted — mirrors the ForEachObject contract.
ChildLike = Union[Program, PrimitiveNode]


# Default iteration cap. 16 is enough for any reasonable ARC settle
# (the largest ARC grid is 30x30, so a "drop" of a column from top to
# bottom needs at most 29 passes — but every realistic primitive moves
# strictly more than one cell per pass, so 16 is generous). The cap
# exists to prevent runaway loops if a child program is genuinely
# non-convergent (e.g. a 180° rotation that flips back-and-forth).
DEFAULT_MAX_ITER = 16


@dataclass
class WhileChanging(Primitive):
    """WhileChanging(child_program, max_iter): iterate child to fixed point.

    The child program is a META-parameter — it is configured at
    construction time and travels with the combinator instance. It does
    NOT consume a typed AST child slot (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        child_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. The program is applied repeatedly to the
            running grid until either the output stops changing
            (fixed point) or `max_iter` iterations have completed.
        max_iter: hard cap on iteration count. Must be a positive int.
            Defaults to 16, which is enough for any plausible ARC
            settle without inviting runaway loops.

    Raises:
        TypeMismatchError: at construction time, if the child program's
            output type is not Grid, or if the child is missing.
        ValueError: at construction time, if `max_iter` is not a
            positive int.
    """

    child_program: ChildLike = None
    max_iter: int = DEFAULT_MAX_ITER

    def __post_init__(self):
        # Validate the child program shape and type at construction time.
        # This is the central type-safety contract: the synthesis engine
        # must never get a WhileChanging whose child can return Number,
        # Color, ObjSet, etc. — only Grid→Grid children compose correctly.
        if self.child_program is None:
            raise TypeMismatchError(
                where="WhileChanging.child_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )

        child_out_type = self._child_output_type()
        if child_out_type != DslType.GRID:
            raise TypeMismatchError(
                where="WhileChanging.child_program",
                expected=DslType.GRID,
                actual=child_out_type,
            )

        # Validate max_iter: must be a positive int. A non-positive cap
        # would make the combinator a no-op (zero iterations) or worse,
        # raise a confusing error deep inside apply().
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool):
            raise ValueError(
                f"WhileChanging.max_iter must be a positive int, "
                f"got {type(self.max_iter).__name__}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"WhileChanging.max_iter must be >= 1, got {self.max_iter}"
            )

    # -----------------------------------------------------------------
    # Internal helpers (mirrors ForEachObject for stylistic consistency)
    # -----------------------------------------------------------------

    def _child_output_type(self) -> DslType:
        """Return the child program's output DslType, regardless of wrap."""
        if isinstance(self.child_program, Program):
            return self.child_program.output_type()
        if isinstance(self.child_program, PrimitiveNode):
            return self.child_program.output_type
        # Anything else is an authoring error.
        raise TypeMismatchError(
            where=f"WhileChanging.child_program ({type(self.child_program).__name__})",
            expected=DslType.GRID,
            actual=DslType.GRID,
        )

    def _child_as_program(self) -> Program:
        """Coerce the child into a Program for evaluation."""
        if isinstance(self.child_program, Program):
            return self.child_program
        # PrimitiveNode → wrap in a Program for the interpreter.
        return Program(root=self.child_program, desired_output=DslType.GRID)

    def _child_hash_key(self) -> str:
        """Stable hash key for the child program (for memoization)."""
        if isinstance(self.child_program, Program):
            return self.child_program.hash_key()
        if isinstance(self.child_program, PrimitiveNode):
            return self.child_program.hash_key()
        return repr(self.child_program)

    # -----------------------------------------------------------------
    # Typed signature
    # -----------------------------------------------------------------

    def signature_typed(self) -> Signature:
        # The child program is a META-parameter — declared in `params`
        # so the type-checker treats it as scalar metadata (not a typed
        # AST slot), but enforced at construction time. max_iter is also
        # a META-parameter; it's part of the combinator instance and
        # contributes to both the MDL cost and the hash key.
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("child_program", object), ("max_iter", int)),
        )

    # -----------------------------------------------------------------
    # apply: iterate child to fixed point or max_iter
    # -----------------------------------------------------------------

    def apply(self, grid):
        """Run the child program repeatedly until convergence or cap.

        The fixed-point test is `np.array_equal(next_grid, current)`:
          - structurally exact (shape AND values)
          - bool-safe (returns False on shape mismatch, not raising)

        If the child returns a shape-different grid on the first pass,
        the comparison returns False and we advance to that new shape.
        Subsequent passes compare at the new shape. This handles child
        programs that change shape (e.g. Crop) correctly: they reach a
        fixed point once the cropped grid stops shrinking.
        """
        # Defensive copy: never mutate the caller's input.
        current = np.asarray(grid).copy()
        child_program = self._child_as_program()

        for _ in range(self.max_iter):
            next_grid = evaluate(child_program, current)
            next_grid = np.asarray(next_grid)
            if np.array_equal(next_grid, current):
                # Fixed point reached. Return the current grid (which is
                # equal to next_grid by definition); we return `current`
                # rather than `next_grid` so the caller always sees the
                # same object identity it would have seen on a no-op
                # convergence path.
                return current
            current = next_grid

        # Iteration cap hit without convergence. Return whatever the
        # last iteration produced. The synthesis engine's MDL prior
        # will deprioritize non-convergent child programs by way of
        # their poor demonstration-fit, not by a runtime error here —
        # the contract is "best effort within max_iter", not "guaranteed
        # convergence". Raising would mask otherwise-useful programs
        # that happen to need >max_iter for an outlier task.
        return current

    # -----------------------------------------------------------------
    # AST identity, MDL, display
    # -----------------------------------------------------------------

    def to_string(self) -> str:
        if isinstance(self.child_program, Program):
            inner = self.child_program.to_string()
        elif isinstance(self.child_program, PrimitiveNode):
            inner = self.child_program.to_string()
        else:
            inner = repr(self.child_program)
        return f"WhileChanging({inner}, max_iter={self.max_iter})"

    def mdl_bits(self) -> float:
        """Total bits to encode this combinator + its child + max_iter.

        Components:
          - log2(|catalog|+1) bits  — picking WhileChanging as the
            composition shape from the catalog
          - 1 bit                   — meta-shape choice (vs nested-prim)
          - sum(child mdl bits)     — encoding the child program
          - log2(max_iter)          — encoding the iteration cap (a real
            choice in the grammar; bigger caps cost more bits, so the
            prior naturally prefers shorter settle horizons when fit
            permits)
        """
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        child_cost = self._child_mdl_bits()
        # max_iter cap cost: log2 of the cap value, clamped to >= 1 bit
        # so a max_iter=1 still costs something to commit to.
        iter_cost = max(1.0, math.log2(max(self.max_iter, 2)))
        return base + 1.0 + child_cost + iter_cost

    def _child_mdl_bits(self) -> float:
        """Sum the MDL bits contributed by the child program."""
        if isinstance(self.child_program, Program):
            return _sum_node_mdl(self.child_program.root)
        if isinstance(self.child_program, PrimitiveNode):
            return _sum_node_mdl(self.child_program)
        return 0.0


def _sum_node_mdl(node) -> float:
    """Sum MDL bits over a node subtree (PrimitiveNode + children)."""
    if isinstance(node, PrimitiveNode):
        total = float(node.primitive.mdl_bits())
        for child in node.children:
            total += _sum_node_mdl(child)
        return total
    return 0.0
