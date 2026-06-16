"""ForEachObject(f) — per-object map combinator.

Role
----
ForEachObject lifts a Grid→Grid child program f into a Grid→Grid combinator
that:

    1. perceives all objects in the input grid (via the perceptor),
    2. extracts each object as its own small "tiny" grid,
    3. evaluates the child program f on each tiny grid,
    4. stamps each per-object result back into the corresponding bbox in
       a copy of the input grid.

This is the canonical Spelke OBJECTNESS lifting: a transformation that is
local to each object (rotate-each, recolor-each, reflect-each, etc.) is
expressed as ForEachObject(<local-transform>) rather than as a global
grid transformation. The synthesis engine treats the child program f as
a META-parameter — f is configured at ForEachObject construction time
and is NOT a regular AST child slot. The combinator's typed signature
declares a single Grid input ("g") and a Grid output; the child program
travels with the combinator instance, contributing to its hash and MDL
bit cost.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("child_program", Program),)

Construction validation
-----------------------
The child program (or a PrimitiveNode used directly) must produce Grid.
Anything else raises TypeMismatchError at construction time — synthesis
never enumerates an ill-typed ForEachObject.

Semantics
---------
ForEachObject.apply(grid) does the perceive → extract → evaluate-child →
stamp-back loop. If perception finds zero objects, the input is returned
unmodified (preserves Identity behaviour for empty scenes). Stamping
uses the perceived bbox; if the child returns a grid whose shape does
not match the bbox, the overlap region is stamped (a defensive choice
that prevents shape-mismatch from blowing up an otherwise-valid program;
the synthesis engine's MDL prior will deprioritize child programs that
do not preserve shape).

Tier-1 disclosure
-----------------
ForEachObject introduces no learned parameters, no pretrained weights,
no LLM calls, and no third-party imports. The perceptor it relies on is
a deterministic flood-fill segmenter under the OBJECTNESS prior. The
child program f is a typed AST node from the same hand-authored grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Union

import numpy as np

from ...perceptor import perceive_grid
from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Signature, TypeMismatchError, DslType
from ..ast import Program, PrimitiveNode
from ..interpreter import evaluate


# A child program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted.
ChildLike = Union[Program, PrimitiveNode]


@dataclass
class ForEachObject(Primitive):
    """ForEachObject(child_program): apply a Grid→Grid child to each object.

    The child program is a META-parameter — it is configured at
    construction time and travels with the combinator instance. It does
    NOT consume a typed AST child slot (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        child_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. The program is applied to each perceived
            object's tiny grid; its result is stamped back into the
            object's bbox in a copy of the input grid.

    Raises:
        TypeMismatchError: at construction time, if the child program's
            output type is not Grid.
    """

    child_program: ChildLike = None

    def __post_init__(self):
        # Validate the child program shape and type at construction time.
        # This is the central type-safety contract: the synthesis engine
        # must never get a ForEachObject whose child can return Number,
        # Color, ObjSet, etc.
        if self.child_program is None:
            raise TypeMismatchError(
                where="ForEachObject.child_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )

        child_out_type = self._child_output_type()
        if child_out_type != DslType.GRID:
            raise TypeMismatchError(
                where="ForEachObject.child_program",
                expected=DslType.GRID,
                actual=child_out_type,
            )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _child_output_type(self) -> DslType:
        """Return the child program's output DslType, regardless of wrap."""
        if isinstance(self.child_program, Program):
            return self.child_program.output_type()
        if isinstance(self.child_program, PrimitiveNode):
            return self.child_program.output_type
        # Anything else is an authoring error.
        raise TypeMismatchError(
            where=f"ForEachObject.child_program ({type(self.child_program).__name__})",
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
        # The child program is a META-parameter — it is declared in
        # params (so the type-checker knows it's not a typed AST slot)
        # but enforced at construction time via _child_output_type().
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(("child_program", object),),
        )

    # -----------------------------------------------------------------
    # apply: perceive → extract → evaluate child → stamp
    # -----------------------------------------------------------------

    def apply(self, grid):
        grid = np.asarray(grid, dtype=np.int32)
        scene = perceive_grid(grid)
        objs = scene.objects
        if not objs:
            # No objects perceived — preserve Identity semantics.
            return grid.copy()

        out = grid.copy()
        child_program = self._child_as_program()

        for obj in objs:
            r0, c0, r1, c1 = obj.bbox
            # Slice out the object's tiny grid from the input. We use the
            # ORIGINAL input grid (not the in-progress output) so that
            # per-object transformations are independent — the child
            # program sees the object in its perceived form, regardless
            # of stamping order.
            tiny = grid[r0:r1 + 1, c0:c1 + 1].copy()

            modified = evaluate(child_program, tiny)
            modified = np.asarray(modified, dtype=np.int32)

            # Stamp the modified tiny back into the bbox. If the child
            # returned a different shape, stamp the overlapping region
            # only — this is a defensive choice; the synthesis engine's
            # MDL prior will deprioritize shape-changing children inside
            # ForEachObject.
            mh, mw = modified.shape
            bh, bw = (r1 - r0 + 1), (c1 - c0 + 1)
            h = min(mh, bh)
            w = min(mw, bw)
            out[r0:r0 + h, c0:c0 + w] = modified[:h, :w]

        return out

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
        return f"ForEachObject({inner})"

    def mdl_bits(self) -> float:
        # Base cost: log2(|primitive catalog| + combinator) for picking
        # ForEachObject as the composition shape. Plus the cost of the
        # child program itself — a longer child means a longer encoding,
        # so the prior already encodes "prefer the simplest per-object
        # transform that explains the demonstration." 1 extra bit pays
        # for the meta-shape choice.
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        child_cost = self._child_mdl_bits()
        return base + 1.0 + child_cost

    def _child_mdl_bits(self) -> float:
        """Sum the MDL bits contributed by the child program."""
        if isinstance(self.child_program, Program):
            # Walk the program and sum every primitive's MDL bits.
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
