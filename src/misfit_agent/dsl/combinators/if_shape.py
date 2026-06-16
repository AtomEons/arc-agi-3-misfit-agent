"""IfShape(target_shape, f_then, g_else) — shape-conditional branch combinator.

Role
----
IfShape is a *Primitive* in the typed AST sense: it declares a typed
signature and an `apply()` method like any other primitive. Its purpose
is to give the synthesis engine a concrete one-grid-in, one-grid-out
shape whose behaviour BRANCHES on whether any perceived object in the
input grid has a bounding-box shape of exactly (h, w):

    IfShape(target_shape=(2,2), then=<Rotate(k=1, g)>, else=<Identity(g)>)

If any object in the input grid has bbox dimensions equal to
target_shape, evaluate the `then_program` against the input grid;
otherwise evaluate the `else_program`. This is the canonical Spelke
OBJECTNESS+GEOMETRY-based branch — a single bit of perceptual
information (does an h×w object exist?) selects between two whole
sub-programs.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("target_shape", tuple),
               ("then_program",  object),
               ("else_program",  object))

Construction validation
-----------------------
Both branch programs (Program objects, or PrimitiveNodes used directly)
must produce Grid. Anything else raises TypeMismatchError at construction
time — synthesis never enumerates an ill-typed IfShape.

Semantics
---------
IfShape.apply(grid) runs perceive_grid(grid) once, walks the perceived
objects, and tests whether any object's bbox dimensions
`(r1 - r0 + 1, c1 - c0 + 1)` equal self.target_shape. The whole input
grid is then dispatched into either the then-branch or the else-branch.
The branch program is evaluated with the SAME input grid (we don't
thread state between branches; the branch is a true if/else, not a
then-and-also).

The combinator's child programs are META-parameters — they are
configured at construction time and travel with the combinator instance.
They do NOT consume typed AST child slots (no PrimitiveNode children);
the only AST-visible input is the outer grid.

MDL accounting
--------------
    catalog_bits + 1 (meta-shape) + shape_bits
                 + then_program_bits + else_program_bits

The shape_bits charge reflects that (h, w) with each dimension in 1..30
costs 2 * ceil(log2(30)) = 10 bits. The two child programs each
contribute their full MDL cost — branches with simpler programs win.

Hash / memoization
------------------
The combinator's `to_string()` and (via wrapping PrimitiveNode) its
`hash_key()` include the target shape literal and both child programs'
structure. Two IfShapes with different target shapes, different
then-branches, or different else-branches MUST produce distinct hash
keys — otherwise the synthesis engine would re-use cached results from
the wrong branch.

Tier-1 disclosure
-----------------
IfShape introduces no learned parameters, no pretrained weights, no LLM
calls, and no third-party imports. The shape-presence predicate is a
single pass over the perceptor's object list (already part of the
substrate). Both branch programs are typed AST nodes from the same
hand-authored grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

import numpy as np

from ...perceptor import perceive_grid
from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Signature, TypeMismatchError, DslType
from ..ast import Program, PrimitiveNode
from ..interpreter import evaluate


# A branch program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted.
BranchLike = Union[Program, PrimitiveNode]


@dataclass
class IfShape(Primitive):
    """IfShape(target_shape, then_program, else_program): shape-conditional branch.

    The two branch programs are META-parameters — they are configured at
    construction time and travel with the combinator instance. They do
    NOT consume typed AST child slots (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        target_shape: a (height, width) tuple to look for in the
            perceived objects' bounding boxes.
        then_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. Evaluated against the input grid when any
            perceived object has bbox shape equal to target_shape.
        else_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. Evaluated against the input grid when no
            perceived object matches the target shape.

    Raises:
        TypeMismatchError: at construction time, if either branch
            program's output type is not Grid, or if either branch
            program is missing.
    """

    target_shape: tuple = (1, 1)
    then_program: BranchLike = None
    else_program: BranchLike = None

    def __post_init__(self):
        # Normalize target_shape to a 2-tuple of ints — accept lists for
        # convenience but always store the canonical form.
        if not isinstance(self.target_shape, tuple):
            try:
                self.target_shape = tuple(self.target_shape)
            except TypeError:
                raise TypeMismatchError(
                    where="IfShape.target_shape",
                    expected=DslType.GRID,
                    actual=DslType.GRID,
                )
        if len(self.target_shape) != 2:
            raise TypeMismatchError(
                where="IfShape.target_shape",
                expected=DslType.GRID,
                actual=DslType.GRID,
            )

        # Validate both branch programs' shape and type at construction
        # time. This is the central type-safety contract: the synthesis
        # engine must never get an IfShape whose branches can return
        # Number, Color, ObjSet, etc.
        if self.then_program is None:
            raise TypeMismatchError(
                where="IfShape.then_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )
        if self.else_program is None:
            raise TypeMismatchError(
                where="IfShape.else_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )

        then_out = self._branch_output_type(self.then_program, "then_program")
        if then_out != DslType.GRID:
            raise TypeMismatchError(
                where="IfShape.then_program",
                expected=DslType.GRID,
                actual=then_out,
            )

        else_out = self._branch_output_type(self.else_program, "else_program")
        if else_out != DslType.GRID:
            raise TypeMismatchError(
                where="IfShape.else_program",
                expected=DslType.GRID,
                actual=else_out,
            )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _branch_output_type(branch: BranchLike, slot: str) -> DslType:
        """Return a branch program's output DslType, regardless of wrap."""
        if isinstance(branch, Program):
            return branch.output_type()
        if isinstance(branch, PrimitiveNode):
            return branch.output_type
        # Anything else is an authoring error — surface as a type error
        # so the synthesis engine's normal type-mismatch handling kicks
        # in instead of a generic TypeError.
        raise TypeMismatchError(
            where=f"IfShape.{slot} ({type(branch).__name__})",
            expected=DslType.GRID,
            actual=DslType.GRID,
        )

    @staticmethod
    def _branch_as_program(branch: BranchLike) -> Program:
        """Coerce a branch into a Program for the interpreter."""
        if isinstance(branch, Program):
            return branch
        # PrimitiveNode → wrap in a Program for the interpreter.
        return Program(root=branch, desired_output=DslType.GRID)

    @staticmethod
    def _branch_hash_key(branch: BranchLike) -> str:
        """Stable hash key fragment for a branch program."""
        if isinstance(branch, Program):
            return branch.hash_key()
        if isinstance(branch, PrimitiveNode):
            return branch.hash_key()
        return repr(branch)

    @staticmethod
    def _branch_to_string(branch: BranchLike) -> str:
        """Human-readable string for a branch program."""
        if isinstance(branch, Program):
            return branch.to_string()
        if isinstance(branch, PrimitiveNode):
            return branch.to_string()
        return repr(branch)

    # -----------------------------------------------------------------
    # Typed signature
    # -----------------------------------------------------------------

    def signature_typed(self) -> Signature:
        # Branch programs are META-parameters — declared in params (so
        # the type-checker knows they're not typed AST slots) but
        # enforced at construction time via _branch_output_type().
        # The target_shape is a plain tuple param.
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(
                ("target_shape", tuple),
                ("then_program", object),
                ("else_program", object),
            ),
        )

    # -----------------------------------------------------------------
    # apply: perceive, predicate, dispatch
    # -----------------------------------------------------------------

    def apply(self, grid):
        grid_np = np.asarray(grid, dtype=np.int32)
        target_h, target_w = int(self.target_shape[0]), int(self.target_shape[1])

        objs = perceive_grid(grid_np).objects
        matched = False
        for obj in objs:
            r0, c0, r1, c1 = obj.bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if (h, w) == (target_h, target_w):
                matched = True
                break

        if matched:
            return evaluate(self._branch_as_program(self.then_program), grid_np)
        return evaluate(self._branch_as_program(self.else_program), grid_np)

    # -----------------------------------------------------------------
    # AST identity, MDL, display
    # -----------------------------------------------------------------

    def to_string(self) -> str:
        then_s = self._branch_to_string(self.then_program)
        else_s = self._branch_to_string(self.else_program)
        h, w = int(self.target_shape[0]), int(self.target_shape[1])
        return f"IfShape(shape=({h},{w}), then={then_s}, else={else_s})"

    def mdl_bits(self) -> float:
        # Base cost: log2(|primitive catalog| + combinator) for picking
        # IfShape as the composition shape. Plus 1 bit for the meta-shape
        # choice (IfShape vs. another single-input Grid→Grid wrapper),
        # plus shape_bits for the (h, w) target choice
        # (each dimension in 1..30 ⇒ ceil(log2(30)) = 5 bits each,
        # total = 10 bits), plus the MDL bits of both branch programs.
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        shape_bits = 10.0
        then_cost = self._branch_mdl_bits(self.then_program)
        else_cost = self._branch_mdl_bits(self.else_program)
        return base + 1.0 + shape_bits + then_cost + else_cost

    @staticmethod
    def _branch_mdl_bits(branch: BranchLike) -> float:
        """Sum the MDL bits contributed by a branch program."""
        if isinstance(branch, Program):
            return _sum_node_mdl(branch.root)
        if isinstance(branch, PrimitiveNode):
            return _sum_node_mdl(branch)
        return 0.0


def _sum_node_mdl(node) -> float:
    """Sum MDL bits over a node subtree (PrimitiveNode + children)."""
    if isinstance(node, PrimitiveNode):
        total = float(node.primitive.mdl_bits())
        for child in node.children:
            total += _sum_node_mdl(child)
        return total
    return 0.0
