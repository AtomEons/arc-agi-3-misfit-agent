"""IfColor(c, f_then, g_else) — color-conditional branch combinator.

Role
----
IfColor is a *Primitive* in the typed AST sense: it declares a typed
signature and an `apply()` method like any other primitive. Its purpose
is to give the synthesis engine a concrete one-grid-in, one-grid-out
shape whose behaviour BRANCHES on whether a specific color is present in
the input grid:

    IfColor(c=2, then=<Identity(g)>, else=<Rotate(k=1, g)>)

If the input grid contains color c, evaluate the `then_program` against
the input grid; otherwise evaluate the `else_program`. This is the
canonical Spelke FORM/PROPERTY-based branch — a single bit of perceptual
information (does color c appear?) selects between two whole
sub-programs.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("color", int),
               ("then_program", Program),
               ("else_program", Program))

Construction validation
-----------------------
Both branch programs (Program objects, or PrimitiveNodes used directly)
must produce Grid. Anything else raises TypeMismatchError at construction
time — synthesis never enumerates an ill-typed IfColor. The color must
be in the ARC palette range 0..9; the type-checker doesn't enforce that
but `mdl_bits()` budgets for exactly 4 bits of color choice.

Semantics
---------
IfColor.apply(grid) evaluates the color predicate
`(grid == self.color).any()` once and dispatches the entire input grid
into either the then-branch or the else-branch. The branch program is
evaluated with the SAME input grid (we don't thread state between
branches; the branch is a true if/else, not a then-and-also).

The combinator's child programs are META-parameters — they are
configured at construction time and travel with the combinator instance.
They do NOT consume typed AST child slots (no PrimitiveNode children);
the only AST-visible input is the outer grid.

MDL accounting
--------------
    catalog_bits + 1 (meta-shape) + 4 (color choice in 0..9)
                 + then_program_bits + else_program_bits

The 4-bit color charge reflects that a single ARC color picks one of 10
palette entries (ceil(log2(10)) = 4). The two child programs each
contribute their full MDL cost — branches with simpler programs win.

Hash / memoization
------------------
The combinator's `to_string()` and (via wrapping PrimitiveNode) its
`hash_key()` include the color literal and both child programs'
structure. Two IfColors with different colors, different then-branches,
or different else-branches MUST produce distinct hash keys — otherwise
the synthesis engine would re-use cached results from the wrong branch.

Tier-1 disclosure
-----------------
IfColor introduces no learned parameters, no pretrained weights, no LLM
calls, and no third-party imports. The color-presence predicate is a
single numpy `.any()` over an integer-equality mask. Both branch
programs are typed AST nodes from the same hand-authored grammar.
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


# A branch program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted.
BranchLike = Union[Program, PrimitiveNode]


@dataclass
class IfColor(Primitive):
    """IfColor(color, then_program, else_program): color-conditional branch.

    The two branch programs are META-parameters — they are configured at
    construction time and travel with the combinator instance. They do
    NOT consume typed AST child slots (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        color: an ARC palette color (int 0..9) to test for presence in
            the input grid.
        then_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. Evaluated against the input grid when the
            color is present.
        else_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. Evaluated against the input grid when the
            color is absent.

    Raises:
        TypeMismatchError: at construction time, if either branch
            program's output type is not Grid, or if either branch
            program is missing.
    """

    color: int = 0
    then_program: BranchLike = None
    else_program: BranchLike = None

    def __post_init__(self):
        # Validate both branch programs' shape and type at construction
        # time. This is the central type-safety contract: the synthesis
        # engine must never get an IfColor whose branches can return
        # Number, Color, ObjSet, etc.
        if self.then_program is None:
            raise TypeMismatchError(
                where="IfColor.then_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )
        if self.else_program is None:
            raise TypeMismatchError(
                where="IfColor.else_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )

        then_out = self._branch_output_type(self.then_program, "then_program")
        if then_out != DslType.GRID:
            raise TypeMismatchError(
                where="IfColor.then_program",
                expected=DslType.GRID,
                actual=then_out,
            )

        else_out = self._branch_output_type(self.else_program, "else_program")
        if else_out != DslType.GRID:
            raise TypeMismatchError(
                where="IfColor.else_program",
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
            where=f"IfColor.{slot} ({type(branch).__name__})",
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
        # The color is a plain int param.
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(
                ("color", int),
                ("then_program", object),
                ("else_program", object),
            ),
        )

    # -----------------------------------------------------------------
    # apply: predicate then dispatch
    # -----------------------------------------------------------------

    def apply(self, grid):
        grid_np = np.asarray(grid, dtype=np.int32)
        if (grid_np == self.color).any():
            return evaluate(self._branch_as_program(self.then_program), grid_np)
        return evaluate(self._branch_as_program(self.else_program), grid_np)

    # -----------------------------------------------------------------
    # AST identity, MDL, display
    # -----------------------------------------------------------------

    def to_string(self) -> str:
        then_s = self._branch_to_string(self.then_program)
        else_s = self._branch_to_string(self.else_program)
        return f"IfColor(c={self.color}, then={then_s}, else={else_s})"

    def mdl_bits(self) -> float:
        # Base cost: log2(|primitive catalog| + combinator) for picking
        # IfColor as the composition shape. Plus 1 bit for the meta-shape
        # choice (IfColor vs. another single-input Grid→Grid wrapper),
        # plus 4 bits for the color choice (ceil(log2(10)) = 4 for ARC
        # palette 0..9), plus the MDL bits of both branch programs.
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        color_bits = 4.0
        then_cost = self._branch_mdl_bits(self.then_program)
        else_cost = self._branch_mdl_bits(self.else_program)
        return base + 1.0 + color_bits + then_cost + else_cost

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
