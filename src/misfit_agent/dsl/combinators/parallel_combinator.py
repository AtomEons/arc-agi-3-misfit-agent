"""Parallel(f, g, merge) — same-input two-branch combinator with cell-wise merge.

Role
----
Parallel is a *Primitive* in the typed AST sense: it declares a typed
signature and an `apply()` method like any other primitive. Its purpose
is to give the synthesis engine a concrete one-grid-in, one-grid-out
shape whose behaviour evaluates TWO whole subprograms (f and g) against
the *same* input grid, and merges their two Grid outputs cell-wise:

    Parallel(f_program=<Identity(g)>,
             g_program=<Rotate(k=2, g)>,
             merge="or")

Both child programs see the same input. The two output grids are merged
under one of three deterministic rules:

  - "or"    : f wins on its non-background cells, else g shows through.
              Implementation: np.where(out_f != bg, out_f, out_g) where
              bg is background_color(input_grid). This is the canonical
              Spelke OBJECTNESS overlay — f's figure on g's ground.
  - "max"   : element-wise integer maximum. Color value 9 beats 0.
              Useful when the synthesis engine has learned that "darker"
              or higher-index palette entries should dominate.
  - "first" : return f's output verbatim. Used as a degenerate-merge
              baseline so the MDL/search engine can probe whether the
              second branch contributes anything.

If the two outputs have different shapes, fall back to f's output. This
keeps the combinator total — synthesis never crashes on a shape mismatch
mid-search; the worse-merged candidate just gets the higher MDL cost.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("f", object),
               ("g", object),
               ("merge", str))

The two branch programs are META-parameters — declared in params (so
the type-checker knows they're not typed AST slots) but enforced at
construction time via _branch_output_type(). The merge tag is a plain
str param drawn from a 3-element vocabulary.

Construction validation
-----------------------
Both branch programs (Program objects or PrimitiveNodes used directly)
must produce Grid. Anything else raises TypeMismatchError at construction
time — synthesis never enumerates an ill-typed Parallel. The merge tag
must be one of {"or", "max", "first"}; an unknown tag raises ValueError.

MDL accounting
--------------
    catalog_bits + 1 (meta-shape) + log2(3) (merge tag in 3-rule set)
                 + f_program_bits + g_program_bits

The log2(3) ≈ 1.585 bit charge reflects that picking 1 of 3 merge rules
is a real choice. The two child programs each contribute their full MDL
cost — Parallel always costs *strictly more* than a bare Primitive, so
the synthesis engine has to be paid by the data to pick it.

Hash / memoization
------------------
The combinator's `to_string()` and (via wrapping PrimitiveNode) its
`hash_key()` include the merge tag literal and both child programs'
structure. Two Parallels with different merges, different f-branches,
or different g-branches MUST produce distinct hash keys — otherwise the
memoization tables would re-use cached results from the wrong branch.

Tier-1 disclosure
-----------------
Parallel introduces no learned parameters, no pretrained weights, no LLM
calls, and no third-party imports. The merge rules are three numpy
expressions over integer arrays. Both branch programs are typed AST
nodes from the same hand-authored grammar. Pure search and structural
operations only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

import numpy as np

from ...perceptor import _background_color
from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Signature, TypeMismatchError, DslType
from ..ast import Program, PrimitiveNode
from ..interpreter import evaluate


# A branch program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted, matching the IfColor convention.
BranchLike = Union[Program, PrimitiveNode]


# The fixed 3-rule merge vocabulary. Any other tag raises at construction
# time. Frozen here so the MDL bit-budget log2(len(MERGE_RULES)) stays
# truthful as the catalog evolves.
MERGE_RULES = ("or", "max", "first")


@dataclass
class Parallel(Primitive):
    """Parallel(f, g, merge): two Grid-producing subprograms on one input.

    The two branch programs are META-parameters — they are configured at
    construction time and travel with the combinator instance. They do
    NOT consume typed AST child slots (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        f: a typed Program (or PrimitiveNode) whose output type is Grid.
            Evaluated against the input grid; its output is one of the
            two grids to merge.
        g: a typed Program (or PrimitiveNode) whose output type is Grid.
            Evaluated against the same input grid; its output is the
            other grid to merge.
        merge: one of {"or", "max", "first"}; selects the cell-wise
            merge rule.

    Raises:
        TypeMismatchError: at construction time, if either branch
            program's output type is not Grid, or if either branch
            program is missing.
        ValueError: at construction time, if `merge` is not in
            {"or", "max", "first"}.
    """

    f: BranchLike = None
    g: BranchLike = None
    merge: str = "or"

    def __post_init__(self):
        # Validate the merge tag against the fixed 3-rule vocabulary
        # before doing anything else; a bad tag is the cheapest reject.
        if self.merge not in MERGE_RULES:
            raise ValueError(
                f"Parallel.merge must be one of {MERGE_RULES}; got "
                f"{self.merge!r}"
            )

        # Validate both branch programs' shape and type at construction
        # time. This is the central type-safety contract: the synthesis
        # engine must never get a Parallel whose branches can return
        # Number, Color, ObjSet, etc.
        if self.f is None:
            raise TypeMismatchError(
                where="Parallel.f",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder; message conveys "missing"
            )
        if self.g is None:
            raise TypeMismatchError(
                where="Parallel.g",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder; message conveys "missing"
            )

        f_out = self._branch_output_type(self.f, "f")
        if f_out != DslType.GRID:
            raise TypeMismatchError(
                where="Parallel.f",
                expected=DslType.GRID,
                actual=f_out,
            )

        g_out = self._branch_output_type(self.g, "g")
        if g_out != DslType.GRID:
            raise TypeMismatchError(
                where="Parallel.g",
                expected=DslType.GRID,
                actual=g_out,
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
            where=f"Parallel.{slot} ({type(branch).__name__})",
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
        # Branch programs are META-parameters; the only AST-visible
        # input is the outer grid. The merge tag is a plain str param.
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(
                ("f", object),
                ("g", object),
                ("merge", str),
            ),
        )

    # -----------------------------------------------------------------
    # apply: evaluate both branches on the same input, then merge
    # -----------------------------------------------------------------

    def apply(self, grid):
        # Coerce input grid to numpy. We use this both for the background
        # lookup (the "or" rule needs it) and to feed both branches a
        # canonical numpy array — passing through Program objects rather
        # than raw lists keeps the interpreter contract clean.
        grid_np = np.asarray(grid, dtype=np.int32)

        out_f = evaluate(self._branch_as_program(self.f), grid_np)
        out_g = evaluate(self._branch_as_program(self.g), grid_np)

        out_f_np = np.asarray(out_f, dtype=np.int32)
        out_g_np = np.asarray(out_g, dtype=np.int32)

        # Shape-mismatch fallback: if the two branches produced grids of
        # different shapes there is no well-defined cell-wise merge.
        # Return f's output so the combinator stays total — the synthesis
        # engine will pay a higher MDL cost for the wrapper and the merge
        # will not have improved the fit.
        if out_f_np.shape != out_g_np.shape:
            return out_f_np

        if self.merge == "or":
            # f's non-background cells dominate; g shows through where
            # f drew background. The background is computed against the
            # ORIGINAL input grid, not f's output — this matches the
            # Spelke-OBJECTNESS reading where f's "figure" overlays g's
            # "ground".
            bg = _background_color(grid_np)
            return np.where(out_f_np != bg, out_f_np, out_g_np)

        if self.merge == "max":
            # Element-wise integer maximum on the int32 grids.
            return np.maximum(out_f_np, out_g_np)

        if self.merge == "first":
            # Degenerate-merge baseline: return f verbatim. Useful so the
            # search engine can probe whether g's branch contributes
            # anything; if "first" wins on MDL, g was dead weight.
            return out_f_np

        # Unreachable: __post_init__ already rejected unknown merge
        # tags. Keep this as defence-in-depth so a future refactor that
        # accidentally widens the vocabulary fails loudly here.
        raise ValueError(  # pragma: no cover
            f"Parallel.merge unexpectedly {self.merge!r} at apply time"
        )

    # -----------------------------------------------------------------
    # AST identity, MDL, display
    # -----------------------------------------------------------------

    def to_string(self) -> str:
        f_s = self._branch_to_string(self.f)
        g_s = self._branch_to_string(self.g)
        return f"Parallel(f={f_s}, g={g_s}, merge={self.merge})"

    def mdl_bits(self) -> float:
        # Base cost: log2(|primitive catalog| + combinator) for picking
        # Parallel as the composition shape. Plus 1 bit for the
        # meta-shape choice (Parallel vs. another single-input Grid→Grid
        # wrapper), plus log2(|MERGE_RULES|) bits for the merge-rule
        # choice (3 rules ≈ 1.585 bits), plus the MDL bits of both
        # branch programs.
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        merge_bits = math.log2(len(MERGE_RULES))
        f_cost = self._branch_mdl_bits(self.f)
        g_cost = self._branch_mdl_bits(self.g)
        return base + 1.0 + merge_bits + f_cost + g_cost

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
