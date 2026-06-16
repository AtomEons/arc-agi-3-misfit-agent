"""MaskBy(predicate, child) — masked-region child-application combinator.

Role
----
MaskBy is a *Primitive* in the typed AST sense: it declares a typed
signature and an `apply()` method like any other primitive. Its purpose
is to give the synthesis engine a concrete one-grid-in, one-grid-out
shape whose semantics are:

    1. compute a 2D bool mask from the input grid via a fixed predicate
       ∈ {"foreground", "background", "edge_touching", "largest_object"}
    2. apply the child program f to the entire input grid (producing a
       full-shape modified grid)
    3. restore the unmasked region from the original input

In one line:

    out = np.where(mask, modified, grid)

This is the canonical Spelke OBJECTNESS / SPATIAL composition for
"transform a region of interest, leave everything else alone." Many ARC
tasks have this shape — only the foreground recolors, only the
background tiles, only the largest object reflects, only edge-touching
cells get cleared. MaskBy gives the synthesizer one combinator shape
that captures the entire family parameterized by the predicate label.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("predicate", str),
               ("child_program", Program))

Construction validation
-----------------------
The child program (or a PrimitiveNode used directly) must produce Grid.
Anything else raises TypeMismatchError at construction time — synthesis
never enumerates an ill-typed MaskBy. The predicate must be one of the
four allowed labels; anything else raises ValueError at construction
time, which keeps the synthesis enumerator from spending budget on dead
shapes.

Predicate semantics
-------------------
The mask is computed deterministically from the input grid under
Spelke priors:

    "foreground"     — mask = (grid != background_color)
                       The background color is selected by the same rule
                       the perceptor uses (0 if present, else most-
                       frequent color).
    "background"     — mask = (grid == background_color)
                       Complement of foreground; useful for "fill the
                       empty cells" tasks.
    "edge_touching"  — mask = True for cells that belong to any object
                       whose bbox touches the grid edge. Objects fully
                       in the interior contribute no True cells. The
                       cells stamped True are the OBJECT cells, not the
                       whole bbox — single-cell flood-fill membership.
    "largest_object" — mask = True for cells that belong to the largest
                       perceived object (max area; ties broken by sort
                       order from perceive_grid, which sorts -area). All
                       other foreground cells AND background cells are
                       False.

For "edge_touching" and "largest_object", the perceptor's flood-fill
defines membership. A perceived object's cell-mask is reconstructed
from the bbox slice by `slice == obj.color & slice != background`,
matching the perceptor's own labelling rule (see `_flood_label` in
`perceptor.py`).

If the predicate produces an all-False mask, the result is equal to the
input — the child program's output is discarded everywhere. This is
the right thing: "transform the largest object" on a grid with no
objects should be a no-op, not a wholesale rewrite.

Semantics
---------
MaskBy.apply(grid) evaluates the child program against the *full* input
grid and then composes with the mask. Two design notes:

  - The child program sees the WHOLE input, not the masked region. This
    is important: many child transforms (Rotate, Translate, Symmetrize)
    are shape-aware and would behave differently on a tiny extracted
    subgrid. By giving the child the whole grid, we keep its
    type-stable Grid → Grid contract and only mask the OUTPUT.

  - If the child returns a grid whose shape does not match the input,
    we use the overlap region only. This matches the defensive choice
    already used by ForEachObject — the synthesis engine's MDL prior
    deprioritizes shape-changing children inside MaskBy.

The combinator's child program is a META-parameter — it is configured
at construction time and travels with the combinator instance. It does
NOT consume a typed AST child slot (no PrimitiveNode children); the
only AST-visible input is the outer grid.

MDL accounting
--------------
    catalog_bits + 1 (meta-shape) + 2 (predicate choice in 4)
                 + child_program_bits

The 2-bit predicate charge reflects exactly 4 allowed labels. The child
program contributes its full MDL cost — masks with simpler interior
programs win.

Hash / memoization
------------------
The combinator's `to_string()` and (via the wrapping PrimitiveNode) its
`hash_key()` include the predicate label and the child program's
structure. Two MaskBys with different predicates, or with different
child programs, MUST produce distinct hash keys — otherwise the
synthesis engine would re-use cached results from the wrong region.

Tier-1 disclosure
-----------------
MaskBy introduces no learned parameters, no pretrained weights, no LLM
calls, and no third-party imports. The four predicates are computed
from the deterministic perceptor (flood-fill + background-color rule).
The child program is a typed AST node from the same hand-authored
grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

import numpy as np

from ...perceptor import perceive_grid, _background_color
from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Signature, TypeMismatchError, DslType
from ..ast import Program, PrimitiveNode
from ..interpreter import evaluate


# A child program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted.
ChildLike = Union[Program, PrimitiveNode]


# The four allowed predicate labels. Anything else is rejected at
# construction time, which keeps the synthesis enumerator honest.
ALLOWED_PREDICATES = ("foreground", "background", "edge_touching", "largest_object")


@dataclass
class MaskBy(Primitive):
    """MaskBy(predicate, child_program): apply child to the masked region only.

    The child program is a META-parameter — it is configured at
    construction time and travels with the combinator instance. It does
    NOT consume a typed AST child slot (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        predicate: one of {"foreground", "background", "edge_touching",
            "largest_object"}. Determines the 2D bool mask computed from
            the input grid.
        child_program: a typed Program (or PrimitiveNode) whose output
            type is Grid. Evaluated against the full input grid; the
            result is composed with the mask to restore the unmasked
            region from the original input.

    Raises:
        TypeMismatchError: at construction time, if the child program's
            output type is not Grid, or if the child program is missing.
        ValueError: at construction time, if the predicate is not one of
            the four allowed labels.
    """

    predicate: str = "foreground"
    child_program: ChildLike = None

    def __post_init__(self):
        # Validate the predicate label first — a bad label is a
        # synthesizer authoring error, not a type error.
        if self.predicate not in ALLOWED_PREDICATES:
            raise ValueError(
                f"MaskBy.predicate must be one of {ALLOWED_PREDICATES}, "
                f"got {self.predicate!r}"
            )

        # Validate the child program's shape and type at construction
        # time. This is the central type-safety contract: the synthesis
        # engine must never get a MaskBy whose child can return Number,
        # Color, ObjSet, etc.
        if self.child_program is None:
            raise TypeMismatchError(
                where="MaskBy.child_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder — message conveys "missing"
            )

        child_out_type = self._child_output_type()
        if child_out_type != DslType.GRID:
            raise TypeMismatchError(
                where="MaskBy.child_program",
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
        # Anything else is an authoring error — surface as a type error
        # so the synthesis engine's normal type-mismatch handling kicks
        # in instead of a generic TypeError.
        raise TypeMismatchError(
            where=f"MaskBy.child_program ({type(self.child_program).__name__})",
            expected=DslType.GRID,
            actual=DslType.GRID,
        )

    def _child_as_program(self) -> Program:
        """Coerce the child into a Program for the interpreter."""
        if isinstance(self.child_program, Program):
            return self.child_program
        # PrimitiveNode → wrap in a Program for the interpreter.
        return Program(root=self.child_program, desired_output=DslType.GRID)

    def _child_hash_key(self) -> str:
        """Stable hash key fragment for the child program."""
        if isinstance(self.child_program, Program):
            return self.child_program.hash_key()
        if isinstance(self.child_program, PrimitiveNode):
            return self.child_program.hash_key()
        return repr(self.child_program)

    def _child_to_string(self) -> str:
        """Human-readable string for the child program."""
        if isinstance(self.child_program, Program):
            return self.child_program.to_string()
        if isinstance(self.child_program, PrimitiveNode):
            return self.child_program.to_string()
        return repr(self.child_program)

    def _child_mdl_bits(self) -> float:
        """Sum the MDL bits contributed by the child program."""
        if isinstance(self.child_program, Program):
            return _sum_node_mdl(self.child_program.root)
        if isinstance(self.child_program, PrimitiveNode):
            return _sum_node_mdl(self.child_program)
        return 0.0

    # -----------------------------------------------------------------
    # Predicate → 2D bool mask
    # -----------------------------------------------------------------

    def compute_mask(self, grid: np.ndarray) -> np.ndarray:
        """Compute the 2D bool mask for the configured predicate.

        Public so tests and synthesis-side reasoning can inspect the
        mask without re-running the whole combinator.
        """
        return _compute_mask(grid, self.predicate)

    # -----------------------------------------------------------------
    # Typed signature
    # -----------------------------------------------------------------

    def signature_typed(self) -> Signature:
        # The child program is a META-parameter — declared in params (so
        # the type-checker knows it's not a typed AST slot) but enforced
        # at construction time via _child_output_type(). The predicate
        # is a plain str param drawn from a fixed label set.
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(
                ("predicate", str),
                ("child_program", object),
            ),
        )

    # -----------------------------------------------------------------
    # apply: predicate → mask → evaluate child → np.where compose
    # -----------------------------------------------------------------

    def apply(self, grid):
        grid_np = np.asarray(grid, dtype=np.int32)
        mask = _compute_mask(grid_np, self.predicate)

        child_program = self._child_as_program()
        modified = evaluate(child_program, grid_np)
        modified = np.asarray(modified, dtype=np.int32)

        # If the child returned a grid whose shape does not match the
        # input, use the overlap region only. The synthesis engine's MDL
        # prior deprioritizes shape-changing children inside MaskBy, but
        # we should not raise here — we should compose what we can.
        if modified.shape != grid_np.shape:
            mh, mw = modified.shape
            gh, gw = grid_np.shape
            h = min(mh, gh)
            w = min(mw, gw)
            out = grid_np.copy()
            # Apply the mask only over the overlap region; outside the
            # overlap, the original grid is preserved (effectively the
            # "restore unmasked region" rule extended to "restore the
            # whole row/col if there is no modified cell available").
            sub_mask = mask[:h, :w]
            out[:h, :w] = np.where(sub_mask, modified[:h, :w], grid_np[:h, :w])
            return out

        return np.where(mask, modified, grid_np).copy()

    # -----------------------------------------------------------------
    # AST identity, MDL, display
    # -----------------------------------------------------------------

    def to_string(self) -> str:
        inner = self._child_to_string()
        return f"MaskBy(pred={self.predicate}, {inner})"

    def mdl_bits(self) -> float:
        # Base cost: log2(|primitive catalog| + combinator) for picking
        # MaskBy as the composition shape. Plus 1 bit for the meta-shape
        # choice (MaskBy vs. another single-input Grid→Grid wrapper),
        # plus 2 bits for the predicate choice (ceil(log2(4)) = 2), plus
        # the MDL bits of the child program.
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        predicate_bits = 2.0
        child_cost = self._child_mdl_bits()
        return base + 1.0 + predicate_bits + child_cost


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------


def _compute_mask(grid: np.ndarray, predicate: str) -> np.ndarray:
    """Compute the 2D bool mask under the requested predicate.

    Deterministic and stateless: same input grid + same predicate label
    always produces the same mask. Used by MaskBy.apply() at runtime and
    exposed (via MaskBy.compute_mask) for synthesis-side inspection.
    """
    grid = np.asarray(grid, dtype=np.int32)
    bg = _background_color(grid)

    if predicate == "foreground":
        return grid != bg

    if predicate == "background":
        return grid == bg

    if predicate == "edge_touching":
        scene = perceive_grid(grid)
        mask = np.zeros_like(grid, dtype=bool)
        for obj in scene.objects:
            if not obj.touches_edge:
                continue
            r0, c0, r1, c1 = obj.bbox
            # Reconstruct the object's cell-mask the same way the
            # perceptor labels it: foreground cells inside the bbox
            # whose color matches the object's color. This matches the
            # perceptor's flood-fill rule (see `_flood_label`).
            sub = grid[r0:r1 + 1, c0:c1 + 1]
            sub_mask = (sub == obj.color) & (sub != bg)
            mask[r0:r1 + 1, c0:c1 + 1] |= sub_mask
        return mask

    if predicate == "largest_object":
        scene = perceive_grid(grid)
        if not scene.objects:
            return np.zeros_like(grid, dtype=bool)
        # perceive_grid sorts objects by -area, so objects[0] is the
        # largest. Ties are broken by the perceptor's own sort order
        # (deterministic across calls on the same grid).
        obj = scene.objects[0]
        r0, c0, r1, c1 = obj.bbox
        mask = np.zeros_like(grid, dtype=bool)
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        sub_mask = (sub == obj.color) & (sub != bg)
        mask[r0:r1 + 1, c0:c1 + 1] = sub_mask
        return mask

    # Should be unreachable — construction-time validation rejects any
    # other label — but we surface a clear error if a caller bypasses
    # the constructor (e.g. by mutating .predicate post-hoc).
    raise ValueError(
        f"unknown MaskBy predicate at runtime: {predicate!r} "
        f"(expected one of {ALLOWED_PREDICATES})"
    )


def _sum_node_mdl(node) -> float:
    """Sum MDL bits over a node subtree (PrimitiveNode + children)."""
    if isinstance(node, PrimitiveNode):
        total = float(node.primitive.mdl_bits())
        for child in node.children:
            total += _sum_node_mdl(child)
        return total
    return 0.0
