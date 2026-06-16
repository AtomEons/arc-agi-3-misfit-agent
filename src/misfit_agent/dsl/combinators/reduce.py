"""Reduce(count_program, transform_program) — count-driven repeated-application combinator.

Role
----
Reduce lifts two child programs into a single Grid→Grid combinator that
evaluates one program (the *count program*) on the input grid to get a
Number n, then repeatedly applies a second program (the *transform
program*) to the input n times. Pseudo-code:

    apply(grid):
        n = evaluate(count_program, grid)
        result = grid.copy()
        for _ in range(int(n)):
            result = evaluate(transform_program, result)
        return result

This is the canonical Spelke NUMEROSITY → ACTION lift: a grid property
(e.g. "how many objects are there?") drives the iteration count of a
geometric transform. Common idioms it captures:

  - "rotate the grid once per object" → Reduce(CountObj, Rotate(k=1))
  - "tile the grid as many times as there are objects" → Reduce(CountObj, Tile(...))
  - "translate one step per object" → Reduce(CountObj, Translate(dy=1, dx=0))

Reduce differs from WhileChanging in three ways. WhileChanging iterates
until a structural fixed point is reached (or a cap is hit); Reduce
iterates a deterministic, *data-derived* number of times. WhileChanging
has one child program; Reduce has two — and the count program's output
type is Number, not Grid. Finally, WhileChanging's iteration cap is a
constant scalar meta-param; Reduce's iteration count is a function of
the input grid that is computed fresh on every call.

Signature
---------
    inputs  = (("g", Grid),)
    output  = Grid
    params  = (("count_program", object),
               ("transform_program", object))

Both child programs are META-parameters: they are configured at Reduce
construction time and travel with the combinator instance. Neither
consumes a typed AST child slot; the only AST-visible input is the outer
grid. This mirrors how WhileChanging and Parallel handle their child
programs and is the same META-parameter contract the synthesis engine
already exercises.

Construction validation
-----------------------
The count program (Program or PrimitiveNode) must produce Number. The
transform program (Program or PrimitiveNode) must produce Grid. Anything
else raises TypeMismatchError at construction time — synthesis never
enumerates an ill-typed Reduce. Missing children are also rejected.

Zero-count semantics
--------------------
If the count program returns 0 (no objects perceived, etc.), the loop
body executes zero times and Reduce returns a defensive copy of the
input grid. This is the right Spelke reading: zero applications of a
transform is the identity transform. It also matches Python's range(0)
contract and keeps the combinator total — synthesis never encounters
a "Reduce blew up on an empty grid" exception, only an MDL signal that
the wrapper added bits without changing the output.

Negative-count safety
---------------------
A negative count from the count program would make range() raise. We
clamp to max(0, int(n)) so the combinator stays total. A negative count
in the input data is itself a signal of a misbehaving count program; the
MDL prior will deprioritize the wrapper rather than the runtime raising.

MDL accounting
--------------
    catalog_bits + 1 (meta-shape) + sum(count_program mdl bits)
                                  + sum(transform_program mdl bits)

The combinator's bit cost is strictly greater than either child's cost
alone, so the synthesis prior pays a real price to pick Reduce over a
bare primitive. There is no iteration-cap meta-param to encode (unlike
WhileChanging) — the count is a function of the input, not a configured
scalar. This keeps the MDL signal honest: the cost reflects exactly the
two committed sub-programs.

Hash / memoization
------------------
The combinator's `to_string()` and (via wrapping PrimitiveNode) its
`hash_key()` include BOTH child program structures. Two Reduces with
different count programs, different transform programs, or different
parameter values inside either subprogram MUST produce distinct hash
keys — otherwise memoization tables would re-use cached results across
genuinely different programs.

Tier-1 disclosure
-----------------
Reduce introduces no learned parameters, no pretrained weights, no LLM
calls, and no third-party imports. The iteration count is derived
structurally from the input grid by an already-disclosed primitive (e.g.
CountObj, whose semantics are documented in primitives.py). Both child
programs are typed AST nodes from the same hand-authored grammar. Pure
search and structural operations only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

import numpy as np

from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Number, Signature, TypeMismatchError, DslType
from ..ast import Program, PrimitiveNode
from ..interpreter import evaluate


# A child program can be supplied either as a wrapped Program or as a
# raw PrimitiveNode (the synthesis engine sometimes hands us a node
# directly). Both are accepted — mirrors the WhileChanging / Parallel
# conventions used elsewhere in the combinator package.
ChildLike = Union[Program, PrimitiveNode]


@dataclass
class Reduce(Primitive):
    """Reduce(count_program, transform_program): repeat a Grid→Grid program
    a count derived from the input grid.

    The two child programs are META-parameters — they are configured at
    construction time and travel with the combinator instance. Neither
    consumes a typed AST child slot (no PrimitiveNode children); the
    only AST-visible input is the outer grid.

    Args:
        count_program: a typed Program (or PrimitiveNode) whose output
            type is Number. Evaluated once per call against the input
            grid to produce the iteration count n. Negative results are
            clamped to 0; non-integer numeric results are coerced via
            int().
        transform_program: a typed Program (or PrimitiveNode) whose
            output type is Grid. Applied repeatedly to the running grid
            n times.

    Raises:
        TypeMismatchError: at construction time, if the count program's
            output type is not Number, the transform program's output
            type is not Grid, or either child is missing.
    """

    count_program: ChildLike = None
    transform_program: ChildLike = None

    def __post_init__(self):
        # Validate both child programs at construction time. This is the
        # central type-safety contract: the synthesis engine must never
        # get a Reduce whose count returns Grid or whose transform
        # returns Number. The error messages name the slot so debugging
        # a synthesis-mid-flight rejection isn't a guessing game.
        if self.count_program is None:
            raise TypeMismatchError(
                where="Reduce.count_program",
                expected=DslType.NUMBER,
                actual=DslType.NUMBER,  # placeholder; message conveys "missing"
            )
        if self.transform_program is None:
            raise TypeMismatchError(
                where="Reduce.transform_program",
                expected=DslType.GRID,
                actual=DslType.GRID,  # placeholder; message conveys "missing"
            )

        count_out = self._child_output_type(self.count_program, "count_program")
        if count_out != DslType.NUMBER:
            raise TypeMismatchError(
                where="Reduce.count_program",
                expected=DslType.NUMBER,
                actual=count_out,
            )

        transform_out = self._child_output_type(
            self.transform_program, "transform_program"
        )
        if transform_out != DslType.GRID:
            raise TypeMismatchError(
                where="Reduce.transform_program",
                expected=DslType.GRID,
                actual=transform_out,
            )

    # -----------------------------------------------------------------
    # Internal helpers (mirrors WhileChanging / Parallel for stylistic
    # consistency across the combinator package)
    # -----------------------------------------------------------------

    @staticmethod
    def _child_output_type(child: ChildLike, slot: str) -> DslType:
        """Return a child program's output DslType, regardless of wrap."""
        if isinstance(child, Program):
            return child.output_type()
        if isinstance(child, PrimitiveNode):
            return child.output_type
        # Anything else is an authoring error — surface as a type error
        # so the synthesis engine's normal mismatch handling kicks in
        # instead of a generic TypeError elsewhere in the stack.
        # The "expected" type here is slot-dependent, but we set it to
        # the slot's declared type so the error message is honest about
        # what the slot would have accepted.
        expected = (
            DslType.NUMBER if slot == "count_program" else DslType.GRID
        )
        raise TypeMismatchError(
            where=f"Reduce.{slot} ({type(child).__name__})",
            expected=expected,
            actual=expected,
        )

    @staticmethod
    def _child_as_program(child: ChildLike, desired: DslType) -> Program:
        """Coerce a child into a Program for the interpreter."""
        if isinstance(child, Program):
            return child
        # PrimitiveNode → wrap in a Program for the interpreter.
        return Program(root=child, desired_output=desired)

    @staticmethod
    def _child_hash_key(child: ChildLike) -> str:
        """Stable hash key fragment for a child program."""
        if isinstance(child, Program):
            return child.hash_key()
        if isinstance(child, PrimitiveNode):
            return child.hash_key()
        return repr(child)

    @staticmethod
    def _child_to_string(child: ChildLike) -> str:
        """Human-readable string for a child program."""
        if isinstance(child, Program):
            return child.to_string()
        if isinstance(child, PrimitiveNode):
            return child.to_string()
        return repr(child)

    # -----------------------------------------------------------------
    # Typed signature
    # -----------------------------------------------------------------

    def signature_typed(self) -> Signature:
        # Child programs are META-parameters declared in `params`; the
        # only AST-visible input is the outer grid. The type checker
        # treats meta-params as scalar metadata and does not walk them,
        # so the construction-time validation above is the sole enforcer
        # of child-output-type correctness.
        return Signature(
            inputs=(("g", Grid),),
            output=Grid,
            params=(
                ("count_program", object),
                ("transform_program", object),
            ),
        )

    # -----------------------------------------------------------------
    # apply: derive n from count_program, repeat transform_program n times
    # -----------------------------------------------------------------

    def apply(self, grid):
        """Evaluate count_program once, then apply transform_program n times.

        Defensive copy: never mutate the caller's input. The running
        grid is always a fresh numpy array, so even if the transform
        program returns a view into its argument, subsequent iterations
        operate on the result, not the input.

        Coercion:
          - count is coerced via int() so float-valued or numpy-scalar
            counts work
          - negative counts clamp to 0 (zero applications = identity)
        """
        # Coerce input to a canonical numpy array so the interpreter
        # contract is consistent (matches Identity().apply convention).
        grid_np = np.asarray(grid).copy()

        # Compute n from the count program. The count program is
        # evaluated against the ORIGINAL input grid, not the running
        # transform output — this matches the semantic reading "the
        # count of objects in the input drives how many times we
        # transform". A different reading ("count the output each
        # iteration") would be a different combinator (WhileChanging is
        # closer to that, with structural fixed-point detection).
        count_prog = self._child_as_program(self.count_program, DslType.NUMBER)
        n_raw = evaluate(count_prog, grid_np)

        # Coerce. int() works for Python ints, floats, numpy scalars,
        # and bool. Negative results clamp to 0 (a no-op iteration).
        n = max(0, int(n_raw))

        # Apply the transform n times. The running grid starts as a
        # copy of the input (matching the spec); each iteration replaces
        # it with the transform's output.
        transform_prog = self._child_as_program(
            self.transform_program, DslType.GRID
        )
        result = grid_np
        for _ in range(n):
            result = np.asarray(evaluate(transform_prog, result))

        return result

    # -----------------------------------------------------------------
    # AST identity, MDL, display
    # -----------------------------------------------------------------

    def to_string(self) -> str:
        count_s = self._child_to_string(self.count_program)
        transform_s = self._child_to_string(self.transform_program)
        return (
            f"Reduce(count={count_s}, transform={transform_s})"
        )

    def mdl_bits(self) -> float:
        """Total bits to encode this combinator + both child programs.

        Components:
          - log2(|catalog|+1) bits  — picking Reduce as the composition
            shape from the catalog
          - 1 bit                   — meta-shape choice (Reduce vs another
            single-input Grid→Grid wrapper that takes two sub-programs)
          - sum(count_program mdl bits)     — encoding the count program
          - sum(transform_program mdl bits) — encoding the transform program

        There is no constant iteration-cap to encode (unlike
        WhileChanging.max_iter): the iteration count is a function of
        the input grid, not a free meta-param. This keeps the MDL signal
        focused on the two committed sub-programs.
        """
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        count_cost = self._child_mdl_bits(self.count_program)
        transform_cost = self._child_mdl_bits(self.transform_program)
        return base + 1.0 + count_cost + transform_cost

    @staticmethod
    def _child_mdl_bits(child: ChildLike) -> float:
        """Sum the MDL bits contributed by a child program subtree."""
        if isinstance(child, Program):
            return _sum_node_mdl(child.root)
        if isinstance(child, PrimitiveNode):
            return _sum_node_mdl(child)
        return 0.0


def _sum_node_mdl(node) -> float:
    """Sum MDL bits over a node subtree (PrimitiveNode + children)."""
    if isinstance(node, PrimitiveNode):
        total = float(node.primitive.mdl_bits())
        for child in node.children:
            total += _sum_node_mdl(child)
        return total
    return 0.0
