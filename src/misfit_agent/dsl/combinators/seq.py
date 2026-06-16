"""Seq(f, g) — the canonical 2-child Grid→Grid sequencing combinator.

Role
----
Seq is a *Primitive* in the typed AST sense: it declares a typed signature
and an `apply()` method like any other primitive. Its purpose is to give
the synthesis engine a concrete two-argument composition shape it can
attach holes to:

    Seq(<Grid?#0>, <Grid?#1>)

The synthesis engine enumerates program shapes by stacking primitive and
combinator nodes; Seq is the canonical "two Grid-producing subprograms"
hole pattern. Without it, every composition would have to be expressed by
*nesting* one primitive inside another, and the enumerator would have to
re-derive the "two-child Grid sibling" shape on the fly.

Semantics
---------
The DSL interpreter already evaluates a `PrimitiveNode`'s children left-
to-right and passes the reduced values up to `primitive.apply(*values)`.
That left-to-right evaluation IS the "f then g" semantics — by the time
Seq.apply() is called, f's value (x) and g's value (y) are already
realized. Seq's job at evaluation time is therefore minimal:

    Seq.apply(x, y) -> y

Returning the second child's value is the right thing: it matches the
"final output of the f-then-g chain is whatever g produced" reading. For
truly *threaded* composition (g operates on f's output), the synthesis
engine fills the holes with nested subprograms — that is the existing
mechanism. Seq does not need to invent threading; it just needs to honour
the sibling shape.

Tier-1 disclosure
-----------------
Seq introduces no learned parameters, no pretrained weights, no LLM
calls, and no third-party imports. It composes the existing typed
substrate. The `mdl_bits()` cost reflects that picking Seq vs. another
composition shape is a real choice in the synthesizer's grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..primitives import Primitive, ALL_PRIMITIVES
from ..types import Grid, Signature


@dataclass
class Seq(Primitive):
    """Seq(f, g): two Grid-producing children, output is g's value.

    Signature: (("f_out", Grid), ("g_out", Grid)) → Grid

    At construction time, PrimitiveNode enforces that both children have
    output_type == Grid; otherwise it raises TypeMismatchError. So a
    Number-typed child where Grid is expected is rejected before the
    program ever runs.
    """

    def signature_typed(self) -> Signature:
        return Signature(
            inputs=(("f_out", Grid), ("g_out", Grid)),
            output=Grid,
            params=(),
        )

    def apply(self, f_out, g_out):
        # Interpreter already evaluated both children. The "f then g"
        # contract: the final value of the chain is what g produced.
        return g_out

    def to_string(self) -> str:
        return "Seq"

    def mdl_bits(self) -> float:
        # Base cost from the primitive catalog plus 1 bit for the
        # composition-shape choice. Seq is one of (at least) two shapes
        # the synthesizer can pick at any Grid→Grid slot (the other being
        # a single nested primitive), so the choice is worth >= 1 bit.
        base = math.log2(max(len(ALL_PRIMITIVES) + 1, 2))
        return base + 1.0
