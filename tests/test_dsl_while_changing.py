"""WhileChanging combinator: fixed-point iteration semantics.

Tests cover:
  - signature_typed() reports Grid → Grid with the right meta-params
  - WhileChanging(Identity) reaches a fixed-point immediately (no change
    after one iteration) and returns the input grid value-equal
  - WhileChanging(Gravity(direction='D')) on a settled grid returns
    immediately; on an unsettled column it converges to the settled grid
  - max_iter cap respected: a deliberately non-convergent child program
    (180° rotate of an asymmetric grid flips back and forth) stops at
    the cap and does not infinite-loop
  - mdl_bits() strictly greater than the bare child program's MDL bits
    (the combinator costs real encoding bits)
  - Hash key carries BOTH the child program and the max_iter param:
    distinct children OR distinct max_iter values produce distinct
    PrimitiveNode hash keys (memoization correctness)
  - Construction with a non-Grid child output type raises
    TypeMismatchError BEFORE any program runs
  - Construction with non-positive max_iter raises ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Number, DslType, TypeMismatchError,
    Identity, Rotate, Gravity, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, make_hole, make_program,
)
from misfit_agent.dsl.combinators.while_changing import WhileChanging
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_child() -> Program:
    """Identity(<input>) — a fixed-point on the first iteration."""
    return make_program(Identity(), make_hole(Grid))


def _gravity_child(direction: str = "D") -> Program:
    """Gravity(direction=...)(<input>) — settles non-bg cells in a direction."""
    return make_program(Gravity(direction=direction), make_hole(Grid))


def _rotate_child(k: int = 1) -> Program:
    """Rotate(k=k)(<input>) — used as a deliberately non-convergent child
    when the grid is asymmetric (k=2 on a non-180°-symmetric grid flips
    between two distinct states forever)."""
    return make_program(Rotate(k=k), make_hole(Grid))


def _unsettled_gravity_grid() -> np.ndarray:
    """A 5x3 grid where each column has a non-bg cell ABOVE empty space.

    Layout (0 = bg):
        1 2 3
        0 0 0
        0 0 0
        0 0 0
        0 0 0

    With Gravity(direction='D'), the converged form is:
        0 0 0
        0 0 0
        0 0 0
        0 0 0
        1 2 3

    The Gravity primitive is implemented as a one-pass column scan that
    collapses non-bg cells to the floor in a single application — so the
    fixed point is reached after exactly one iteration. That's still a
    meaningful test: WhileChanging must (a) detect convergence and (b)
    return the right grid, not just the input.
    """
    g = np.zeros((5, 3), dtype=np.int32)
    g[0, 0] = 1
    g[0, 1] = 2
    g[0, 2] = 3
    return g


def _settled_gravity_grid() -> np.ndarray:
    """The same shape after gravity has already settled. WhileChanging
    should detect "no change on first iteration" and return immediately."""
    g = np.zeros((5, 3), dtype=np.int32)
    g[4, 0] = 1
    g[4, 1] = 2
    g[4, 2] = 3
    return g


def _asymmetric_grid_for_rot180() -> np.ndarray:
    """An asymmetric grid that is NOT 180°-rotation-invariant.

    With Rotate(k=2), this grid flips between itself and its rot180,
    never reaching a fixed point — useful for testing the max_iter cap.

    Layout (0 = bg):
        1 0 0
        0 0 0
        0 0 0

    rot180 of this is:
        0 0 0
        0 0 0
        0 0 1

    These are distinct grids, so Rotate(k=2) bounces between them.
    """
    g = np.zeros((3, 3), dtype=np.int32)
    g[0, 0] = 1
    return g


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """WhileChanging(f, max_iter) declares Grid → Grid with one AST input."""
    wc = WhileChanging(child_program=_identity_child())
    sig = wc.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    # And the type-checked alias matches:
    assert sig.output == DslType.GRID
    # The signature declares child_program AND max_iter as meta-params.
    param_names = [n for n, _ in sig.params]
    assert "child_program" in param_names
    assert "max_iter" in param_names


def test_signature_max_iter_param_is_int_typed():
    """The max_iter meta-param is declared as a Python int."""
    wc = WhileChanging(child_program=_identity_child(), max_iter=8)
    sig = wc.signature_typed()
    max_iter_types = [t for n, t in sig.params if n == "max_iter"]
    assert max_iter_types == [int]


# ---------------------------------------------------------------------------
# 2. Identity child → fixed-point on iteration 1
# ---------------------------------------------------------------------------


def test_identity_child_reaches_fixed_point_immediately():
    """WhileChanging(Identity) detects "no change" on iteration 1 and
    returns a grid value-equal to the input."""
    g = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 7]], dtype=np.int32)
    wc = WhileChanging(child_program=_identity_child())
    out = wc.apply(g)
    assert np.array_equal(out, g), (
        f"WhileChanging(Identity) altered the input:\nIN:\n{g}\nOUT:\n{out}"
    )


def test_identity_child_via_program_node_round_trip():
    """WhileChanging is reachable through the typed AST."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    wc = WhileChanging(child_program=_identity_child())
    root = PrimitiveNode(primitive=wc, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    assert np.array_equal(out, g)


def test_identity_child_does_not_mutate_caller_input():
    """The input grid object must not be mutated by apply()."""
    g = np.array([[1, 0], [0, 2]], dtype=np.int32)
    g_before = g.copy()
    wc = WhileChanging(child_program=_identity_child())
    _ = wc.apply(g)
    assert np.array_equal(g, g_before), (
        f"WhileChanging mutated the caller's input:\n"
        f"before:\n{g_before}\nafter:\n{g}"
    )


# ---------------------------------------------------------------------------
# 3. Gravity child → fixed point after settling
# ---------------------------------------------------------------------------


def test_gravity_child_on_already_settled_grid_is_noop():
    """A grid where gravity has nothing to do hits fixed-point in 1 pass."""
    g = _settled_gravity_grid()
    wc = WhileChanging(child_program=_gravity_child(direction="D"))
    out = wc.apply(g)
    assert np.array_equal(out, g), (
        f"WhileChanging(Gravity-D) altered an already-settled grid:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_gravity_child_converges_to_settled_grid():
    """An unsettled grid converges to its gravity-settled form."""
    g = _unsettled_gravity_grid()
    expected = _settled_gravity_grid()
    wc = WhileChanging(child_program=_gravity_child(direction="D"))
    out = wc.apply(g)
    assert np.array_equal(out, expected), (
        f"WhileChanging(Gravity-D) did not converge to expected:\n"
        f"IN:\n{g}\nOUT:\n{out}\nEXPECTED:\n{expected}"
    )


def test_gravity_child_converges_at_or_before_max_iter():
    """Even with a very small max_iter, a Gravity settle that completes
    in one pass must converge correctly. This guards against an off-by-one
    in the fixed-point detection loop."""
    g = _unsettled_gravity_grid()
    expected = _settled_gravity_grid()
    wc = WhileChanging(child_program=_gravity_child(direction="D"), max_iter=2)
    out = wc.apply(g)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# 4. max_iter cap respected when fixed point never reached
# ---------------------------------------------------------------------------


def test_max_iter_cap_respected_on_non_convergent_child():
    """Rotate(k=2) on an asymmetric grid never converges — it flips
    between two distinct states. WhileChanging must stop at max_iter
    rather than loop forever. With max_iter even, output equals the
    original grid (back where we started). The key assertion is that
    apply() returns at all — no infinite loop, no exception."""
    g = _asymmetric_grid_for_rot180()
    wc = WhileChanging(child_program=_rotate_child(k=2), max_iter=4)
    out = wc.apply(g)
    # After 4 flips of a 2-cycle, we are back at the original grid.
    assert np.array_equal(out, g), (
        f"After even number of rot180 flips, expected original:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_max_iter_cap_with_odd_iterations_yields_other_state():
    """With max_iter=3 (odd) on a 2-cycle, we end on the OTHER state.
    This proves that max_iter actually controls the iteration count
    rather than being silently overridden."""
    g = _asymmetric_grid_for_rot180()
    wc = WhileChanging(child_program=_rotate_child(k=2), max_iter=3)
    out = wc.apply(g)
    expected_rot180 = np.rot90(g, k=2)
    assert np.array_equal(out, expected_rot180), (
        f"After odd number of rot180 flips, expected rot180:\n"
        f"IN:\n{g}\nOUT:\n{out}\nEXPECTED (rot180):\n{expected_rot180}"
    )


def test_default_max_iter_is_sensible_for_arc_grids():
    """The default max_iter is positive and at least large enough to
    settle any plausible 30x30 ARC gravity column (need <= 30 passes
    in the worst case, but real primitives single-pass these)."""
    wc = WhileChanging(child_program=_identity_child())
    assert isinstance(wc.max_iter, int)
    assert wc.max_iter >= 1


# ---------------------------------------------------------------------------
# 5. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_strictly_greater_than_bare_child():
    """Wrapping a child in WhileChanging must cost strictly more bits
    than the child alone. The wrapping commits to a composition shape
    AND an iteration cap, both of which are real encoding choices."""
    bare_child_bits = Identity().mdl_bits()
    wc = WhileChanging(child_program=_identity_child())
    assert wc.mdl_bits() > bare_child_bits, (
        f"WhileChanging did not cost more bits than its bare child: "
        f"wc={wc.mdl_bits()}  bare={bare_child_bits}"
    )


def test_mdl_bits_is_strictly_positive_and_finite():
    """The MDL cost is well-formed: positive, finite, not NaN."""
    wc = WhileChanging(child_program=_identity_child())
    bits = wc.mdl_bits()
    assert bits > 0.0
    assert bits == bits  # not NaN
    assert bits != float("inf")


def test_mdl_bits_scales_with_max_iter():
    """A larger max_iter costs strictly more bits than a smaller one.
    The synthesis prior should prefer the shortest cap that still fits."""
    wc_short = WhileChanging(child_program=_identity_child(), max_iter=2)
    wc_long = WhileChanging(child_program=_identity_child(), max_iter=64)
    assert wc_long.mdl_bits() > wc_short.mdl_bits(), (
        f"MDL bits did not grow with max_iter: "
        f"short(max=2)={wc_short.mdl_bits()} long(max=64)={wc_long.mdl_bits()}"
    )


def test_mdl_bits_scales_with_child_complexity():
    """A WhileChanging wrapping a parameterized child (Rotate) costs
    strictly more bits than one wrapping Identity."""
    wc_short = WhileChanging(child_program=_identity_child())
    wc_long = WhileChanging(child_program=_rotate_child(k=1))
    assert wc_long.mdl_bits() > wc_short.mdl_bits()


# ---------------------------------------------------------------------------
# 6. Hash key includes BOTH child program AND max_iter
# ---------------------------------------------------------------------------


def test_hash_key_distinguishes_distinct_children():
    """Two WhileChangings with different child programs must produce
    different PrimitiveNode hash keys — memoization correctness."""
    wc_id = WhileChanging(child_program=_identity_child())
    wc_rot = WhileChanging(child_program=_rotate_child(k=1))
    node_id = PrimitiveNode(primitive=wc_id, children=[make_hole(Grid)])
    node_rot = PrimitiveNode(primitive=wc_rot, children=[make_hole(Grid)])
    assert node_id.hash_key() != node_rot.hash_key(), (
        f"hash_key collision between distinct WhileChanging children:\n"
        f"  identity: {node_id.hash_key()}\n"
        f"  rotate:   {node_rot.hash_key()}"
    )
    # And the child's name shows up inside the hash key:
    assert "Identity" in node_id.hash_key()
    assert "Rotate" in node_rot.hash_key()


def test_hash_key_distinguishes_max_iter_values():
    """Same child program with different max_iter values must hash
    differently — otherwise the synthesis engine could reuse a cached
    result computed under a tighter cap for a query with a looser cap."""
    wc_4 = WhileChanging(child_program=_identity_child(), max_iter=4)
    wc_32 = WhileChanging(child_program=_identity_child(), max_iter=32)
    node_4 = PrimitiveNode(primitive=wc_4, children=[make_hole(Grid)])
    node_32 = PrimitiveNode(primitive=wc_32, children=[make_hole(Grid)])
    assert node_4.hash_key() != node_32.hash_key(), (
        f"hash_key collision between max_iter=4 and max_iter=32:\n"
        f"  4:  {node_4.hash_key()}\n"
        f"  32: {node_32.hash_key()}"
    )


def test_hash_key_carries_combinator_marker():
    """The hash key must mention 'WhileChanging' so memoization tables
    stay disambiguating from ForEachObject or any other combinator."""
    wc = WhileChanging(child_program=_identity_child())
    node = PrimitiveNode(primitive=wc, children=[make_hole(Grid)])
    prog = Program(root=node, desired_output=Grid)
    assert "WhileChanging" in prog.hash_key()
    assert "WhileChanging" in prog.to_string()


def test_hash_key_distinguishes_child_parameters():
    """Even within the same child primitive family, different parameter
    values must yield different hash keys — guards Rotate(k=1) vs
    Rotate(k=2) inside WhileChanging."""
    wc_k1 = WhileChanging(child_program=_rotate_child(k=1))
    wc_k2 = WhileChanging(child_program=_rotate_child(k=2))
    n1 = PrimitiveNode(primitive=wc_k1, children=[make_hole(Grid)])
    n2 = PrimitiveNode(primitive=wc_k2, children=[make_hole(Grid)])
    assert n1.hash_key() != n2.hash_key()


# ---------------------------------------------------------------------------
# 7. Type-mismatch on bad child output
# ---------------------------------------------------------------------------


def test_type_mismatch_raised_if_child_output_is_not_grid():
    """A child program whose output type is Number (CountObj) must be
    rejected at construction time — synthesis must never enumerate
    a WhileChanging that tries to iterate a number-producing program."""
    bad_child = make_program(CountObj(), make_hole(Grid))
    assert bad_child.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        WhileChanging(child_program=bad_child)
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_child_program_missing():
    """No child program at all is a type error — there is no default."""
    with pytest.raises(TypeMismatchError):
        WhileChanging(child_program=None)


# ---------------------------------------------------------------------------
# 8. max_iter validation at construction time
# ---------------------------------------------------------------------------


def test_max_iter_zero_rejected():
    """max_iter=0 would make the combinator a no-op — rejected."""
    with pytest.raises(ValueError):
        WhileChanging(child_program=_identity_child(), max_iter=0)


def test_max_iter_negative_rejected():
    """Negative max_iter is nonsensical — rejected."""
    with pytest.raises(ValueError):
        WhileChanging(child_program=_identity_child(), max_iter=-3)


def test_max_iter_non_int_rejected():
    """A float max_iter is rejected (only positive int allowed)."""
    with pytest.raises(ValueError):
        WhileChanging(child_program=_identity_child(), max_iter=3.5)


# ---------------------------------------------------------------------------
# 9. to_string display
# ---------------------------------------------------------------------------


def test_to_string_includes_child_and_max_iter():
    """The string display must surface both the child structure and
    the iteration cap so operator-readable program dumps are unambiguous."""
    wc = WhileChanging(child_program=_gravity_child(direction="D"), max_iter=8)
    s = wc.to_string()
    assert "WhileChanging" in s
    assert "Gravity" in s
    assert "max_iter=8" in s
