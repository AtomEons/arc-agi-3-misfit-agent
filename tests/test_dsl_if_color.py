"""IfColor combinator: typed signature + color-conditional branch semantics.

Tests cover:
  - signature_typed() reports Grid → Grid
  - IfColor(c, then, else) on a grid containing color c evaluates the
    then-branch (identity-like child program returns the input unchanged)
  - Same combinator on a grid NOT containing color c evaluates the
    else-branch (rotate-like child program returns the rotated grid)
  - mdl_bits() includes >= 4 bits for the color choice (ARC palette 0..9)
  - mdl_bits() scales with branch program complexity
  - Hash key of the wrapping PrimitiveNode includes the color literal
    and both branch programs (so two IfColors with different colors,
    different then-branches, or different else-branches get different
    hashes — required for memoization correctness)
  - Construction with a non-Grid branch output type raises
    TypeMismatchError BEFORE any program runs
  - Construction with a missing branch raises TypeMismatchError
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Number, DslType, TypeMismatchError,
    Identity, Rotate, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, make_hole, make_program,
)
from misfit_agent.dsl.combinators.if_color import IfColor
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_child() -> Program:
    """Identity(<input>) — a no-op child program."""
    return make_program(Identity(), make_hole(Grid))


def _rotate_child(k: int = 1) -> Program:
    """Rotate(k=k)(<input>) — a rotation child program."""
    return make_program(Rotate(k=k), make_hole(Grid))


def _grid_with_color(color: int) -> np.ndarray:
    """A small grid that definitely contains the requested color."""
    g = np.zeros((3, 3), dtype=np.int32)
    g[1, 1] = color
    g[0, 0] = color
    return g


def _grid_without_color(missing: int) -> np.ndarray:
    """A small grid that definitely does NOT contain `missing`.

    We hand-pick cells whose values avoid the requested color.
    """
    # Build a 3x3 of 1s, then ensure no cell equals `missing`.
    g = np.ones((3, 3), dtype=np.int32)
    if missing == 1:
        # Shift to another palette entry that isn't `missing`.
        g = np.full((3, 3), fill_value=3, dtype=np.int32)
    # Sanity: no cell equals `missing`.
    assert not (g == missing).any(), (
        f"helper failure: built a grid that accidentally contained color "
        f"{missing}:\n{g}"
    )
    return g


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """IfColor declares Grid → Grid with one input slot."""
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    sig = ic.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    # And the type-checked alias matches:
    assert sig.output == DslType.GRID


def test_signature_params_declare_color_and_two_branches():
    """The signature's params list must name the color and both branches."""
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    sig = ic.signature_typed()
    param_names = [n for n, _ in sig.params]
    assert "color" in param_names
    assert "then_program" in param_names
    assert "else_program" in param_names


# ---------------------------------------------------------------------------
# 2. Branch dispatch — then-branch when color present
# ---------------------------------------------------------------------------


def test_then_branch_taken_when_color_present():
    """A grid that contains the test color makes IfColor evaluate the
    then-branch. With Identity as the then-branch, the output equals the
    input."""
    g = _grid_with_color(2)
    assert (g == 2).any(), "fixture failure: grid lost color 2"
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    out = ic.apply(g)
    assert np.array_equal(out, g), (
        f"IfColor(2) with color present should take Identity branch:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_then_branch_via_program_node_round_trip():
    """IfColor is reachable through the typed AST: build a PrimitiveNode
    wrapping IfColor, make a Program out of it, and evaluate against a
    grid containing the test color. Result equals the input (Identity
    branch)."""
    g = _grid_with_color(2)
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    root = PrimitiveNode(primitive=ic, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    assert np.array_equal(out, g)


# ---------------------------------------------------------------------------
# 3. Branch dispatch — else-branch when color absent
# ---------------------------------------------------------------------------


def test_else_branch_taken_when_color_absent():
    """A grid that does NOT contain the test color makes IfColor evaluate
    the else-branch. With Rotate(k=1) as the else-branch, the output
    equals np.rot90 of the input."""
    g = _grid_without_color(2)
    assert not (g == 2).any(), "fixture failure: grid contains color 2"
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    out = ic.apply(g)
    expected = np.rot90(g, k=1)
    assert np.array_equal(out, expected), (
        f"IfColor(2) with color absent should take Rotate(k=1) branch:\n"
        f"IN:\n{g}\nEXPECTED:\n{expected}\nOUT:\n{out}"
    )


def test_else_branch_via_program_node_round_trip():
    """IfColor is reachable through the typed AST and routes the
    else-branch via the interpreter when the color is absent."""
    g = _grid_without_color(2)
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    root = PrimitiveNode(primitive=ic, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    expected = np.rot90(g, k=1)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# 4. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_includes_at_least_four_bits_for_color_choice():
    """The color literal occupies log2(10) ≈ 3.32 bits, rounded up to 4
    in the MDL budget — picking 1 of 10 ARC palette entries is a real
    choice and must be charged for in the prior.

    The cleanest mechanical check: an IfColor with two Identity branches
    must cost at least 4 bits MORE than a bare Identity, because the
    extra is exactly the IfColor wrapper (catalog+meta+color)+branch_costs
    — and the color charge alone is >= 4 bits.
    """
    ic = IfColor(
        color=3,
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    id_only = Identity()
    diff = ic.mdl_bits() - id_only.mdl_bits()
    # Wrapper cost = catalog + 1 (meta) + 4 (color) + 2 * Identity_bits.
    # In particular, the gap is >= 4 bits from the color choice alone,
    # plus at least catalog_bits + 1 + 2*Identity_bits which is strictly
    # positive. So a >= 4 bound is conservative and locks the color
    # charge in place.
    assert diff >= 4.0, (
        f"IfColor MDL did not budget >= 4 bits for the color choice: "
        f"diff={diff} (ic={ic.mdl_bits()}, identity={id_only.mdl_bits()})"
    )


def test_mdl_bits_is_strictly_positive_and_finite():
    """Every primitive costs at least the catalog-encoding bits; IfColor
    additionally adds meta+color+two branch programs. The cost must be a
    finite positive float — an infinite or NaN cost would break MDL
    scoring."""
    ic = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    bits = ic.mdl_bits()
    assert isinstance(bits, float)
    assert bits > 0.0
    assert bits == bits  # not NaN
    assert bits != float("inf")


def test_mdl_bits_scales_with_branch_complexity():
    """An IfColor wrapping a Rotate else-branch costs strictly more bits
    than one wrapping Identity in both branches, because Rotate's k
    parameter adds bits."""
    ic_short = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    ic_long = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    assert ic_long.mdl_bits() > ic_short.mdl_bits(), (
        f"MDL bits did not grow with branch complexity: "
        f"short={ic_short.mdl_bits()} long={ic_long.mdl_bits()}"
    )


# ---------------------------------------------------------------------------
# 5. Hash key — color + both branch programs
# ---------------------------------------------------------------------------


def test_hash_key_distinguishes_different_colors():
    """Two IfColors with different test colors must produce different
    PrimitiveNode hash keys — otherwise memoization tables would collide
    and the synthesis engine would re-use cached results from the wrong
    branch condition."""
    ic_2 = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    ic_5 = IfColor(
        color=5,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    n2 = PrimitiveNode(primitive=ic_2, children=[make_hole(Grid)])
    n5 = PrimitiveNode(primitive=ic_5, children=[make_hole(Grid)])
    assert n2.hash_key() != n5.hash_key(), (
        f"hash_key collision between IfColor(c=2) and IfColor(c=5):\n"
        f"  c=2: {n2.hash_key()}\n  c=5: {n5.hash_key()}"
    )


def test_hash_key_distinguishes_different_then_branches():
    """Two IfColors with the same color but different then-branches must
    produce different hash keys."""
    ic_id = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    ic_rot = IfColor(
        color=2,
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    n_id = PrimitiveNode(primitive=ic_id, children=[make_hole(Grid)])
    n_rot = PrimitiveNode(primitive=ic_rot, children=[make_hole(Grid)])
    assert n_id.hash_key() != n_rot.hash_key(), (
        f"hash_key collision between distinct then-branches:\n"
        f"  identity: {n_id.hash_key()}\n"
        f"  rotate:   {n_rot.hash_key()}"
    )
    # The structural inclusion is real — both child to_strings appear.
    assert "Identity" in n_id.hash_key()
    assert "Rotate" in n_rot.hash_key()


def test_hash_key_distinguishes_different_else_branches():
    """Two IfColors with the same color but different else-branches must
    produce different hash keys."""
    ic_id = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    ic_rot = IfColor(
        color=2,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    n_id = PrimitiveNode(primitive=ic_id, children=[make_hole(Grid)])
    n_rot = PrimitiveNode(primitive=ic_rot, children=[make_hole(Grid)])
    assert n_id.hash_key() != n_rot.hash_key(), (
        f"hash_key collision between distinct else-branches:\n"
        f"  identity: {n_id.hash_key()}\n"
        f"  rotate:   {n_rot.hash_key()}"
    )


def test_hash_key_carries_ifcolor_marker_and_color_literal():
    """The hash key must mention 'IfColor' and the chosen color so
    memoization tables stay disambiguating and human-debuggable."""
    ic = IfColor(
        color=7,
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    node = PrimitiveNode(primitive=ic, children=[make_hole(Grid)])
    key = node.hash_key()
    assert "IfColor" in key, f"hash_key missing IfColor marker: {key}"
    assert "7" in key, f"hash_key missing color literal '7': {key}"
    # And the human-readable form mentions it too.
    assert "IfColor" in node.to_string()
    assert "7" in node.to_string()


# ---------------------------------------------------------------------------
# 6. Type-mismatch on bad branch outputs
# ---------------------------------------------------------------------------


def test_type_mismatch_raised_if_then_branch_output_is_not_grid():
    """A then-branch whose output type is Number (CountObj) must be
    rejected at construction time — synthesis must never enumerate
    an IfColor that tries to dispatch a Number-producing program where
    a Grid is required."""
    bad_branch = make_program(CountObj(), make_hole(Grid))
    assert bad_branch.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        IfColor(
            color=2,
            then_program=bad_branch,
            else_program=_identity_child(),
        )
    # The error message must mention Grid (expected) and Number (actual).
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_else_branch_output_is_not_grid():
    """An else-branch with non-Grid output must also be rejected at
    construction time."""
    bad_branch = make_program(CountObj(), make_hole(Grid))
    assert bad_branch.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        IfColor(
            color=2,
            then_program=_identity_child(),
            else_program=bad_branch,
        )
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_then_branch_missing():
    """No then-branch at all is also a type error — there is no
    default. The synthesis engine must always commit to a specific
    program for both sides of the branch."""
    with pytest.raises(TypeMismatchError):
        IfColor(color=2, then_program=None, else_program=_identity_child())


def test_type_mismatch_raised_if_else_branch_missing():
    """No else-branch at all is also a type error."""
    with pytest.raises(TypeMismatchError):
        IfColor(color=2, then_program=_identity_child(), else_program=None)
