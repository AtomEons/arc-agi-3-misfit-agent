"""IfShape combinator: typed signature + shape-conditional branch semantics.

Tests cover:
  - signature_typed() reports Grid → Grid with one input slot
  - IfShape((2,2), Rotate(k=1), Identity) on a grid that contains a 2x2
    object evaluates the then-branch (Rotate, not Identity)
  - Same combinator on a grid whose objects are ONLY 3x3 evaluates the
    else-branch (Identity returns input unchanged)
  - mdl_bits is strictly greater than a bare Identity (the combinator
    overhead — meta-shape + target_shape + two branches — must cost
    real bits in the prior)
  - mdl_bits scales with branch program complexity
  - Hash key of the wrapping PrimitiveNode includes target_shape AND
    both child program hashes (different target_shape, different
    then-branch, OR different else-branch ⇒ different hash key)
  - Construction with a non-Grid branch output type raises
    TypeMismatchError BEFORE any program runs
  - Construction with a missing branch is also a type error
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path — tests run from repo root with `python -m pytest`.
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
from misfit_agent.dsl.combinators.if_shape import IfShape
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


def _grid_with_2x2_object() -> np.ndarray:
    """A 6x6 background-0 grid with a single 2x2 colored object.

    The perceptor flood-fills connected non-background cells. A 2x2 block
    of the same color is one object whose bbox dims are exactly (2, 2).
    """
    g = np.zeros((6, 6), dtype=np.int32)
    g[1:3, 1:3] = 4  # 2x2 block of color 4
    return g


def _grid_with_only_3x3_objects() -> np.ndarray:
    """A 7x7 background-0 grid containing only a 3x3 object.

    No object in this grid has bbox shape (2, 2) — the only foreground
    object is a 3x3 solid block at rows 1..3, cols 1..3.
    """
    g = np.zeros((7, 7), dtype=np.int32)
    g[1:4, 1:4] = 5  # 3x3 block of color 5
    return g


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """IfShape declares Grid → Grid with one input slot."""
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    sig = ish.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    # And the type-checked alias matches:
    assert sig.output == DslType.GRID


def test_signature_params_declare_target_shape_and_two_branches():
    """The signature's params list must name target_shape and both branches."""
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    sig = ish.signature_typed()
    param_names = [n for n, _ in sig.params]
    assert "target_shape" in param_names
    assert "then_program" in param_names
    assert "else_program" in param_names


# ---------------------------------------------------------------------------
# 2. Branch dispatch — then-branch when target shape is present
# ---------------------------------------------------------------------------


def test_then_branch_taken_when_target_shape_present():
    """A grid containing a 2x2 object makes IfShape((2,2), Rotate, Identity)
    take the Rotate branch — output should equal np.rot90 of the input."""
    g = _grid_with_2x2_object()
    # Sanity-check the fixture: the perceptor really does see a 2x2 obj.
    from misfit_agent.perceptor import perceive_grid
    bboxes = [o.bbox for o in perceive_grid(g).objects]
    shapes = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in bboxes]
    assert (2, 2) in shapes, (
        f"fixture failure: no 2x2 object in grid:\n{g}\nshapes={shapes}"
    )

    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    out = ish.apply(g)
    expected = np.rot90(g, k=1)
    assert np.array_equal(out, expected), (
        f"IfShape((2,2)) with 2x2 object present should take Rotate(k=1)"
        f" branch:\nIN:\n{g}\nEXPECTED:\n{expected}\nGOT:\n{out}"
    )


def test_then_branch_via_program_node_round_trip():
    """IfShape is reachable through the typed AST: wrap in a PrimitiveNode,
    build a Program, evaluate via the interpreter. With a 2x2 object the
    Rotate branch fires."""
    g = _grid_with_2x2_object()
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    root = PrimitiveNode(primitive=ish, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    expected = np.rot90(g, k=1)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# 3. Branch dispatch — else-branch when target shape is absent
# ---------------------------------------------------------------------------


def test_else_branch_taken_when_target_shape_absent():
    """A grid whose objects are ONLY 3x3 takes the else-branch (Identity),
    so the output equals the input."""
    g = _grid_with_only_3x3_objects()
    # Sanity-check the fixture: no 2x2 object in this grid.
    from misfit_agent.perceptor import perceive_grid
    bboxes = [o.bbox for o in perceive_grid(g).objects]
    shapes = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in bboxes]
    assert (2, 2) not in shapes, (
        f"fixture failure: 2x2 object snuck into 3x3-only grid:\n{g}\n"
        f"shapes={shapes}"
    )

    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    out = ish.apply(g)
    assert np.array_equal(out, g), (
        f"IfShape((2,2)) with no 2x2 object should take Identity branch:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_else_branch_via_program_node_round_trip():
    """IfShape with no matching object routes through the interpreter and
    returns the Identity-branch output (the input unchanged)."""
    g = _grid_with_only_3x3_objects()
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    root = PrimitiveNode(primitive=ish, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    assert np.array_equal(out, g)


# ---------------------------------------------------------------------------
# 4. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_greater_than_bare_identity():
    """An IfShape with two Identity branches must cost strictly more bits
    than a bare Identity — the combinator wrapper (meta-shape + target
    shape + two branch programs) must be charged for in the prior.
    """
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    id_only = Identity()
    assert ish.mdl_bits() > id_only.mdl_bits(), (
        f"IfShape MDL did not exceed bare Identity: "
        f"ish={ish.mdl_bits()}, identity={id_only.mdl_bits()}"
    )


def test_mdl_bits_is_strictly_positive_and_finite():
    """Every primitive costs at least the catalog-encoding bits; IfShape
    additionally adds meta+target_shape+two branch programs. The cost
    must be a finite positive float — an infinite or NaN cost would
    break MDL scoring."""
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    bits = ish.mdl_bits()
    assert isinstance(bits, float)
    assert bits > 0.0
    assert bits == bits  # not NaN
    assert bits != float("inf")


def test_mdl_bits_scales_with_branch_complexity():
    """An IfShape wrapping a Rotate else-branch costs strictly more bits
    than one wrapping Identity in both branches, because Rotate's k
    parameter adds bits."""
    ish_short = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    ish_long = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    assert ish_long.mdl_bits() > ish_short.mdl_bits(), (
        f"MDL bits did not grow with branch complexity: "
        f"short={ish_short.mdl_bits()} long={ish_long.mdl_bits()}"
    )


# ---------------------------------------------------------------------------
# 5. Hash key — target_shape + both branch programs
# ---------------------------------------------------------------------------


def test_hash_key_distinguishes_different_target_shapes():
    """Two IfShapes with different target shapes must produce different
    PrimitiveNode hash keys — otherwise memoization tables would collide
    across distinct branch conditions."""
    ish_22 = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    ish_33 = IfShape(
        target_shape=(3, 3),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    n22 = PrimitiveNode(primitive=ish_22, children=[make_hole(Grid)])
    n33 = PrimitiveNode(primitive=ish_33, children=[make_hole(Grid)])
    assert n22.hash_key() != n33.hash_key(), (
        f"hash_key collision between IfShape((2,2)) and IfShape((3,3)):\n"
        f"  (2,2): {n22.hash_key()}\n  (3,3): {n33.hash_key()}"
    )


def test_hash_key_distinguishes_different_then_branches():
    """Two IfShapes with the same target_shape but different then-branches
    must produce different hash keys."""
    ish_id = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    ish_rot = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    n_id = PrimitiveNode(primitive=ish_id, children=[make_hole(Grid)])
    n_rot = PrimitiveNode(primitive=ish_rot, children=[make_hole(Grid)])
    assert n_id.hash_key() != n_rot.hash_key(), (
        f"hash_key collision between distinct then-branches:\n"
        f"  identity: {n_id.hash_key()}\n"
        f"  rotate:   {n_rot.hash_key()}"
    )
    # The structural inclusion is real — both child to_strings appear.
    assert "Identity" in n_id.hash_key()
    assert "Rotate" in n_rot.hash_key()


def test_hash_key_distinguishes_different_else_branches():
    """Two IfShapes with the same target_shape but different else-branches
    must produce different hash keys."""
    ish_id = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_identity_child(),
    )
    ish_rot = IfShape(
        target_shape=(2, 2),
        then_program=_identity_child(),
        else_program=_rotate_child(k=1),
    )
    n_id = PrimitiveNode(primitive=ish_id, children=[make_hole(Grid)])
    n_rot = PrimitiveNode(primitive=ish_rot, children=[make_hole(Grid)])
    assert n_id.hash_key() != n_rot.hash_key(), (
        f"hash_key collision between distinct else-branches:\n"
        f"  identity: {n_id.hash_key()}\n"
        f"  rotate:   {n_rot.hash_key()}"
    )


def test_hash_key_carries_ifshape_marker_and_shape_literal():
    """The hash key must mention 'IfShape' and the target shape numerals
    so memoization tables stay disambiguating and human-debuggable."""
    ish = IfShape(
        target_shape=(2, 2),
        then_program=_rotate_child(k=1),
        else_program=_identity_child(),
    )
    node = PrimitiveNode(primitive=ish, children=[make_hole(Grid)])
    key = node.hash_key()
    assert "IfShape" in key, f"hash_key missing IfShape marker: {key}"
    assert "2" in key, f"hash_key missing target shape literal '2': {key}"
    # And the human-readable form mentions it too.
    assert "IfShape" in node.to_string()
    assert "2" in node.to_string()


# ---------------------------------------------------------------------------
# 6. Type-mismatch on bad branch outputs
# ---------------------------------------------------------------------------


def test_type_mismatch_raised_if_then_branch_output_is_not_grid():
    """A then-branch whose output type is Number (CountObj) must be
    rejected at construction time — synthesis must never enumerate an
    IfShape that tries to dispatch a Number-producing program where a
    Grid is required."""
    bad_branch = make_program(CountObj(), make_hole(Grid))
    assert bad_branch.output_type() == Number  # sanity-check the fixture
    with pytest.raises(TypeMismatchError) as ei:
        IfShape(
            target_shape=(2, 2),
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
    assert bad_branch.output_type() == Number  # sanity-check the fixture
    with pytest.raises(TypeMismatchError) as ei:
        IfShape(
            target_shape=(2, 2),
            then_program=_identity_child(),
            else_program=bad_branch,
        )
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_then_branch_missing():
    """No then-branch at all is also a type error — there is no default.
    The synthesis engine must always commit to a specific program for
    both sides of the branch."""
    with pytest.raises(TypeMismatchError):
        IfShape(
            target_shape=(2, 2),
            then_program=None,
            else_program=_identity_child(),
        )


def test_type_mismatch_raised_if_else_branch_missing():
    """No else-branch at all is also a type error."""
    with pytest.raises(TypeMismatchError):
        IfShape(
            target_shape=(2, 2),
            then_program=_identity_child(),
            else_program=None,
        )
