"""ForEachObject combinator: typed signature + per-object map semantics.

Tests cover:
  - signature_typed() reports Grid → Grid
  - Identity child program behaves as a no-op on a multi-object grid
  - Rotate(k=1) child program rotates each object's shape in place
  - mdl_bits() is strictly positive (every node costs bits)
  - Hash key of the wrapping PrimitiveNode includes the child program's
    structure (so two ForEachObjects with different children get
    different hashes — required for memoization correctness)
  - Construction with a non-Grid child output type raises
    TypeMismatchError BEFORE any program runs
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
from misfit_agent.dsl.combinators.foreach_object import ForEachObject
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_child() -> Program:
    """Identity(<input>) — a no-op child program."""
    return make_program(Identity(), make_hole(Grid))


def _rotate_child(k: int = 1) -> Program:
    """Rotate(k=k)(<input>) — a per-object rotation child program."""
    return make_program(Rotate(k=k), make_hole(Grid))


def _multi_object_grid() -> np.ndarray:
    """A 5x5 grid with two clearly-separated single-cell objects.

    Layout (0 = bg):
        . 1 . . .
        . . . . .
        . . . 2 .
        . . . . .
        . . . . .

    Two perceived objects: a '1' at (0,1) and a '2' at (2,3).
    Single-cell objects are robust against any rotation (1x1 is invariant
    under rot90) — this lets the rotate-each test focus on stamp position,
    not orientation.
    """
    g = np.zeros((5, 5), dtype=np.int32)
    g[0, 1] = 1
    g[2, 3] = 2
    return g


def _two_rect_object_grid() -> np.ndarray:
    """A 6x6 grid with two distinguishable 2x3 rectangles of different
    color/orientation, so rotation produces a SHAPE-stable but
    position-preserving result we can assert on.

    Layout (0 = bg):
        1 1 1 . . .
        1 1 1 . . .
        . . . . . .
        . . . . . .
        . . . 2 2 2
        . . . 2 2 2

    Each rectangle is 2 rows x 3 cols. Rot90 of a 2x3 is a 3x2 — but the
    bbox is 2x3, so stamping uses the overlap region (2x2). For the
    Identity case, the stamp is identical to the source — exactly a
    no-op.
    """
    g = np.zeros((6, 6), dtype=np.int32)
    g[0:2, 0:3] = 1
    g[4:6, 3:6] = 2
    return g


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """ForEachObject(f) declares Grid → Grid with one input slot."""
    feo = ForEachObject(child_program=_identity_child())
    sig = feo.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    # And the type-checked alias matches:
    assert sig.output == DslType.GRID


# ---------------------------------------------------------------------------
# 2. Identity child program → no-op on multi-object grid
# ---------------------------------------------------------------------------


def test_identity_child_is_noop_on_multi_object_grid():
    """ForEachObject(Identity) applied to a multi-object grid returns
    a grid equal to the input. The combinator perceives every object,
    runs Identity on each tiny grid, and stamps each unchanged tiny
    back where it came from. Net effect: no change."""
    g = _two_rect_object_grid()
    feo = ForEachObject(child_program=_identity_child())
    out = feo.apply(g)
    assert np.array_equal(out, g), (
        f"ForEachObject(Identity) changed the grid:\nIN:\n{g}\nOUT:\n{out}"
    )


def test_identity_child_via_program_node_round_trip():
    """ForEachObject is reachable through the typed AST: build a
    PrimitiveNode wrapping ForEachObject(Identity), make a Program out
    of it, and evaluate against a multi-object grid. Result equals input."""
    g = _multi_object_grid()
    feo = ForEachObject(child_program=_identity_child())
    root = PrimitiveNode(primitive=feo, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    assert np.array_equal(out, g)


# ---------------------------------------------------------------------------
# 3. Rotate(k=1) child program → rotates each object in place
# ---------------------------------------------------------------------------


def test_rotate_child_rotates_each_1x1_object_in_place():
    """For 1x1 objects, rot90 is invariant — the objects stay in their
    exact positions. This isolates the "stamping returns objects to
    their bboxes" contract from the orientation-change contract."""
    g = _multi_object_grid()
    feo = ForEachObject(child_program=_rotate_child(k=1))
    out = feo.apply(g)
    # Both 1x1 objects unchanged after rotation; positions preserved.
    assert out[0, 1] == 1, f"object '1' lost its position; out=\n{out}"
    assert out[2, 3] == 2, f"object '2' lost its position; out=\n{out}"
    # Background untouched everywhere else.
    bg_mask = np.ones_like(g, dtype=bool)
    bg_mask[0, 1] = False
    bg_mask[2, 3] = False
    assert (out[bg_mask] == 0).all(), (
        f"ForEachObject(Rotate) bled foreground into background:\n{out}"
    )


def test_rotate_child_applied_to_each_object_bbox_individually():
    """Build a grid with one 2x3 rectangle of color 1. Rotating the
    extracted tiny by 90° gives a 3x2 rectangle of color 1; since the
    rectangle is solid (no shape change other than dimensions), the
    overlap stamp in the original 2x3 bbox keeps the color, so the
    output bbox is still solid color-1. This verifies the per-object
    pipeline runs without raising and produces a sensible result even
    when the child changes tiny-grid shape."""
    g = np.zeros((4, 4), dtype=np.int32)
    g[0:2, 0:3] = 1  # a single 2x3 rectangle of color 1
    feo = ForEachObject(child_program=_rotate_child(k=1))
    out = feo.apply(g)
    # The 2x3 bbox should still be filled with color 1 (rotated solid
    # block is still a solid block; overlap stamping preserves color).
    assert (out[0:2, 0:2] == 1).all(), (
        f"ForEachObject(Rotate) failed to stamp into bbox overlap:\n{out}"
    )
    # And the rest of the grid is background.
    untouched = np.zeros_like(g, dtype=bool)
    untouched[2:, :] = True
    untouched[:, 3:] = True
    assert (out[untouched] == 0).all(), (
        f"ForEachObject(Rotate) bled outside the bbox:\n{out}"
    )


# ---------------------------------------------------------------------------
# 4. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_is_strictly_positive():
    """Every primitive costs at least the catalog-encoding bits;
    ForEachObject adds a meta-shape bit plus its child program's bits."""
    feo = ForEachObject(child_program=_identity_child())
    assert feo.mdl_bits() > 0.0


def test_mdl_bits_scales_with_child_complexity():
    """A ForEachObject wrapping a Rotate child costs strictly more bits
    than one wrapping Identity, because Rotate's k parameter adds bits."""
    feo_short = ForEachObject(child_program=_identity_child())
    feo_long = ForEachObject(child_program=_rotate_child(k=1))
    assert feo_long.mdl_bits() > feo_short.mdl_bits(), (
        f"MDL bits did not grow with child complexity: "
        f"short={feo_short.mdl_bits()} long={feo_long.mdl_bits()}"
    )


# ---------------------------------------------------------------------------
# 5. Hash key includes the child program's hash
# ---------------------------------------------------------------------------


def test_hash_key_includes_child_program_signature():
    """Two ForEachObjects with different child programs must produce
    different PrimitiveNode hash keys — otherwise memoization tables
    would collide and the synthesis engine would re-use cached results
    from the wrong child program."""
    feo_id = ForEachObject(child_program=_identity_child())
    feo_rot = ForEachObject(child_program=_rotate_child(k=1))
    node_id = PrimitiveNode(primitive=feo_id, children=[make_hole(Grid)])
    node_rot = PrimitiveNode(primitive=feo_rot, children=[make_hole(Grid)])
    assert node_id.hash_key() != node_rot.hash_key(), (
        f"hash_key collision between distinct ForEachObject children:\n"
        f"  identity: {node_id.hash_key()}\n"
        f"  rotate:   {node_rot.hash_key()}"
    )
    # And the child's to_string() shows up inside the hash key —
    # confirming the inclusion is structural, not accidental.
    assert "Identity" in node_id.hash_key()
    assert "Rotate" in node_rot.hash_key()


def test_hash_key_distinguishes_rotate_parameters():
    """Even within the same child primitive family, different
    parameter values must yield different hash keys."""
    feo_k1 = ForEachObject(child_program=_rotate_child(k=1))
    feo_k2 = ForEachObject(child_program=_rotate_child(k=2))
    n1 = PrimitiveNode(primitive=feo_k1, children=[make_hole(Grid)])
    n2 = PrimitiveNode(primitive=feo_k2, children=[make_hole(Grid)])
    assert n1.hash_key() != n2.hash_key(), (
        f"hash_key collision between Rotate(k=1) and Rotate(k=2) "
        f"children:\n  k=1: {n1.hash_key()}\n  k=2: {n2.hash_key()}"
    )


# ---------------------------------------------------------------------------
# 6. Type-mismatch on bad child output
# ---------------------------------------------------------------------------


def test_type_mismatch_raised_if_child_output_is_not_grid():
    """A child program whose output type is Number (CountObj) must be
    rejected at construction time — synthesis must never enumerate
    a ForEachObject that tries to stamp a number back into a grid."""
    bad_child = make_program(CountObj(), make_hole(Grid))
    assert bad_child.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        ForEachObject(child_program=bad_child)
    # The error message must mention Grid (expected) and Number (actual).
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_child_program_missing():
    """No child program at all is also a type error — there is no
    default. The synthesis engine must always commit to a specific
    child."""
    with pytest.raises(TypeMismatchError):
        ForEachObject(child_program=None)
