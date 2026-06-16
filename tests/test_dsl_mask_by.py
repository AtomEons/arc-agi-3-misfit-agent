"""MaskBy combinator: typed signature + masked-region semantics.

Tests cover:
  - signature_typed() reports Grid → Grid with predicate + child_program params
  - foreground mask + Rotate child rotates the FULL grid but the
    np.where compose keeps only the rotated-foreground-cell values
    inside the original foreground mask, restoring the background
    elsewhere (verifies the masking contract even though "rotate the
    foreground" doesn't geometrically make sense in isolation)
  - background mask + Recolor child recolors only the background
    (foreground untouched)
  - edge_touching mask + Recolor child recolors only edge-touching
    object cells
  - largest_object mask isolates the largest object's cells
  - mdl_bits() strictly greater than a bare primitive's mdl_bits
  - Hash key of the wrapping PrimitiveNode includes the predicate label
    AND the child program's structure (so MaskBys with the same
    predicate but different children, or the same child but different
    predicates, get different hashes — required for memoization
    correctness)
  - Construction with a bad predicate raises ValueError
  - Construction with a non-Grid child output type raises
    TypeMismatchError BEFORE any program runs
  - Construction with a missing child program raises TypeMismatchError
  - Empty scene (no objects) leaves grid unchanged for object-based
    predicates
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Number, DslType, TypeMismatchError,
    Identity, Rotate, Recolor, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, make_hole, make_program,
)
from misfit_agent.dsl.combinators.mask_by import (
    MaskBy, ALLOWED_PREDICATES, _compute_mask,
)
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_child() -> Program:
    """Identity(<input>) — a no-op child program."""
    return make_program(Identity(), make_hole(Grid))


def _rotate_child(k: int = 1) -> Program:
    """Rotate(k=k)(<input>) — child program that rotates the full grid."""
    return make_program(Rotate(k=k), make_hole(Grid))


def _recolor_child(mapping: dict) -> Program:
    """Recolor(mapping)(<input>) — child program that swaps colors."""
    return make_program(Recolor(mapping=mapping), make_hole(Grid))


def _foreground_grid() -> np.ndarray:
    """A 4x4 grid with a clear foreground / background split.

    Layout (0 = bg):
        0 0 0 0
        0 1 1 0
        0 1 1 0
        0 0 0 0
    """
    g = np.zeros((4, 4), dtype=np.int32)
    g[1:3, 1:3] = 1
    return g


def _edge_and_interior_grid() -> np.ndarray:
    """A 5x5 grid with one edge-touching object and one interior object.

    Layout (0 = bg):
        1 . . . .
        . . . . .
        . . 2 . .
        . . . . .
        . . . . .

    Two perceived objects:
      - color 1 at (0,0) — touches edge (top-left corner)
      - color 2 at (2,2) — interior, does not touch edge
    """
    g = np.zeros((5, 5), dtype=np.int32)
    g[0, 0] = 1
    g[2, 2] = 2
    return g


def _two_size_grid() -> np.ndarray:
    """A 5x5 grid with a large object (4 cells) and a small object (1 cell).

    Layout (0 = bg):
        1 1 . . .
        1 1 . . .
        . . . . .
        . . . . 2
        . . . . .

    Two perceived objects:
      - color 1, 2x2 block at rows 0-1, cols 0-1, area = 4 (LARGEST)
      - color 2, single cell at (3,4), area = 1
    """
    g = np.zeros((5, 5), dtype=np.int32)
    g[0:2, 0:2] = 1
    g[3, 4] = 2
    return g


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """MaskBy(predicate, child) declares Grid → Grid with one input slot
    and predicate + child_program meta-params."""
    mb = MaskBy(predicate="foreground", child_program=_identity_child())
    sig = mb.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    assert sig.output == DslType.GRID
    # The params declare predicate (str) and child_program (object).
    param_names = [name for name, _ in sig.params]
    assert "predicate" in param_names
    assert "child_program" in param_names


def test_allowed_predicates_is_exactly_four():
    """The synthesis enumerator depends on the predicate set being
    exactly 4 labels so the 2-bit MDL charge is tight."""
    assert tuple(ALLOWED_PREDICATES) == (
        "foreground", "background", "edge_touching", "largest_object"
    )
    assert len(ALLOWED_PREDICATES) == 4


# ---------------------------------------------------------------------------
# 2. Foreground mask + Rotate child → restores background, masks rotated region
# ---------------------------------------------------------------------------


def test_foreground_mask_with_rotate_child_restores_background():
    """MaskBy(foreground, Rotate(k=2)) on a centered 2x2 block produces
    an output where the BACKGROUND cells equal the original background
    (untouched by the rotation) and the FOREGROUND cells take values
    from the rotated grid at the same positions.

    For the 4x4 _foreground_grid(), the centered 2x2 block is invariant
    under rot180 (it stays a centered 2x2 block of color 1), so the
    visible output equals the input. This verifies the masking contract:
    even though the rotation is global, the np.where compose restores
    the background exactly."""
    g = _foreground_grid()
    mb = MaskBy(predicate="foreground", child_program=_rotate_child(k=2))
    out = mb.apply(g)
    # Background restored exactly.
    fg_mask = g != 0
    bg_mask = ~fg_mask
    assert (out[bg_mask] == 0).all(), (
        f"MaskBy(foreground, Rotate) bled into background:\n{out}"
    )
    # Foreground cells came from the rotated grid at those positions.
    rotated = np.rot90(g, k=2)
    assert np.array_equal(out[fg_mask], rotated[fg_mask]), (
        f"MaskBy(foreground, Rotate) did not stamp rotated foreground:\n"
        f"IN:\n{g}\nROT:\n{rotated}\nOUT:\n{out}"
    )


def test_foreground_mask_with_identity_child_is_noop():
    """MaskBy(foreground, Identity) is a global no-op: both branches of
    np.where give back the original cell value. This is the canonical
    smoke test for the np.where compose contract."""
    g = _foreground_grid()
    mb = MaskBy(predicate="foreground", child_program=_identity_child())
    out = mb.apply(g)
    assert np.array_equal(out, g), (
        f"MaskBy(foreground, Identity) changed the grid:\nIN:\n{g}\nOUT:\n{out}"
    )


# ---------------------------------------------------------------------------
# 3. Background mask + Recolor child → recolors only background
# ---------------------------------------------------------------------------


def test_background_mask_with_recolor_child_recolors_only_background():
    """MaskBy(background, Recolor({0: 5})) sets every background cell to
    5 and leaves every foreground cell intact. The foreground 2x2 block
    of color 1 stays color 1."""
    g = _foreground_grid()
    mb = MaskBy(
        predicate="background",
        child_program=_recolor_child(mapping={0: 5}),
    )
    out = mb.apply(g)
    # Foreground untouched.
    fg_mask = g == 1
    assert (out[fg_mask] == 1).all(), (
        f"MaskBy(background, Recolor) bled into foreground:\n{out}"
    )
    # Background recolored to 5.
    bg_mask = g == 0
    assert (out[bg_mask] == 5).all(), (
        f"MaskBy(background, Recolor) failed to recolor background:\n{out}"
    )


def test_background_mask_with_recolor_does_not_touch_foreground_color():
    """Even if the child Recolor includes a foreground color in its
    mapping, the mask restricts the effect to background cells. Recolor
    runs on the full grid (turning the 1s into 7s in `modified`), but
    np.where with the background mask discards those changes — the
    final output keeps 1s where the input had 1s."""
    g = _foreground_grid()
    mb = MaskBy(
        predicate="background",
        child_program=_recolor_child(mapping={0: 5, 1: 7}),
    )
    out = mb.apply(g)
    # Foreground cells still 1, not 7.
    assert (out[g == 1] == 1).all(), (
        f"MaskBy(background) leaked Recolor into foreground:\n{out}"
    )
    # Background cells became 5, not 7.
    assert (out[g == 0] == 5).all()


# ---------------------------------------------------------------------------
# 4. edge_touching mask + Recolor child → recolors only edge-touching object
# ---------------------------------------------------------------------------


def test_edge_touching_mask_targets_edge_objects_only():
    """MaskBy(edge_touching, Recolor({1: 9, 2: 9})) on the edge+interior
    grid recolors the corner '1' object (touches edge) but leaves the
    interior '2' object untouched."""
    g = _edge_and_interior_grid()
    mb = MaskBy(
        predicate="edge_touching",
        child_program=_recolor_child(mapping={1: 9, 2: 9}),
    )
    out = mb.apply(g)
    # The edge-touching '1' at (0,0) becomes 9.
    assert out[0, 0] == 9, (
        f"MaskBy(edge_touching, Recolor) missed corner cell:\n{out}"
    )
    # The interior '2' at (2,2) stays 2.
    assert out[2, 2] == 2, (
        f"MaskBy(edge_touching, Recolor) leaked into interior object:\n{out}"
    )
    # Background unchanged.
    bg_positions = (g == 0)
    assert (out[bg_positions] == 0).all(), (
        f"MaskBy(edge_touching, Recolor) bled into background:\n{out}"
    )


# ---------------------------------------------------------------------------
# 5. largest_object mask isolates the largest object's cells
# ---------------------------------------------------------------------------


def test_largest_object_mask_targets_only_the_largest():
    """MaskBy(largest_object, Recolor({1: 8, 2: 8})) on the
    _two_size_grid recolors the 2x2 block (largest, area=4) but leaves
    the single-cell '2' (area=1) untouched."""
    g = _two_size_grid()
    mb = MaskBy(
        predicate="largest_object",
        child_program=_recolor_child(mapping={1: 8, 2: 8}),
    )
    out = mb.apply(g)
    # The 2x2 block becomes 8 everywhere.
    assert (out[0:2, 0:2] == 8).all(), (
        f"MaskBy(largest_object, Recolor) missed largest object cells:\n{out}"
    )
    # The lone '2' at (3,4) stays 2.
    assert out[3, 4] == 2, (
        f"MaskBy(largest_object, Recolor) leaked into smaller object:\n{out}"
    )


def test_compute_mask_largest_object_matches_perceived_area():
    """The mask for largest_object covers exactly area-many cells when
    the largest object is unambiguous. For _two_size_grid that's 4."""
    g = _two_size_grid()
    mb = MaskBy(predicate="largest_object", child_program=_identity_child())
    mask = mb.compute_mask(g)
    assert mask.dtype == bool
    assert mask.shape == g.shape
    assert int(mask.sum()) == 4, (
        f"largest_object mask did not cover 4 cells; sum={mask.sum()}, "
        f"mask=\n{mask.astype(int)}"
    )


# ---------------------------------------------------------------------------
# 6. Empty-scene fallback for object-based predicates
# ---------------------------------------------------------------------------


def test_largest_object_mask_on_empty_grid_is_all_false():
    """An empty (all-background) grid has no perceived objects; the
    largest_object mask must be all-False and the apply() result must
    equal the input."""
    g = np.zeros((3, 3), dtype=np.int32)
    mb = MaskBy(
        predicate="largest_object",
        child_program=_recolor_child(mapping={0: 9}),
    )
    mask = mb.compute_mask(g)
    assert mask.dtype == bool
    assert not mask.any()
    out = mb.apply(g)
    assert np.array_equal(out, g), (
        f"empty-scene largest_object should be no-op; got:\n{out}"
    )


# ---------------------------------------------------------------------------
# 7. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_is_strictly_positive():
    """Every primitive costs at least the catalog-encoding bits; MaskBy
    adds meta-shape + predicate + child program bits."""
    mb = MaskBy(predicate="foreground", child_program=_identity_child())
    assert mb.mdl_bits() > 0.0


def test_mdl_bits_strictly_greater_than_bare_child():
    """A MaskBy wrapping a child must cost strictly more bits than the
    child's bare primitive cost: catalog + 1 (shape) + 2 (predicate) on
    top of the child's contribution."""
    bare_identity = Identity()
    bare_bits = bare_identity.mdl_bits()
    mb = MaskBy(predicate="foreground", child_program=_identity_child())
    assert mb.mdl_bits() > bare_bits, (
        f"MaskBy mdl_bits ({mb.mdl_bits()}) did not exceed bare Identity "
        f"mdl_bits ({bare_bits})"
    )


def test_mdl_bits_scales_with_child_complexity():
    """MaskBy wrapping a Rotate child costs strictly more bits than one
    wrapping Identity, because Rotate's k parameter adds bits."""
    mb_short = MaskBy(predicate="foreground", child_program=_identity_child())
    mb_long = MaskBy(predicate="foreground", child_program=_rotate_child(k=1))
    assert mb_long.mdl_bits() > mb_short.mdl_bits(), (
        f"MDL bits did not grow with child complexity: "
        f"short={mb_short.mdl_bits()} long={mb_long.mdl_bits()}"
    )


# ---------------------------------------------------------------------------
# 8. Hash key includes predicate AND child program
# ---------------------------------------------------------------------------


def test_hash_key_distinguishes_predicates():
    """Two MaskBys with the same child but different predicates must
    produce different PrimitiveNode hash keys — otherwise memoization
    tables would collide and the synthesis engine would re-use cached
    results from the wrong region."""
    mb_fg = MaskBy(predicate="foreground", child_program=_identity_child())
    mb_bg = MaskBy(predicate="background", child_program=_identity_child())
    node_fg = PrimitiveNode(primitive=mb_fg, children=[make_hole(Grid)])
    node_bg = PrimitiveNode(primitive=mb_bg, children=[make_hole(Grid)])
    assert node_fg.hash_key() != node_bg.hash_key(), (
        f"hash_key collision between predicates:\n"
        f"  fg: {node_fg.hash_key()}\n"
        f"  bg: {node_bg.hash_key()}"
    )
    # The predicate label appears inside the hash key — confirming the
    # inclusion is structural, not accidental.
    assert "foreground" in node_fg.hash_key()
    assert "background" in node_bg.hash_key()


def test_hash_key_distinguishes_child_programs():
    """Two MaskBys with the same predicate but different child programs
    must produce different hash keys."""
    mb_id = MaskBy(predicate="foreground", child_program=_identity_child())
    mb_rot = MaskBy(predicate="foreground", child_program=_rotate_child(k=1))
    node_id = PrimitiveNode(primitive=mb_id, children=[make_hole(Grid)])
    node_rot = PrimitiveNode(primitive=mb_rot, children=[make_hole(Grid)])
    assert node_id.hash_key() != node_rot.hash_key(), (
        f"hash_key collision between MaskBy children:\n"
        f"  identity: {node_id.hash_key()}\n"
        f"  rotate:   {node_rot.hash_key()}"
    )
    assert "Identity" in node_id.hash_key()
    assert "Rotate" in node_rot.hash_key()


def test_hash_key_distinguishes_child_parameters():
    """Even within the same child primitive family, different parameter
    values must yield different hash keys."""
    mb_k1 = MaskBy(predicate="foreground", child_program=_rotate_child(k=1))
    mb_k2 = MaskBy(predicate="foreground", child_program=_rotate_child(k=2))
    n1 = PrimitiveNode(primitive=mb_k1, children=[make_hole(Grid)])
    n2 = PrimitiveNode(primitive=mb_k2, children=[make_hole(Grid)])
    assert n1.hash_key() != n2.hash_key(), (
        f"hash_key collision between Rotate(k=1) and Rotate(k=2) "
        f"children:\n  k=1: {n1.hash_key()}\n  k=2: {n2.hash_key()}"
    )


# ---------------------------------------------------------------------------
# 9. Construction-time validation
# ---------------------------------------------------------------------------


def test_bad_predicate_raises_value_error():
    """A predicate label outside the allowed set is a synthesizer
    authoring error and must be rejected at construction time."""
    with pytest.raises(ValueError) as ei:
        MaskBy(predicate="not_a_real_predicate",
               child_program=_identity_child())
    msg = str(ei.value)
    assert "predicate" in msg.lower()


def test_type_mismatch_raised_if_child_output_is_not_grid():
    """A child program whose output type is Number (CountObj) must be
    rejected at construction time — synthesis must never enumerate a
    MaskBy that tries to mask a number into a grid."""
    bad_child = make_program(CountObj(), make_hole(Grid))
    assert bad_child.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        MaskBy(predicate="foreground", child_program=bad_child)
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_child_program_missing():
    """No child program at all is also a type error — there is no
    default. The synthesis engine must always commit to a specific
    child."""
    with pytest.raises(TypeMismatchError):
        MaskBy(predicate="foreground", child_program=None)


# ---------------------------------------------------------------------------
# 10. End-to-end through the typed AST
# ---------------------------------------------------------------------------


def test_round_trip_through_ast_and_interpreter():
    """MaskBy is reachable through the typed AST: build a PrimitiveNode
    wrapping MaskBy(background, Recolor), make a Program out of it, and
    evaluate against a grid. Result equals the direct .apply()."""
    g = _foreground_grid()
    mb = MaskBy(
        predicate="background",
        child_program=_recolor_child(mapping={0: 3}),
    )
    direct = mb.apply(g)

    root = PrimitiveNode(primitive=mb, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    assert np.array_equal(out, direct), (
        f"AST round-trip diverged from direct apply:\n"
        f"DIRECT:\n{direct}\nAST:\n{out}"
    )


def test_compute_mask_foreground_matches_simple_predicate():
    """The foreground mask under the standard ARC background rule
    (0 if present, else most-frequent) must equal `grid != 0` on a
    grid where 0 is present."""
    g = _foreground_grid()
    mask = _compute_mask(g, "foreground")
    assert np.array_equal(mask, g != 0)


def test_compute_mask_background_is_complement_of_foreground():
    """The background mask must be the elementwise complement of the
    foreground mask for any grid."""
    g = _edge_and_interior_grid()
    fg = _compute_mask(g, "foreground")
    bg = _compute_mask(g, "background")
    assert np.array_equal(bg, ~fg)
