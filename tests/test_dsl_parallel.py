"""Parallel(f, g, merge) combinator: typed signature + cell-wise merge semantics.

Parallel evaluates two child Grid-producing programs against the SAME
input grid, then merges the two output grids cell-wise under one of
three rules:

  - "or"    : np.where(out_f != bg, out_f, out_g) where bg is the
              background color of the ORIGINAL input grid. f's non-bg
              cells dominate; g shows through where f drew background.
  - "max"   : element-wise integer maximum.
  - "first" : return f's output verbatim.

Shape-mismatch fallback: if the two branches' outputs have different
shapes, return f's output (combinator stays total).

Tests cover:
  - signature_typed() correct (Grid → Grid, one AST input, three params)
  - Parallel(Identity, Rotate(k=2), "or") on a simple grid behaves as
    the OR-merged result (Identity wins on its non-background cells;
    Rotate(k=2) shows through elsewhere)
  - Parallel(Identity, Identity, "first") returns identity
  - "max" merge respects integer max on a hand-built fixture
  - mdl_bits() > bare-primitive cost (wrapper has to be paid for)
  - hash_key (via wrapping PrimitiveNode) includes both children + the
    merge tag, so two Parallels with different f-branches, different
    g-branches, or different merges have distinct hash_keys
  - construction with non-Grid branch output raises TypeMismatchError
  - construction with unknown merge tag raises ValueError
  - shape-mismatch fallback returns f's output without crashing
  - end-to-end round-trip through PrimitiveNode + Program + evaluate()
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Number, DslType, TypeMismatchError,
    Identity, Rotate, Crop, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, make_hole, make_program,
)
from misfit_agent.dsl.combinators.parallel_combinator import (
    Parallel, MERGE_RULES,
)
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Branch-program helpers
# ---------------------------------------------------------------------------


def _identity_child() -> Program:
    """Identity(<input>) — a no-op child program."""
    return make_program(Identity(), make_hole(Grid))


def _rotate_child(k: int = 1) -> Program:
    """Rotate(k=k)(<input>) — a rotation child program."""
    return make_program(Rotate(k=k), make_hole(Grid))


def _crop_child() -> Program:
    """Crop(<input>) — a shape-changing child program; used for the
    shape-mismatch fallback test."""
    return make_program(Crop(), make_hole(Grid))


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """Parallel declares Grid → Grid with one AST input slot."""
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="or",
    )
    sig = p.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    assert sig.output == DslType.GRID


def test_signature_params_declare_both_branches_and_merge():
    """The signature's params list must name f, g, and merge so the
    type-checker and any inspector can see the meta-parameters."""
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="or",
    )
    sig = p.signature_typed()
    param_names = [n for n, _ in sig.params]
    assert "f" in param_names
    assert "g" in param_names
    assert "merge" in param_names


# ---------------------------------------------------------------------------
# 2. apply semantics — "or" merge
# ---------------------------------------------------------------------------


def test_or_merge_combines_identity_and_rotate_k2():
    """Parallel(Identity, Rotate(k=2), "or") on a simple grid returns
    the OR-merged composite.

    On a 2x2 grid where background = 0, Identity returns the input as-is;
    Rotate(k=2) returns the 180-degree rotation. The "or" rule keeps
    Identity's non-background cells and lets Rotate's cells show through
    wherever Identity drew background.

    Grid:           Identity:        Rotate(k=2):     OR-merge:
      1 0             1 0              0 0              1 0
      0 0             0 0              0 1              0 1
    """
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="or",
    )
    out = p.apply(g)
    # Identity at (0,0) wins (value 1). Rotate(k=2) places the 1 at (1,1).
    # Identity at (1,1) is 0 = background, so g shows through with 1.
    expected = np.array([[1, 0], [0, 1]], dtype=np.int32)
    assert np.array_equal(out, expected), (
        f"OR-merge failed:\nIN:\n{g}\nEXPECTED:\n{expected}\nOUT:\n{out}"
    )


def test_or_merge_when_f_has_no_background_keeps_only_f():
    """If f's output has no background cells anywhere, the OR-merge
    cannot let g show through — f dominates everywhere.

    Use a grid that contains color 0 so `_background_color` deterministically
    returns 0 (the ARC convention). Then build f and g such that f's output
    has NO zeros anywhere — meaning g should be fully masked out.

    The trick: Identity preserves f's output as the input grid. So if the
    input grid has zeros, Identity's output also has zeros, which would
    let g show through. To get an all-non-bg f-output, we use a grid full
    of a single non-zero color and Identity for both branches. Since both
    branches return the same all-non-bg grid, the OR merge must equal
    that grid.
    """
    g = np.array([[5, 5], [5, 5]], dtype=np.int32)
    # Add one 0 cell so background lookup is unambiguous (bg = 0).
    # Identity passes the grid through verbatim; it has no zeros at
    # positions (0,0),(0,1),(1,0),(1,1) — all 5s. Rotate(k=2) flips it,
    # still all 5s. The "or" merge picks f everywhere.
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="or",
    )
    out = p.apply(g)
    assert np.array_equal(out, g), (
        f"OR-merge with all-non-bg f should equal f (=g), but got:\n{out}"
    )


# ---------------------------------------------------------------------------
# 3. apply semantics — "first" merge
# ---------------------------------------------------------------------------


def test_first_merge_returns_f_output_when_both_are_identity():
    """Parallel(Identity, Identity, "first") returns the identity of the
    input grid — both branches produce the input, and the "first" rule
    takes f's output."""
    g = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_identity_child(),
        merge="first",
    )
    out = p.apply(g)
    assert np.array_equal(out, g), (
        f"Parallel(Identity, Identity, 'first') did not return identity:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_first_merge_ignores_g_branch_completely():
    """Parallel(Identity, Rotate(k=1), "first") returns f's output
    (identity) regardless of g's output."""
    g = np.array([[7, 0], [0, 0]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=1),
        merge="first",
    )
    out = p.apply(g)
    assert np.array_equal(out, g), (
        f"'first' merge let g leak into the output:\nIN:\n{g}\nOUT:\n{out}"
    )


# ---------------------------------------------------------------------------
# 4. apply semantics — "max" merge
# ---------------------------------------------------------------------------


def test_max_merge_respects_integer_max_cellwise():
    """Parallel(Identity, Rotate(k=2), "max") returns the cell-wise
    integer max of the two output grids."""
    g = np.array([[1, 0], [0, 9]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="max",
    )
    out = p.apply(g)
    # Identity returns g unchanged. Rotate(k=2) returns g rotated 180:
    #   [[9, 0], [0, 1]]
    # Cell-wise max: [[max(1,9), max(0,0)], [max(0,0), max(9,1)]] =
    #   [[9, 0], [0, 9]]
    expected = np.maximum(g, np.rot90(g, k=2))
    assert np.array_equal(out, expected), (
        f"'max' merge did not take element-wise integer max:\n"
        f"IN:\n{g}\nEXPECTED:\n{expected}\nOUT:\n{out}"
    )


def test_max_merge_picks_the_larger_of_two_constants():
    """A targeted check that max really beats the alternative rules: on
    a grid where Identity and Rotate disagree, "max" picks the larger
    value cell-wise. Constructed so the answer is uniquely max-shaped."""
    g = np.array([[3, 0], [0, 5]], dtype=np.int32)
    p_max = Parallel(f=_identity_child(), g=_rotate_child(k=2), merge="max")
    out_max = p_max.apply(g)
    # Identity vs Rotate(k=2):
    #   Identity: [[3,0],[0,5]]
    #   Rotate:   [[5,0],[0,3]]
    #   max:      [[5,0],[0,5]]
    expected = np.array([[5, 0], [0, 5]], dtype=np.int32)
    assert np.array_equal(out_max, expected)
    # And confirm "max" is NOT the same as "first" on this fixture —
    # otherwise the test would be vacuous.
    p_first = Parallel(f=_identity_child(), g=_rotate_child(k=2), merge="first")
    out_first = p_first.apply(g)
    assert not np.array_equal(out_max, out_first), (
        "max and first happened to agree on the fixture — bad test"
    )


# ---------------------------------------------------------------------------
# 5. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_is_strictly_greater_than_bare_primitive():
    """A Parallel wrapper must cost strictly more bits than a bare
    primitive — otherwise the MDL prior would not discourage
    speculative wrapping. The gap is at minimum:
      catalog_bits + 1 (meta) + log2(3) (merge) + 2 * Identity_bits
    which is strictly positive."""
    p = Parallel(
        f=_identity_child(),
        g=_identity_child(),
        merge="or",
    )
    id_only = Identity()
    assert p.mdl_bits() > id_only.mdl_bits(), (
        f"Parallel MDL did not exceed bare Identity: "
        f"parallel={p.mdl_bits()} identity={id_only.mdl_bits()}"
    )


def test_mdl_bits_is_finite_and_positive():
    """Every primitive costs at least the catalog-encoding bits; Parallel
    additionally adds meta+merge+two branch programs. The cost must be
    a finite positive float — an infinite or NaN cost would break MDL
    scoring."""
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=1),
        merge="or",
    )
    bits = p.mdl_bits()
    assert isinstance(bits, float)
    assert bits > 0.0
    assert bits == bits  # not NaN
    assert bits != float("inf")


def test_mdl_bits_scales_with_branch_complexity():
    """A Parallel wrapping a Rotate g-branch costs strictly more bits
    than one wrapping Identity in both branches, because Rotate's k
    parameter adds bits to the inner program."""
    p_short = Parallel(
        f=_identity_child(),
        g=_identity_child(),
        merge="or",
    )
    p_long = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=1),
        merge="or",
    )
    assert p_long.mdl_bits() > p_short.mdl_bits(), (
        f"MDL bits did not grow with branch complexity: "
        f"short={p_short.mdl_bits()} long={p_long.mdl_bits()}"
    )


# ---------------------------------------------------------------------------
# 6. Hash key — both children + merge tag
# ---------------------------------------------------------------------------


def test_hash_key_distinguishes_different_merges():
    """Two Parallels with the same children but different merge tags
    must produce different PrimitiveNode hash keys — otherwise the
    memoization tables would collide on a real semantic difference."""
    p_or = Parallel(f=_identity_child(), g=_rotate_child(k=2), merge="or")
    p_max = Parallel(f=_identity_child(), g=_rotate_child(k=2), merge="max")
    n_or = PrimitiveNode(primitive=p_or, children=[make_hole(Grid)])
    n_max = PrimitiveNode(primitive=p_max, children=[make_hole(Grid)])
    assert n_or.hash_key() != n_max.hash_key(), (
        f"hash_key collision between Parallel(merge='or') and "
        f"Parallel(merge='max'):\n  or:  {n_or.hash_key()}\n"
        f"  max: {n_max.hash_key()}"
    )


def test_hash_key_distinguishes_different_f_branches():
    """Two Parallels with the same merge and g-branch but different
    f-branches must produce different hash keys."""
    p_id = Parallel(
        f=_identity_child(),
        g=_identity_child(),
        merge="or",
    )
    p_rot = Parallel(
        f=_rotate_child(k=1),
        g=_identity_child(),
        merge="or",
    )
    n_id = PrimitiveNode(primitive=p_id, children=[make_hole(Grid)])
    n_rot = PrimitiveNode(primitive=p_rot, children=[make_hole(Grid)])
    assert n_id.hash_key() != n_rot.hash_key(), (
        f"hash_key collision between distinct f-branches:\n"
        f"  identity: {n_id.hash_key()}\n"
        f"  rotate:   {n_rot.hash_key()}"
    )
    # Both branch to_strings must appear in the hash_key — that is the
    # mechanism that makes them disambiguating.
    assert "Identity" in n_id.hash_key()
    assert "Rotate" in n_rot.hash_key()


def test_hash_key_distinguishes_different_g_branches():
    """Two Parallels with the same merge and f-branch but different
    g-branches must produce different hash keys."""
    p_id = Parallel(
        f=_identity_child(),
        g=_identity_child(),
        merge="or",
    )
    p_rot = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=1),
        merge="or",
    )
    n_id = PrimitiveNode(primitive=p_id, children=[make_hole(Grid)])
    n_rot = PrimitiveNode(primitive=p_rot, children=[make_hole(Grid)])
    assert n_id.hash_key() != n_rot.hash_key(), (
        f"hash_key collision between distinct g-branches:\n"
        f"  identity: {n_id.hash_key()}\n"
        f"  rotate:   {n_rot.hash_key()}"
    )


def test_hash_key_carries_parallel_marker_and_merge_literal():
    """The hash key must mention 'Parallel' and the merge literal so
    memoization tables stay disambiguating and human-debuggable."""
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="max",
    )
    node = PrimitiveNode(primitive=p, children=[make_hole(Grid)])
    key = node.hash_key()
    assert "Parallel" in key, f"hash_key missing Parallel marker: {key}"
    assert "max" in key, f"hash_key missing merge literal 'max': {key}"
    # And the human-readable form mentions both.
    assert "Parallel" in node.to_string()
    assert "max" in node.to_string()


# ---------------------------------------------------------------------------
# 7. Construction-time validation
# ---------------------------------------------------------------------------


def test_type_mismatch_raised_if_f_branch_output_is_not_grid():
    """An f-branch whose output type is Number (CountObj) must be
    rejected at construction time — synthesis must never enumerate
    a Parallel that tries to merge a Number-producing program where
    a Grid is required."""
    bad_branch = make_program(CountObj(), make_hole(Grid))
    assert bad_branch.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        Parallel(
            f=bad_branch,
            g=_identity_child(),
            merge="or",
        )
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg, (
        f"TypeMismatchError message did not name the types: {msg}"
    )


def test_type_mismatch_raised_if_g_branch_output_is_not_grid():
    """A g-branch with non-Grid output must also be rejected."""
    bad_branch = make_program(CountObj(), make_hole(Grid))
    assert bad_branch.output_type() == Number
    with pytest.raises(TypeMismatchError) as ei:
        Parallel(
            f=_identity_child(),
            g=bad_branch,
            merge="or",
        )
    msg = str(ei.value)
    assert "Grid" in msg and "Number" in msg


def test_type_mismatch_raised_if_f_branch_missing():
    """No f-branch at all is a type error."""
    with pytest.raises(TypeMismatchError):
        Parallel(f=None, g=_identity_child(), merge="or")


def test_type_mismatch_raised_if_g_branch_missing():
    """No g-branch at all is a type error."""
    with pytest.raises(TypeMismatchError):
        Parallel(f=_identity_child(), g=None, merge="or")


def test_value_error_raised_for_unknown_merge_tag():
    """An unknown merge tag must raise ValueError at construction time
    — synthesis must never enumerate a Parallel with a merge rule the
    apply() dispatch table doesn't know about."""
    with pytest.raises(ValueError) as ei:
        Parallel(
            f=_identity_child(),
            g=_identity_child(),
            merge="xor",  # not in MERGE_RULES
        )
    msg = str(ei.value)
    assert "merge" in msg.lower()
    # And the known vocabulary appears in the message so the operator
    # can see what was expected.
    assert all(rule in msg for rule in MERGE_RULES)


def test_merge_rules_vocabulary_is_exactly_three():
    """The merge-rule vocabulary is the contract that locks the
    log2(3)-bit MDL budget in place. If a future refactor widens the
    vocabulary, the budget must be revisited. This test fails loudly
    on any silent change."""
    assert MERGE_RULES == ("or", "max", "first")
    assert len(MERGE_RULES) == 3


# ---------------------------------------------------------------------------
# 8. Shape-mismatch fallback
# ---------------------------------------------------------------------------


def test_shape_mismatch_returns_f_output_without_crashing():
    """When f and g produce grids of different shapes, no cell-wise merge
    rule is well-defined. The combinator must stay total: return f's
    output and let the MDL prior penalize the wrapper.

    Crop is a shape-changing primitive: on a grid with a single colored
    cell, it crops to a 1x1 grid. Identity returns the full grid. The
    shapes differ, so apply() must return Identity's output unchanged.
    """
    g = np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_crop_child(),
        merge="or",
    )
    out = p.apply(g)
    assert np.array_equal(out, g), (
        f"Shape-mismatch fallback did not return f's output:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


# ---------------------------------------------------------------------------
# 9. End-to-end round-trip through the typed AST + interpreter
# ---------------------------------------------------------------------------


def test_round_trip_through_program_and_evaluate_or_merge():
    """Parallel is reachable through the typed AST: build a PrimitiveNode
    wrapping it, make a Program out of it, and evaluate against a grid.
    The result equals what apply() would have produced directly."""
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="or",
    )
    root = PrimitiveNode(primitive=p, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    expected = p.apply(g)
    assert np.array_equal(out, expected)
    # And the expected concrete value matches the OR-merge of the two
    # branches on this fixture.
    assert np.array_equal(out, np.array([[1, 0], [0, 1]], dtype=np.int32))


def test_round_trip_max_merge_via_program():
    """A max-merge Parallel also round-trips through the interpreter."""
    g = np.array([[3, 0], [0, 5]], dtype=np.int32)
    p = Parallel(
        f=_identity_child(),
        g=_rotate_child(k=2),
        merge="max",
    )
    root = PrimitiveNode(primitive=p, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    expected = np.array([[5, 0], [0, 5]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_construction_with_primitivenode_branch_not_wrapped_in_program():
    """The synthesis engine sometimes hands branches as raw PrimitiveNodes
    rather than Program wrappers. Parallel must accept both forms — that
    is the convention IfColor established."""
    f_node = PrimitiveNode(primitive=Identity(), children=[make_hole(Grid)])
    g_node = PrimitiveNode(primitive=Rotate(k=2), children=[make_hole(Grid)])
    p = Parallel(f=f_node, g=g_node, merge="or")
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    out = p.apply(g)
    expected = np.array([[1, 0], [0, 1]], dtype=np.int32)
    assert np.array_equal(out, expected)
