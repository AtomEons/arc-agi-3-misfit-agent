"""Day-4 DSL: Seq(f, g) sequencing combinator.

Covers:
  - Signature shape: two Grid-typed children, Grid output, no scalar params
  - apply(x, y) returns y (the second child's value)
  - mdl_bits() >= 1 (real composition choice)
  - End-to-end evaluation: Seq(Identity_child, Rotate(k=1)_child) produces
    the rotated grid (g's effect on the initial input)
  - TypeMismatchError when a Number-typed child is passed where Grid expected
  - Hash key for a Seq node is distinct from any single-primitive node's
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Number,
    TypeMismatchError,
    Identity, Rotate, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, HoleNode,
    make_hole, make_program,
)
from misfit_agent.dsl.combinators import Seq
from misfit_agent.dsl.combinators.seq import Seq as SeqDirect
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


def test_seq_signature_shape():
    """Seq declares two Grid-typed inputs and a Grid output, no scalar params."""
    sig = Seq().signature_typed()
    assert sig.output == Grid
    assert len(sig.inputs) == 2
    names = [n for n, _ in sig.inputs]
    types = [t for _, t in sig.inputs]
    assert types == [Grid, Grid]
    # Both inputs are named (whatever the names are, they are non-empty
    # strings and they are not duplicates — synthesis prints these in
    # error messages).
    assert all(isinstance(n, str) and n for n in names)
    assert len(set(names)) == 2
    assert sig.params == ()


def test_seq_import_paths_are_consistent():
    """Public re-export from `combinators/__init__.py` matches the module class."""
    assert Seq is SeqDirect


# ---------------------------------------------------------------------------
# apply semantics
# ---------------------------------------------------------------------------


def test_seq_apply_returns_second_value():
    """Seq.apply(x, y) == y for any pair of Grid-shaped numpy arrays."""
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[9, 8], [7, 6]], dtype=np.int32)
    out = Seq().apply(x, y)
    assert np.array_equal(out, y)
    # Identity check — Seq doesn't have to copy; it just forwards.
    # We don't assert identity, only value equality, because future
    # implementations may choose to copy for safety.


def test_seq_apply_returns_second_value_ignores_first_completely():
    """Even when the two children produce wildly different grids, Seq returns y."""
    x = np.zeros((5, 5), dtype=np.int32)
    y = np.full((2, 3), 7, dtype=np.int32)
    out = Seq().apply(x, y)
    assert np.array_equal(out, y)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# MDL cost
# ---------------------------------------------------------------------------


def test_seq_mdl_bits_is_at_least_one_bit():
    """Picking Seq is a real composition choice — at least 1 bit of code length."""
    bits = Seq().mdl_bits()
    assert isinstance(bits, float)
    assert bits >= 1.0


def test_seq_mdl_bits_is_finite():
    """The cost must be finite; an infinite cost would break MDL scoring."""
    bits = Seq().mdl_bits()
    assert bits == bits  # not NaN
    assert bits != float("inf")


# ---------------------------------------------------------------------------
# Type-checking at construction time
# ---------------------------------------------------------------------------


def test_seq_rejects_number_typed_child_where_grid_expected():
    """A Number-typed hole as the first child must raise TypeMismatchError."""
    with pytest.raises(TypeMismatchError) as exc:
        PrimitiveNode(
            primitive=Seq(),
            children=[make_hole(Number, hole_id=0), make_hole(Grid, hole_id=1)],
        )
    assert exc.value.expected == Grid
    assert exc.value.actual == Number


def test_seq_rejects_number_typed_second_child():
    """A Number-typed hole as the second child must also raise."""
    with pytest.raises(TypeMismatchError):
        PrimitiveNode(
            primitive=Seq(),
            children=[make_hole(Grid, hole_id=0), make_hole(Number, hole_id=1)],
        )


def test_seq_rejects_number_producing_primitive_child():
    """A real primitive that outputs Number (CountObj) must also be rejected."""
    count_child = PrimitiveNode(
        primitive=CountObj(),
        children=[make_hole(Grid, hole_id=99)],
    )
    with pytest.raises(TypeMismatchError):
        PrimitiveNode(
            primitive=Seq(),
            children=[make_hole(Grid, hole_id=0), count_child],
        )


def test_seq_rejects_wrong_arity():
    """Seq needs exactly 2 children — zero or one must raise."""
    with pytest.raises(TypeMismatchError):
        PrimitiveNode(primitive=Seq(), children=[])
    with pytest.raises(TypeMismatchError):
        PrimitiveNode(primitive=Seq(), children=[make_hole(Grid, hole_id=0)])


def test_seq_accepts_two_grid_holes():
    """The canonical synthesis shape: Seq with two Grid holes constructs OK."""
    node = PrimitiveNode(
        primitive=Seq(),
        children=[make_hole(Grid, hole_id=0), make_hole(Grid, hole_id=1)],
    )
    assert node.output_type == Grid
    assert len(node.children) == 2
    assert not node.is_complete()  # both children are holes


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------


def test_two_level_seq_identity_then_rotate_evaluates_correctly():
    """Program: Seq(Identity(<grid?>), Rotate(k=1)(<grid?>)) executes.

    Both children consume the initial-input stream left-to-right; the
    interpreter pops one input per leaf hole. We pass the same grid twice
    so both children see it. Seq returns the second child's value =
    Rotate(k=1).apply(grid).
    """
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    seq_root = PrimitiveNode(
        primitive=Seq(),
        children=[
            PrimitiveNode(primitive=Identity(), children=[make_hole(Grid, hole_id=0)]),
            PrimitiveNode(primitive=Rotate(k=1), children=[make_hole(Grid, hole_id=1)]),
        ],
    )
    prog = Program(root=seq_root, desired_output=Grid)
    out = evaluate(prog, g, g)
    expected = np.rot90(g, k=1)
    assert np.array_equal(out, expected)


def test_seq_program_completeness_and_shape_metadata():
    """A Seq program with two atomic children has depth 2 and 5 nodes."""
    seq_root = PrimitiveNode(
        primitive=Seq(),
        children=[
            PrimitiveNode(primitive=Identity(), children=[make_hole(Grid, hole_id=0)]),
            PrimitiveNode(primitive=Rotate(k=2), children=[make_hole(Grid, hole_id=1)]),
        ],
    )
    prog = Program(root=seq_root, desired_output=Grid)
    # depth: Seq(level 1) > Identity/Rotate(level 2) > hole; depth counts
    # PrimitiveNode levels only.
    assert prog.depth() == 2
    # node_count: 1 Seq + 2 primitives + 2 holes = 5
    assert prog.node_count() == 5
    assert prog.output_type() == Grid
    assert not prog.is_complete()  # holes remain


# ---------------------------------------------------------------------------
# Hash distinctness
# ---------------------------------------------------------------------------


def test_seq_hash_key_distinct_from_any_single_primitive_node():
    """A Seq-rooted program's hash must not collide with a single-primitive program."""
    seq_node = PrimitiveNode(
        primitive=Seq(),
        children=[make_hole(Grid, hole_id=0), make_hole(Grid, hole_id=1)],
    )
    seq_prog = Program(root=seq_node, desired_output=Grid)

    identity_prog = make_program(Identity(), make_hole(Grid, hole_id=0))
    rotate_prog = make_program(Rotate(k=1), make_hole(Grid, hole_id=0))

    assert seq_prog.hash_key() != identity_prog.hash_key()
    assert seq_prog.hash_key() != rotate_prog.hash_key()
    assert seq_prog.sha256_hash() != identity_prog.sha256_hash()
    assert seq_prog.sha256_hash() != rotate_prog.sha256_hash()


def test_seq_hash_key_carries_seq_marker():
    """The hash key must mention 'Seq' so memoization tables stay disambiguating."""
    seq_node = PrimitiveNode(
        primitive=Seq(),
        children=[make_hole(Grid, hole_id=0), make_hole(Grid, hole_id=1)],
    )
    seq_prog = Program(root=seq_node, desired_output=Grid)
    assert "Seq" in seq_prog.hash_key()
    assert "Seq" in seq_prog.to_string()


def test_seq_hash_distinct_between_different_grid_children():
    """Two Seq programs with different children must hash differently."""
    prog_a = Program(
        root=PrimitiveNode(
            primitive=Seq(),
            children=[
                PrimitiveNode(primitive=Identity(), children=[make_hole(Grid, 0)]),
                PrimitiveNode(primitive=Rotate(k=1), children=[make_hole(Grid, 1)]),
            ],
        ),
        desired_output=Grid,
    )
    prog_b = Program(
        root=PrimitiveNode(
            primitive=Seq(),
            children=[
                PrimitiveNode(primitive=Identity(), children=[make_hole(Grid, 0)]),
                PrimitiveNode(primitive=Rotate(k=2), children=[make_hole(Grid, 1)]),
            ],
        ),
        desired_output=Grid,
    )
    assert prog_a.sha256_hash() != prog_b.sha256_hash()
