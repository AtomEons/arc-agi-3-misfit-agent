"""Day-2 DSL AST: typed Program tree with construction-time type checking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from misfit_agent.dsl import (
    DslType, Grid, Color, Number, Object as DslObject, ObjSet,
    TypeMismatchError,
    Identity, Translate, Rotate, Recolor, Crop, CountObj, ShapeOf,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, HoleNode, ConstNode,
    make_hole, make_program,
)


# ---------------------------------------------------------------------------
# Hole / Const node basics
# ---------------------------------------------------------------------------


def test_hole_has_expected_output_type():
    h = make_hole(Grid, hole_id=7)
    assert h.output_type == Grid
    assert h.is_hole()
    assert not h.is_complete()
    assert "Grid" in h.to_string()
    assert "#7" in h.to_string()


def test_const_node_records_typed_literal():
    c = ConstNode(value_type=Number, value=42)
    assert c.output_type == Number
    assert c.is_complete()
    assert c.value == 42


# ---------------------------------------------------------------------------
# Type-correct Program construction
# ---------------------------------------------------------------------------


def test_identity_program_constructs():
    p = make_program(Identity(), make_hole(Grid))
    assert p.depth() == 1
    assert p.output_type() == Grid
    assert not p.is_complete()  # has a hole


def test_translate_with_hole_constructs():
    p = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    assert isinstance(p.root, PrimitiveNode)
    assert p.root.output_type == Grid


def test_nested_program_chains_types_correctly():
    # Identity(Translate(<Grid?>))  — both expect Grid in, return Grid
    inner = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    outer = make_program(Identity(), inner.root)
    assert outer.depth() == 2
    assert outer.output_type() == Grid


def test_count_obj_returns_number_type():
    p = make_program(CountObj(), make_hole(Grid))
    assert p.output_type() == Number


def test_shape_of_takes_object_returns_grid():
    p = make_program(ShapeOf(), make_hole(DslType.OBJECT))
    assert p.output_type() == Grid


# ---------------------------------------------------------------------------
# Type-mismatch errors
# ---------------------------------------------------------------------------


def test_passing_number_where_grid_expected_raises():
    # Translate expects Grid in, but we hand it a hole of type Number
    with pytest.raises(TypeMismatchError) as exc:
        make_program(Translate(dy=1, dx=0), make_hole(Number))
    assert exc.value.expected == Grid
    assert exc.value.actual == Number


def test_passing_object_where_grid_expected_raises():
    with pytest.raises(TypeMismatchError):
        make_program(Identity(), make_hole(DslType.OBJECT))


def test_shape_of_rejects_grid_input():
    # ShapeOf wants Object as input, not Grid
    with pytest.raises(TypeMismatchError):
        make_program(ShapeOf(), make_hole(Grid))


def test_wrong_arity_raises():
    # Identity expects exactly 1 child; pass zero
    with pytest.raises(TypeMismatchError):
        make_program(Identity())


# ---------------------------------------------------------------------------
# Hashing + depth + node count
# ---------------------------------------------------------------------------


def test_two_identical_programs_hash_equal():
    p1 = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    p2 = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    assert p1.sha256_hash() == p2.sha256_hash()


def test_different_parameter_values_hash_differently():
    p1 = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    p2 = make_program(Translate(dy=2, dx=0), make_hole(Grid))
    assert p1.sha256_hash() != p2.sha256_hash()


def test_depth_counts_nesting():
    deep = make_program(
        Identity(),
        PrimitiveNode(
            primitive=Identity(),
            children=[make_hole(Grid)],
        ),
    )
    assert deep.depth() == 2


def test_node_count_includes_holes():
    p = make_program(Identity(), make_hole(Grid))
    # 1 primitive node + 1 hole = 2
    assert p.node_count() == 2


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


def test_to_string_shows_hierarchy():
    p = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    s = p.to_string()
    assert "Translate" in s
    assert "Grid" in s  # hole's type appears
