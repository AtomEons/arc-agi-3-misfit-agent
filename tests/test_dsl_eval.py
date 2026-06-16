"""Day-3 DSL: evaluator + walker."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    DslType, Grid, Number,
    Identity, Translate, Rotate, Reflect, Recolor, Crop, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, HoleNode, ConstNode,
    make_hole, make_program,
)
from misfit_agent.dsl.interpreter import evaluate, IncompleteProgramError
from misfit_agent.dsl.walker import (
    walk_preorder, walk_postorder, find_holes,
    count_primitives, total_mdl_bits, visit,
)


# ---------------------------------------------------------------------------
# Evaluator basics
# ---------------------------------------------------------------------------


def test_identity_program_evaluates_to_input():
    g = np.array([[0, 1], [2, 3]], dtype=np.int32)
    p = make_program(Identity(), make_hole(Grid))
    out = evaluate(p, g)
    assert np.array_equal(out, g)


def test_translate_program_evaluates():
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    p = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    out = evaluate(p, g)
    expected = np.array([[0, 0], [1, 0]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_nested_program_evaluates_correctly():
    # Identity(Translate(<input>))  — should equal Translate alone
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    inner = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    outer_root = PrimitiveNode(primitive=Identity(), children=[inner.root])
    outer = Program(root=outer_root, desired_output=Grid)
    out = evaluate(outer, g)
    expected = np.array([[0, 0], [1, 0]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_count_obj_returns_int():
    g = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 0]], dtype=np.int32)
    p = make_program(CountObj(), make_hole(Grid))
    n = evaluate(p, g)
    assert isinstance(n, int)
    assert n == 3


def test_incomplete_program_raises_on_evaluate():
    """If a hole isn't bound to an input, evaluate raises."""
    p = make_program(Identity(), make_hole(Grid))
    g = np.array([[1]], dtype=np.int32)
    # Hole binds to the input, so this succeeds:
    out = evaluate(p, g)
    assert np.array_equal(out, g)
    # But evaluating with no input leaves the hole unbound:
    with pytest.raises(IncompleteProgramError):
        evaluate(p)


# ---------------------------------------------------------------------------
# Walker basics
# ---------------------------------------------------------------------------


def test_walk_preorder_yields_root_first():
    p = make_program(Identity(), make_hole(Grid))
    nodes = list(walk_preorder(p))
    assert len(nodes) == 2
    # First node is the PrimitiveNode (root)
    assert isinstance(nodes[0], PrimitiveNode)
    # Second is the hole child
    assert isinstance(nodes[1], HoleNode)


def test_walk_postorder_yields_root_last():
    p = make_program(Identity(), make_hole(Grid))
    nodes = list(walk_postorder(p))
    assert len(nodes) == 2
    assert isinstance(nodes[0], HoleNode)
    assert isinstance(nodes[-1], PrimitiveNode)


def test_find_holes_returns_all_open_slots():
    p = make_program(Identity(), make_hole(Grid, hole_id=5))
    holes = find_holes(p)
    assert len(holes) == 1
    assert holes[0].hole_id == 5


def test_find_holes_in_nested_program_returns_all():
    inner = make_program(Translate(dy=1, dx=0), make_hole(Grid, hole_id=1))
    outer_root = PrimitiveNode(primitive=Identity(), children=[inner.root])
    outer = Program(root=outer_root, desired_output=Grid)
    holes = find_holes(outer)
    assert len(holes) == 1
    assert holes[0].hole_id == 1


def test_count_primitives_excludes_holes():
    p = make_program(Identity(), make_hole(Grid))
    assert count_primitives(p) == 1


def test_total_mdl_bits_sums_per_node():
    p_short = make_program(Identity(), make_hole(Grid))
    p_long = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    # Translate has 2 integer parameters → costs more bits than Identity
    assert total_mdl_bits(p_long) > total_mdl_bits(p_short)


def test_visit_collects_per_node_results():
    p = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    types_seen = visit(p, lambda n: n.output_type)
    # Both root and hole are Grid-typed
    assert types_seen == [Grid, Grid]


# ---------------------------------------------------------------------------
# Compositional evaluation correctness
# ---------------------------------------------------------------------------


def test_rotate_compose_translate_executes_left_to_right():
    """Verify a 2-level program: Rotate(Translate(input))."""
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    inner = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    outer_root = PrimitiveNode(primitive=Rotate(k=1), children=[inner.root])
    outer = Program(root=outer_root, desired_output=Grid)
    out = evaluate(outer, g)
    # Translate dy=1,dx=0 sends 1 from (0,0) to (1,0) → [[0,0],[1,0]]
    # Rotate k=1: 90° CCW → np.rot90 → [[0,0],[0,1]]
    expected = np.rot90(np.array([[0, 0], [1, 0]], dtype=np.int32))
    assert np.array_equal(out, expected)


def test_reflect_compose_recolor_executes():
    g = np.array([[1, 2], [0, 0]], dtype=np.int32)
    inner = make_program(Recolor(mapping={1: 5}), make_hole(Grid))
    outer_root = PrimitiveNode(primitive=Reflect(axis="H"),
                               children=[inner.root])
    outer = Program(root=outer_root, desired_output=Grid)
    out = evaluate(outer, g)
    # Recolor 1→5: [[5,2],[0,0]]
    # Reflect H (fliplr): [[2,5],[0,0]]
    expected = np.array([[2, 5], [0, 0]], dtype=np.int32)
    assert np.array_equal(out, expected)
