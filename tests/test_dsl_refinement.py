"""TEAM REFINEMENT — HRM-style outer refinement loop over typed Programs.

Required guarantees mechanically tested here:
  - refine on a perfect program returns the same program unchanged
  - refine on a wrong Translate(dy=1, dx=0) when the correct answer is
    Translate(dy=2, dx=0) eventually finds the right offset (or a
    reasonable improvement)
  - refine respects max_iters (a 0-iter call cannot edit anything)
  - refine never returns a program that scores LOWER than the input
  - hash of the refined program differs from the input hash when any
    edit was actually applied

Additional coverage that earns its place:
  - swap_primitive returns a program with a different hash and a different
    root primitive type
  - swap_primitive raises TypeMismatchError when the new primitive's
    typed signature doesn't fit the children
  - wrap_program adds exactly one level to the AST depth
  - wrap_program raises TypeMismatchError when the wrapper expects a
    different input type than the current root produces
  - mutate_param produces a different program hash on a real change
  - mutate_param raises AttributeError on an undeclared parameter name
  - mutate_param on an out-of-range index raises IndexError
  - refine on an incorrect Rotate(k=1) when the right answer is
    Rotate(k=2) finds the right k
  - refine on a wrong Reflect axis finds the right axis
  - the input program is never mutated by refine (deep-copy contract)
  - refine returns the input when train_pairs is empty
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import copy

import numpy as np
import pytest

from misfit_agent.dsl import Grid, DslType, TypeMismatchError
from misfit_agent.dsl.primitives import (
    Identity, Translate, Rotate, Reflect, Recolor, CountObj,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, HoleNode, make_hole, make_program,
)
from misfit_agent.dsl.interpreter import evaluate
from misfit_agent.dsl.refinement import (
    refine,
    swap_primitive,
    wrap_program,
    mutate_param,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_program() -> Program:
    return make_program(Identity(), make_hole(Grid))


def _translate_program(dy: int, dx: int) -> Program:
    return make_program(Translate(dy=dy, dx=dx), make_hole(Grid))


def _rotate_program(k: int) -> Program:
    return make_program(Rotate(k=k), make_hole(Grid))


def _reflect_program(axis: str) -> Program:
    return make_program(Reflect(axis=axis), make_hole(Grid))


def _score(program: Program,
           train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mirror the refinement scorer for assertions (cell-accuracy mean)."""
    if not train_pairs:
        return 0.0
    acc_sum = 0.0
    for x, y in train_pairs:
        try:
            pred = evaluate(program, x)
        except Exception:
            return 0.0
        if not isinstance(pred, np.ndarray) or pred.shape != y.shape:
            continue
        acc_sum += float(np.sum(pred == y)) / y.size
    return acc_sum / len(train_pairs)


# ---------------------------------------------------------------------------
# Core required guarantees
# ---------------------------------------------------------------------------


def test_perfect_program_returned_unchanged():
    """An identity-task input + Identity program is already perfect; refine
    must return the same Program object (no edits, no work)."""
    g = np.array([[1, 2, 0], [0, 3, 4]], dtype=np.int32)
    program = _identity_program()
    train_pairs = [(g, g.copy())]
    refined = refine(program, train_pairs, max_iters=4)

    # Score didn't change, hash didn't change.
    assert refined.sha256_hash() == program.sha256_hash()
    out = evaluate(refined, g)
    assert np.array_equal(out, g)


def test_wrong_translate_dy_refined_toward_correct():
    """Start with Translate(dy=1, dx=0); correct answer is Translate(dy=2,
    dx=0). Refinement should find the right offset or at least improve
    monotonically."""
    g = np.array(
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]], dtype=np.int32,
    )
    correct = Translate(dy=2, dx=0).apply(g)
    train_pairs = [(g, correct)]

    wrong = _translate_program(dy=1, dx=0)
    wrong_score = _score(wrong, train_pairs)

    refined = refine(wrong, train_pairs, max_iters=4)
    refined_score = _score(refined, train_pairs)

    # Refinement must IMPROVE the program.
    assert refined_score >= wrong_score
    # And in this particular case, the dy/dx grid sweep should reach the
    # perfect program in a single iteration.
    assert refined_score == pytest.approx(1.0)

    # The refined root should be Translate(dy=2, dx=0).
    root_prim = refined.root.primitive
    assert isinstance(root_prim, Translate)
    assert root_prim.dy == 2
    assert root_prim.dx == 0


def test_refine_respects_max_iters_zero():
    """With max_iters=0 no edits may be tried; output == input."""
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    correct = Translate(dy=1, dx=0).apply(g)
    train_pairs = [(g, correct)]

    program = _identity_program()
    refined = refine(program, train_pairs, max_iters=0)
    # Hash must match — zero edits possible.
    assert refined.sha256_hash() == program.sha256_hash()


def test_refine_respects_max_iters_one():
    """max_iters=1 caps at one round of edits."""
    g = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32)
    correct = Translate(dy=1, dx=2).apply(g)
    train_pairs = [(g, correct)]

    wrong = _translate_program(dy=0, dx=0)
    # 1 iter is enough for a single-edit fix on this task.
    refined = refine(wrong, train_pairs, max_iters=1)
    refined_score = _score(refined, train_pairs)
    wrong_score = _score(wrong, train_pairs)
    assert refined_score >= wrong_score


def test_refine_never_returns_lower_scoring_program():
    """Across a basket of mismatched tasks, refine must never decrease score."""
    rng = np.random.default_rng(seed=42)

    test_grids = [
        np.array([[1, 0], [0, 2]], dtype=np.int32),
        np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]], dtype=np.int32),
        rng.integers(0, 10, size=(3, 3), dtype=np.int32),
        rng.integers(0, 10, size=(4, 4), dtype=np.int32),
    ]
    # A handful of (correct_op, starting_op) pairs:
    cases = [
        (Translate(dy=1, dx=0), Translate(dy=0, dx=0)),
        (Rotate(k=2), Rotate(k=1)),
        (Reflect(axis="V"), Reflect(axis="H")),
        (Translate(dy=0, dx=1), Identity()),
    ]

    for correct_op, start_op in cases:
        for g in test_grids:
            try:
                target = correct_op.apply(g)
            except Exception:
                continue
            train_pairs = [(g, target)]
            wrong = make_program(start_op, make_hole(Grid))
            before = _score(wrong, train_pairs)
            refined = refine(wrong, train_pairs, max_iters=3)
            after = _score(refined, train_pairs)
            assert after >= before, (
                f"refine decreased score for correct={correct_op} "
                f"start={start_op}: {before} → {after}"
            )


def test_refined_hash_differs_when_edit_made():
    """If refinement actually applies an edit, the program hash must change."""
    g = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32)
    correct = Translate(dy=1, dx=0).apply(g)
    train_pairs = [(g, correct)]

    wrong = _translate_program(dy=0, dx=0)
    refined = refine(wrong, train_pairs, max_iters=4)
    # We know refinement must make an edit here (start is wrong, target is
    # within the dy/dx grid).
    assert refined.sha256_hash() != wrong.sha256_hash()


# ---------------------------------------------------------------------------
# Additional coverage: edit operations
# ---------------------------------------------------------------------------


def test_swap_primitive_changes_root():
    """swap_primitive at idx=0 replaces the root primitive while keeping
    children."""
    program = _translate_program(dy=1, dx=0)
    edited = swap_primitive(program, target_idx=0,
                            new_primitive=Identity())
    assert isinstance(edited.root.primitive, Identity)
    assert edited.sha256_hash() != program.sha256_hash()
    # Children preserved.
    assert len(edited.root.children) == 1
    assert isinstance(edited.root.children[0], HoleNode)


def test_swap_primitive_type_mismatch_raises():
    """Swapping a Grid-out primitive for a Number-out primitive at a
    non-root position would break a parent's child type — but at the root,
    the program's desired_output changes and the swap succeeds. Test the
    real type-mismatch path: swap a single-child Grid-in primitive with
    one whose input type does not match the existing child."""
    # Inner program with a Translate root, child is a Grid hole.
    program = _translate_program(dy=1, dx=0)
    # ShapeOf expects an Object input, but the child is a Grid hole.
    from misfit_agent.dsl.primitives import ShapeOf
    with pytest.raises(TypeMismatchError):
        swap_primitive(program, target_idx=0, new_primitive=ShapeOf())


def test_swap_primitive_out_of_range_index_raises():
    program = _translate_program(dy=1, dx=0)
    with pytest.raises(IndexError):
        swap_primitive(program, target_idx=99, new_primitive=Identity())


def test_wrap_program_adds_depth():
    """wrap_program nests the root, increasing depth by exactly 1."""
    program = _translate_program(dy=1, dx=0)
    before_depth = program.depth()
    wrapped = wrap_program(program, Identity())
    assert wrapped.depth() == before_depth + 1
    # Hash changes because the AST shape changed.
    assert wrapped.sha256_hash() != program.sha256_hash()


def test_wrap_program_type_mismatch_raises():
    """A wrapper that expects something other than the root's output type
    must be rejected at wrap time."""
    # CountObj expects Grid in, outputs Number — wrapping a Grid-producing
    # program with CountObj would change desired_output to Number, but the
    # wrap function uses the typed sig check; CountObj accepts Grid so it
    # actually succeeds. The error path is when the wrapper expects Object
    # input but root produces Grid:
    from misfit_agent.dsl.primitives import ShapeOf
    program = _translate_program(dy=1, dx=0)  # root produces Grid
    with pytest.raises(TypeMismatchError):
        wrap_program(program, ShapeOf())  # ShapeOf expects Object


def test_mutate_param_changes_hash_and_param():
    program = _translate_program(dy=1, dx=0)
    edited = mutate_param(program, target_idx=0,
                          param_name="dy", new_value=5)
    assert edited.sha256_hash() != program.sha256_hash()
    assert edited.root.primitive.dy == 5
    # Untouched param preserved.
    assert edited.root.primitive.dx == 0


def test_mutate_param_unknown_param_raises():
    program = _translate_program(dy=1, dx=0)
    with pytest.raises(AttributeError):
        mutate_param(program, target_idx=0,
                     param_name="nonsense", new_value=3)


def test_mutate_param_out_of_range_index_raises():
    program = _translate_program(dy=1, dx=0)
    with pytest.raises(IndexError):
        mutate_param(program, target_idx=42,
                     param_name="dy", new_value=1)


def test_swap_primitive_does_not_mutate_input():
    """The caller's program must NOT be modified by an edit op."""
    program = _translate_program(dy=1, dx=0)
    original_hash = program.sha256_hash()
    _ = swap_primitive(program, target_idx=0, new_primitive=Identity())
    assert program.sha256_hash() == original_hash
    assert isinstance(program.root.primitive, Translate)


def test_mutate_param_does_not_mutate_input():
    program = _translate_program(dy=1, dx=0)
    original_hash = program.sha256_hash()
    _ = mutate_param(program, 0, "dy", 9)
    assert program.sha256_hash() == original_hash
    # The original primitive's dy is still 1.
    assert program.root.primitive.dy == 1


# ---------------------------------------------------------------------------
# Additional coverage: refinement loop on other primitive families
# ---------------------------------------------------------------------------


def test_refine_finds_correct_rotate_k():
    """Start with Rotate(k=1), target is Rotate(k=2). Refinement should
    sweep k ∈ {1,2,3} and land on k=2."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    correct = Rotate(k=2).apply(g)
    train_pairs = [(g, correct)]

    wrong = _rotate_program(k=1)
    refined = refine(wrong, train_pairs, max_iters=4)
    refined_score = _score(refined, train_pairs)
    assert refined_score == pytest.approx(1.0)


def test_refine_finds_correct_reflect_axis():
    """Start with Reflect(axis='H'), target is Reflect(axis='V')."""
    g = np.array([[1, 2, 0], [3, 0, 0], [0, 0, 0]], dtype=np.int32)
    correct = Reflect(axis="V").apply(g)
    train_pairs = [(g, correct)]

    wrong = _reflect_program(axis="H")
    refined = refine(wrong, train_pairs, max_iters=4)
    refined_score = _score(refined, train_pairs)
    assert refined_score == pytest.approx(1.0)


def test_refine_input_program_not_mutated():
    """End-to-end: refine() must never mutate the caller's program."""
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    correct = Translate(dy=1, dx=0).apply(g)
    train_pairs = [(g, correct)]

    wrong = _translate_program(dy=0, dx=0)
    wrong_hash_before = wrong.sha256_hash()
    wrong_string_before = wrong.to_string()

    _ = refine(wrong, train_pairs, max_iters=4)

    assert wrong.sha256_hash() == wrong_hash_before
    assert wrong.to_string() == wrong_string_before
    # The root primitive's parameters are still the original.
    assert wrong.root.primitive.dy == 0
    assert wrong.root.primitive.dx == 0


def test_refine_empty_train_pairs_returns_input():
    """With no train pairs there is nothing to score against — the input
    must be returned unchanged."""
    program = _identity_program()
    refined = refine(program, [], max_iters=4)
    assert refined.sha256_hash() == program.sha256_hash()


def test_refine_negative_max_iters_raises():
    """Defensive contract: a negative iteration budget is a bug, not a
    silent no-op."""
    g = np.array([[1]], dtype=np.int32)
    program = _identity_program()
    with pytest.raises(ValueError):
        refine(program, [(g, g.copy())], max_iters=-1)
