"""Reduce combinator: count-driven repeated-application semantics.

Tests cover:
  - signature_typed() reports Grid → Grid with both meta-params declared
  - Reduce(CountObj, Identity) returns the identity (n iterations of
    identity = identity, for any n)
  - Reduce(CountObj, Rotate(k=1)) on a grid with 4 perceived objects =
    rotated 4 times = the original (full 360° via 4 × 90° rotations)
  - Reduce with a count of 0 (no perceived objects) returns the input
    unchanged
  - mdl_bits() strictly greater than either bare child's MDL alone
  - Hash key carries BOTH child programs' structure: distinct count
    programs OR distinct transform programs produce distinct PrimitiveNode
    hash keys (memoization correctness)
  - Construction with a non-Number count_program raises TypeMismatchError
  - Construction with a non-Grid transform_program raises TypeMismatchError
  - Missing children raise TypeMismatchError
  - apply() does not mutate the caller's input grid
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Number, DslType, TypeMismatchError,
    Identity, Rotate, CountObj, Translate,
)
from misfit_agent.dsl.ast import (
    Program, PrimitiveNode, make_hole, make_program,
)
from misfit_agent.dsl.combinators.reduce import Reduce
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_obj_program() -> Program:
    """CountObj(<input>) — a Grid → Number program."""
    return make_program(CountObj(), make_hole(Grid))


def _identity_transform() -> Program:
    """Identity(<input>) — a Grid → Grid program."""
    return make_program(Identity(), make_hole(Grid))


def _rotate_transform(k: int = 1) -> Program:
    """Rotate(k=k)(<input>) — a Grid → Grid program with a parameter."""
    return make_program(Rotate(k=k), make_hole(Grid))


def _translate_transform(dy: int = 0, dx: int = 0) -> Program:
    """Translate(dy=dy, dx=dx)(<input>) — a Grid → Grid program."""
    return make_program(Translate(dy=dy, dx=dx), make_hole(Grid))


def _four_object_square_grid() -> np.ndarray:
    """A 4x4 grid containing exactly 4 isolated foreground objects.

    Layout (0 = bg):
        1 0 2 0
        0 0 0 0
        3 0 0 4
        0 0 0 0

    Each non-background cell is its own object under the perceptor's
    4-connectivity flood-fill rule (no two non-bg cells share an edge),
    so CountObj on this grid returns exactly 4. Rotating a square grid
    by k=1 (90° counter-clockwise) four times returns the original grid
    structurally — every cell visits four positions in a cycle of length
    4 and ends back where it started.
    """
    g = np.zeros((4, 4), dtype=np.int32)
    g[0, 0] = 1
    g[0, 2] = 2
    g[2, 0] = 3
    g[2, 3] = 4
    return g


def _empty_grid() -> np.ndarray:
    """A grid with zero perceived foreground objects — CountObj returns 0.

    A uniform-background grid has no non-bg cells, so the perceptor's
    flood-fill finds no foreground objects. CountObj returns 0 and a
    Reduce wrapping it should apply the transform zero times and return
    the input unchanged.
    """
    return np.zeros((3, 3), dtype=np.int32)


# ---------------------------------------------------------------------------
# 1. Typed signature
# ---------------------------------------------------------------------------


def test_signature_typed_is_grid_to_grid():
    """Reduce(count, transform) declares Grid → Grid with one AST input."""
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    sig = r.signature_typed()
    assert sig.inputs == (("g", Grid),)
    assert sig.output == Grid
    # And the type-checked alias matches:
    assert sig.output == DslType.GRID


def test_signature_meta_params_declared():
    """Both child programs are declared as meta-params in the signature."""
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    sig = r.signature_typed()
    param_names = [n for n, _ in sig.params]
    assert "count_program" in param_names
    assert "transform_program" in param_names


# ---------------------------------------------------------------------------
# 2. Identity transform: n iterations = identity for any n
# ---------------------------------------------------------------------------


def test_identity_transform_returns_input_value_equal():
    """Reduce(CountObj, Identity) is the identity transform: applying
    Identity any number of times leaves the grid unchanged."""
    g = _four_object_square_grid()
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    out = r.apply(g)
    assert np.array_equal(out, g), (
        f"Reduce(CountObj, Identity) altered the input:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_identity_transform_via_program_node_round_trip():
    """Reduce is reachable through the typed AST and produces the same
    result as a direct apply() call."""
    g = _four_object_square_grid()
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    root = PrimitiveNode(primitive=r, children=[make_hole(Grid)])
    prog = Program(root=root, desired_output=Grid)
    out = evaluate(prog, g)
    assert np.array_equal(out, g)


# ---------------------------------------------------------------------------
# 3. Rotate transform: 4 rotations of k=1 on 4-object square = original
# ---------------------------------------------------------------------------


def test_rotate_k1_four_times_equals_original():
    """A 4x4 grid with exactly 4 perceived objects, rotated four times
    by k=1 (90°), returns to its original orientation. This is the
    canonical NUMEROSITY → ACTION lift: the object count drives the
    iteration count, and the iteration count is exactly the cycle
    length of the transform on a square grid."""
    g = _four_object_square_grid()
    assert int(CountObj().apply(g)) == 4, (
        f"fixture assumption violated: expected 4 objects, got "
        f"{int(CountObj().apply(g))}"
    )

    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=1),
    )
    out = r.apply(g)
    assert np.array_equal(out, g), (
        f"4× Rotate(k=1) did not return the original 4x4 grid:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_rotate_k1_intermediate_step_differs():
    """Sanity check: a single rotation of k=1 produces a grid distinct
    from the input — so the four-rotation round-trip is non-trivial.
    Without this, the Identity test above could be silently masking a
    bug where the transform is never applied."""
    g = _four_object_square_grid()
    once = Rotate(k=1).apply(g)
    assert not np.array_equal(once, g), (
        f"Rotate(k=1) returned the input unchanged — fixture is bad:\n"
        f"IN:\n{g}\nOUT:\n{once}"
    )


# ---------------------------------------------------------------------------
# 4. Zero-count: no iterations, return input
# ---------------------------------------------------------------------------


def test_zero_count_returns_input_unchanged():
    """When the count program returns 0, Reduce applies the transform
    zero times and returns the input grid. The transform program is
    *configured* (so synthesis treats Reduce as a real wrapper) but is
    not *invoked* — even a transform that would otherwise alter the
    grid (e.g. Rotate) does not affect the output when n=0."""
    g = _empty_grid()
    assert int(CountObj().apply(g)) == 0, (
        f"fixture assumption violated: expected 0 objects, got "
        f"{int(CountObj().apply(g))}"
    )

    r = Reduce(
        count_program=_count_obj_program(),
        # A transform that DOES change the grid — but n=0 means it's
        # never called, so the output equals the input.
        transform_program=_rotate_transform(k=1),
    )
    out = r.apply(g)
    assert np.array_equal(out, g), (
        f"Reduce with count=0 altered the input:\n"
        f"IN:\n{g}\nOUT:\n{out}"
    )


def test_zero_count_with_translate_returns_input():
    """Same as above with a Translate transform — confirms the "n=0 is
    identity" contract is independent of the transform program shape."""
    g = _empty_grid()
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_translate_transform(dy=1, dx=1),
    )
    out = r.apply(g)
    assert np.array_equal(out, g)


# ---------------------------------------------------------------------------
# 5. Caller-input safety
# ---------------------------------------------------------------------------


def test_apply_does_not_mutate_caller_input():
    """The input grid object must not be mutated by apply() — even when
    the transform program is applied multiple times. This is a defence
    against in-place numpy operations leaking back through the input
    reference."""
    g = _four_object_square_grid()
    g_before = g.copy()
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=1),
    )
    _ = r.apply(g)
    assert np.array_equal(g, g_before), (
        f"Reduce mutated the caller's input:\n"
        f"before:\n{g_before}\nafter:\n{g}"
    )


# ---------------------------------------------------------------------------
# 6. MDL bits accounting
# ---------------------------------------------------------------------------


def test_mdl_bits_strictly_greater_than_bare_count():
    """Reduce must cost strictly more bits than its count program alone.
    Wrapping commits to a composition shape AND a second sub-program,
    both of which are real encoding choices the synthesis prior pays
    bits for."""
    bare_count_bits = CountObj().mdl_bits()
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    assert r.mdl_bits() > bare_count_bits, (
        f"Reduce did not cost more bits than its bare count: "
        f"r={r.mdl_bits()}  bare_count={bare_count_bits}"
    )


def test_mdl_bits_strictly_greater_than_bare_transform():
    """Reduce must also cost strictly more bits than the transform
    program alone — confirms BOTH sub-programs contribute to the wrapper
    cost, not just one."""
    bare_transform_bits = Identity().mdl_bits()
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    assert r.mdl_bits() > bare_transform_bits


def test_mdl_bits_is_strictly_positive_and_finite():
    """The MDL cost is well-formed: positive, finite, not NaN."""
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    bits = r.mdl_bits()
    assert bits > 0.0
    assert bits == bits  # not NaN
    assert bits != float("inf")


def test_mdl_bits_scales_with_transform_complexity():
    """A Reduce wrapping a parameterized transform (Rotate) costs strictly
    more bits than one wrapping Identity. Confirms the transform program's
    parameter MDL is included in the wrapper cost."""
    r_simple = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    r_complex = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=1),
    )
    assert r_complex.mdl_bits() > r_simple.mdl_bits()


# ---------------------------------------------------------------------------
# 7. Hash key includes BOTH child programs
# ---------------------------------------------------------------------------


def test_hash_key_distinguishes_distinct_transforms():
    """Two Reduces with the same count program but different transform
    programs MUST produce different hash keys — memoization correctness."""
    r_id = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    r_rot = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=1),
    )
    node_id = PrimitiveNode(primitive=r_id, children=[make_hole(Grid)])
    node_rot = PrimitiveNode(primitive=r_rot, children=[make_hole(Grid)])
    assert node_id.hash_key() != node_rot.hash_key(), (
        f"hash_key collision between distinct Reduce transforms:\n"
        f"  identity: {node_id.hash_key()}\n"
        f"  rotate:   {node_rot.hash_key()}"
    )
    # And the transform's name shows up inside the hash key:
    assert "Identity" in node_id.hash_key()
    assert "Rotate" in node_rot.hash_key()


def test_hash_key_distinguishes_transform_parameter_values():
    """Same transform family with different parameter values must hash
    differently — guards Rotate(k=1) vs Rotate(k=2) inside Reduce."""
    r_k1 = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=1),
    )
    r_k2 = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=2),
    )
    n1 = PrimitiveNode(primitive=r_k1, children=[make_hole(Grid)])
    n2 = PrimitiveNode(primitive=r_k2, children=[make_hole(Grid)])
    assert n1.hash_key() != n2.hash_key()


def test_hash_key_carries_combinator_marker():
    """The hash key must mention 'Reduce' so memoization tables stay
    disambiguating from other combinators (WhileChanging, ForEachObject,
    Parallel, etc.)."""
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_identity_transform(),
    )
    node = PrimitiveNode(primitive=r, children=[make_hole(Grid)])
    prog = Program(root=node, desired_output=Grid)
    assert "Reduce" in prog.hash_key()
    assert "Reduce" in prog.to_string()


def test_hash_key_carries_both_child_names():
    """The hash key must mention BOTH the count program and the transform
    program by name — guards against memoization mis-keying on only one
    of the two sub-programs."""
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=1),
    )
    node = PrimitiveNode(primitive=r, children=[make_hole(Grid)])
    key = node.hash_key()
    assert "CountObj" in key, f"hash_key missing CountObj: {key}"
    assert "Rotate" in key, f"hash_key missing Rotate: {key}"


# ---------------------------------------------------------------------------
# 8. Type-mismatch validation at construction time
# ---------------------------------------------------------------------------


def test_type_mismatch_raised_if_count_program_returns_grid():
    """A count program whose output type is Grid (Identity) must be
    rejected at construction time — synthesis must never enumerate a
    Reduce that tries to use a grid as an iteration count."""
    bad_count = _identity_transform()
    assert bad_count.output_type() == Grid  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        Reduce(
            count_program=bad_count,
            transform_program=_identity_transform(),
        )
    msg = str(ei.value)
    assert "Number" in msg, (
        f"TypeMismatchError message did not name expected type Number: {msg}"
    )


def test_type_mismatch_raised_if_transform_returns_number():
    """A transform program whose output type is Number (CountObj) must
    be rejected at construction time — synthesis must never enumerate
    a Reduce whose 'transform' degrades the running grid to a number."""
    bad_transform = _count_obj_program()
    assert bad_transform.output_type() == Number  # sanity check on fixture
    with pytest.raises(TypeMismatchError) as ei:
        Reduce(
            count_program=_count_obj_program(),
            transform_program=bad_transform,
        )
    msg = str(ei.value)
    assert "Grid" in msg, (
        f"TypeMismatchError message did not name expected type Grid: {msg}"
    )


def test_type_mismatch_raised_if_count_program_missing():
    """No count program at all is a type error — there is no default."""
    with pytest.raises(TypeMismatchError):
        Reduce(
            count_program=None,
            transform_program=_identity_transform(),
        )


def test_type_mismatch_raised_if_transform_program_missing():
    """No transform program at all is a type error — there is no default."""
    with pytest.raises(TypeMismatchError):
        Reduce(
            count_program=_count_obj_program(),
            transform_program=None,
        )


# ---------------------------------------------------------------------------
# 9. to_string display
# ---------------------------------------------------------------------------


def test_to_string_includes_both_children():
    """The string display must surface BOTH child programs so
    operator-readable program dumps are unambiguous."""
    r = Reduce(
        count_program=_count_obj_program(),
        transform_program=_rotate_transform(k=2),
    )
    s = r.to_string()
    assert "Reduce" in s
    assert "CountObj" in s
    assert "Rotate" in s
    assert "k=2" in s
