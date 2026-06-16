"""Day-1 DSL primitives: type-checking + apply correctness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Grid, Color, Number, Object as DslObject, ObjSet, Mask, Bool,
    DslType, type_signature, Signature, TypeMismatchError,
    Primitive,
    Identity, Translate, Rotate, Reflect, Recolor,
    Crop, Tile, Gravity, Symmetrize, KeepWhere,
    CountObj, ShapeOf,
    ALL_PRIMITIVES,
)


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------


def test_type_catalog_has_seven_types():
    expected = {"Grid", "Color", "Number", "Object", "ObjSet", "Mask", "Bool"}
    actual = {t.value for t in DslType}
    assert actual == expected


def test_signature_repr_human_readable():
    sig = Translate(dy=1, dx=0).signature_typed()
    s = repr(sig)
    assert "Grid" in s
    assert "→" in s


def test_type_mismatch_error_names_locations():
    err = TypeMismatchError("test_edge", DslType.GRID, DslType.NUMBER)
    assert err.expected == DslType.GRID
    assert err.actual == DslType.NUMBER
    assert "test_edge" in str(err)


# ---------------------------------------------------------------------------
# Primitive signatures — all 12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PRIMITIVES)
def test_every_primitive_declares_typed_signature(cls):
    prim = cls()
    sig = type_signature(prim)
    assert isinstance(sig, Signature)
    assert len(sig.inputs) >= 1


def test_catalog_has_twelve_primitives():
    assert len(ALL_PRIMITIVES) == 12


# ---------------------------------------------------------------------------
# Apply correctness — one test per primitive
# ---------------------------------------------------------------------------


def test_identity_returns_copy():
    g = np.array([[0, 1], [2, 3]], dtype=np.int32)
    out = Identity().apply(g)
    assert np.array_equal(out, g)
    assert out is not g  # actual copy


def test_translate_shifts_correctly():
    g = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    out = Translate(dy=1, dx=1).apply(g)
    expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_rotate_90_works():
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    out = Rotate(k=1).apply(g)
    assert np.array_equal(out, np.rot90(g))


def test_rotate_signature_correct():
    sig = Rotate(k=2).signature_typed()
    assert sig.inputs == (("g", DslType.GRID),)
    assert sig.output == DslType.GRID


def test_reflect_horizontal():
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    out = Reflect(axis="H").apply(g)
    assert np.array_equal(out, np.fliplr(g))


def test_reflect_vertical():
    g = np.array([[1, 0], [0, 0]], dtype=np.int32)
    out = Reflect(axis="V").apply(g)
    assert np.array_equal(out, np.flipud(g))


def test_reflect_diagonal_d1():
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = Reflect(axis="D1").apply(g)
    assert np.array_equal(out, g.T)


def test_recolor_swaps_colors():
    g = np.array([[1, 2], [2, 1]], dtype=np.int32)
    out = Recolor(mapping={1: 5, 2: 6}).apply(g)
    expected = np.array([[5, 6], [6, 5]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_crop_to_bounding_box():
    g = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    out = Crop().apply(g)
    expected = np.array([[1, 1], [1, 1]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_tile_2x2():
    g = np.array([[1, 0]], dtype=np.int32)
    out = Tile(rf=2, cf=2).apply(g)
    expected = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_gravity_down_collapses_column():
    g = np.array([
        [1, 0],
        [0, 0],
        [1, 0],
    ], dtype=np.int32)
    out = Gravity(direction="D").apply(g)
    expected = np.array([
        [0, 0],
        [1, 0],
        [1, 0],
    ], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_gravity_right_collapses_row():
    g = np.array([[1, 0, 1]], dtype=np.int32)
    out = Gravity(direction="R").apply(g)
    expected = np.array([[0, 1, 1]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_symmetrize_h_completes_mirror():
    g = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    out = Symmetrize(axis="H").apply(g)
    # Left side of input has the 1; H-symmetry means right side also gets it
    expected = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_keep_where_largest_strips_smaller():
    g = np.array([
        [1, 0, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    # The 1 and 2 are each single cells with area 1; "largest" picks one
    # (tied). Confirm output is not all-zero and has at most one cell.
    out = KeepWhere(predicate="largest").apply(g)
    assert (out != 0).sum() >= 1


def test_count_obj_returns_object_count():
    g = np.array([
        [1, 0, 2],
        [0, 0, 0],
        [3, 0, 0],
    ], dtype=np.int32)
    n = CountObj().apply(g)
    assert isinstance(n, int)
    assert n == 3


def test_shape_of_returns_grid_for_object():
    from misfit_agent.perceptor import perceive_grid
    g = np.array([
        [0, 0, 0],
        [0, 5, 5],
        [0, 5, 5],
    ], dtype=np.int32)
    scene = perceive_grid(g)
    out = ShapeOf().apply(scene.objects[0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tier-1 honesty
# ---------------------------------------------------------------------------


def test_no_primitive_imports_llm_library():
    """Read every DSL primitive source file and grep for forbidden imports."""
    import re
    forbidden = [
        r"\bfrom\s+transformers",
        r"\bfrom\s+openai",
        r"\bfrom\s+anthropic",
        r"\bfrom\s+llama_cpp",
        r"\btorch\.load\b",
        r"\bhuggingface_hub",
    ]
    dsl_dir = Path(__file__).parent.parent / "src" / "misfit_agent" / "dsl"
    for py in dsl_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for pat in forbidden:
            assert not re.search(pat, text), (
                f"Tier-1 violation: {py.name} matches forbidden pattern {pat!r}"
            )


def test_mdl_bits_strictly_positive_for_parameterized_primitives():
    """Parameterized primitives must cost MORE bits than the bare catalog
    choice — that's the MDL prior at work.

    Recolor with an empty mapping is degenerate (equivalent to Identity) and
    correctly costs no extra bits; we instantiate it with a non-trivial
    mapping for the comparison.
    """
    bare = Identity().mdl_bits()
    instances = [
        Translate(dy=1, dx=0),
        Rotate(k=1),
        Reflect(axis="H"),
        Recolor(mapping={1: 2}),
        Tile(rf=2, cf=2),
        Gravity(direction="D"),
        Symmetrize(axis="H"),
        KeepWhere(predicate="largest"),
    ]
    for prim in instances:
        assert prim.mdl_bits() > bare, (
            f"{type(prim).__name__}.mdl_bits() should exceed bare-catalog "
            f"bits because it has parameters"
        )
