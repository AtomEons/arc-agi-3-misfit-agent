"""Team MDL — arithmetic-coded prior scorer over typed Program ASTs.

Covers:
  - encoding_bits(Identity-only)              < encoding_bits(Translate-only)
  - encoding_bits is small for the cheapest 1-node program
  - Deeper program with the same root costs strictly more bits
  - score on a perfect-fit program > score on a wrong program
  - Shape-mismatched outputs contribute 0.0 to cell accuracy
  - mdl_lambda = 0 collapses score to train_cell_accuracy
  - Higher mdl_lambda penalizes longer programs more aggressively
  - HoleNode and ConstNode each contribute non-zero bit costs
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.dsl import (
    Color,
    Grid,
    Identity,
    Number,
    Recolor,
    Reflect,
    Rotate,
    Translate,
)
from misfit_agent.dsl.ast import (
    ConstNode,
    HoleNode,
    PrimitiveNode,
    Program,
    make_hole,
    make_program,
)
from misfit_agent.dsl.mdl import encoding_bits, score, train_cell_accuracy
from misfit_agent.dsl.primitives import ALL_PRIMITIVES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_program() -> Program:
    return make_program(Identity(), make_hole(Grid))


def _translate_program(dy: int = 1, dx: int = 0) -> Program:
    return make_program(Translate(dy=dy, dx=dx), make_hole(Grid))


def _shallow_translate_program() -> Program:
    """Translate(<hole>) — depth 2 AST (root + hole)."""
    return make_program(Translate(dy=1, dx=0), make_hole(Grid))


def _deep_translate_program() -> Program:
    """Translate(Translate(<hole>)) — depth 3 AST, same root primitive."""
    inner = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    outer_root = PrimitiveNode(
        primitive=Translate(dy=1, dx=0),
        children=[inner.root],
    )
    return Program(root=outer_root, desired_output=Grid)


def _grid_2x2() -> np.ndarray:
    return np.array([[1, 0], [0, 0]], dtype=np.int32)


# ---------------------------------------------------------------------------
# encoding_bits — structural prior
# ---------------------------------------------------------------------------


def test_encoding_bits_identity_is_small():
    """Identity has no parameters, so its bit cost is exactly the catalog
    discrimination cost plus the hole cost — small and bounded."""
    p = _identity_program()
    bits = encoding_bits(p)
    catalog_bits = math.log2(len(ALL_PRIMITIVES))
    # Identity primitive: log2(12) ≈ 3.58 bits.
    # Hole: log2(12) ≈ 3.58 bits.
    expected = 2 * catalog_bits
    assert math.isclose(bits, expected, abs_tol=1e-6), (
        f"Identity program bits = {bits}, expected ≈ {expected}"
    )
    # Small in absolute terms — under 10 bits.
    assert bits < 10.0


def test_encoding_bits_translate_exceeds_identity():
    """Translate carries two integer parameters → strictly more bits than
    Identity, same hole count."""
    bits_id = encoding_bits(_identity_program())
    bits_tr = encoding_bits(_translate_program())
    assert bits_tr > bits_id, (
        f"Translate({bits_tr}) must cost more bits than Identity({bits_id})"
    )
    # The gap should be roughly Translate's parameter cost (~9.92 bits).
    assert bits_tr - bits_id > 5.0


def test_encoding_bits_deeper_program_costs_more():
    """Adding another Translate node above a Translate node must increase
    encoding_bits — depth penalty under the prior."""
    shallow = _shallow_translate_program()
    deep = _deep_translate_program()
    bits_shallow = encoding_bits(shallow)
    bits_deep = encoding_bits(deep)
    assert bits_deep > bits_shallow, (
        f"Deep({bits_deep}) must cost more than shallow({bits_shallow})"
    )
    # Each extra Translate node adds at least its own discrimination cost.
    assert (bits_deep - bits_shallow) > math.log2(len(ALL_PRIMITIVES))


def test_encoding_bits_hole_contributes_catalog_cost():
    """A bare HoleNode-rooted Program costs exactly log2(|catalog|) bits."""
    hole_program = Program(root=make_hole(Grid), desired_output=Grid)
    bits = encoding_bits(hole_program)
    expected = math.log2(len(ALL_PRIMITIVES))
    assert math.isclose(bits, expected, abs_tol=1e-6)


def test_encoding_bits_const_node_contributes_domain_bits():
    """A ConstNode literal contributes log2(domain) bits — Color is 10."""
    const_root = ConstNode(value_type=Color, value=3)
    const_program = Program(root=const_root, desired_output=Color)
    bits = encoding_bits(const_program)
    expected = math.log2(10)
    assert math.isclose(bits, expected, abs_tol=1e-6), (
        f"ConstNode(Color=3) bits = {bits}, expected ≈ {expected}"
    )


# ---------------------------------------------------------------------------
# score — fit minus MDL penalty
# ---------------------------------------------------------------------------


def test_score_perfect_fit_beats_wrong_program():
    """Identity should perfectly fit an input==output pair; Rotate(k=1) will
    not. The score must rank Identity above Rotate."""
    g = _grid_2x2()
    pairs = [(g, g.copy())]
    p_right = _identity_program()
    p_wrong = make_program(Rotate(k=1), make_hole(Grid))
    s_right = score(p_right, pairs, mdl_lambda=0.01)
    s_wrong = score(p_wrong, pairs, mdl_lambda=0.01)
    assert s_right > s_wrong, (
        f"perfect-fit({s_right}) must beat wrong-program({s_wrong})"
    )


def test_score_shape_mismatch_yields_zero_accuracy():
    """When the program produces a differently-shaped output than the target,
    cell accuracy on that pair is exactly 0.0 — and the overall score equals
    -mdl_lambda * encoding_bits."""
    # Input is 2x3; we craft a target that differs in shape from the program's
    # natural output. Identity preserves input shape (2x3), but the target is
    # 4x4 — guaranteed mismatch.
    inp = np.zeros((2, 3), dtype=np.int32)
    target = np.zeros((4, 4), dtype=np.int32)
    pairs = [(inp, target)]
    p = _identity_program()
    acc = train_cell_accuracy(p, pairs)
    assert acc == 0.0, f"shape mismatch must score 0.0, got {acc}"
    # And the full score should equal -mdl_lambda * bits.
    lam = 0.01
    s = score(p, pairs, mdl_lambda=lam)
    expected = 0.0 - lam * encoding_bits(p)
    assert math.isclose(s, expected, abs_tol=1e-9)


def test_score_mdl_lambda_zero_equals_train_accuracy():
    """At mdl_lambda = 0 the MDL penalty drops out and score == accuracy."""
    g = _grid_2x2()
    pairs = [(g, g.copy())]
    p = _identity_program()
    acc = train_cell_accuracy(p, pairs)
    s_zero = score(p, pairs, mdl_lambda=0.0)
    assert math.isclose(s_zero, acc, abs_tol=1e-9), (
        f"score(λ=0)={s_zero} must equal cell accuracy={acc}"
    )
    # Sanity: Identity perfectly recovers the input, so accuracy is 1.0.
    assert math.isclose(acc, 1.0, abs_tol=1e-9)


def test_score_higher_lambda_penalizes_longer_programs_more():
    """For two programs with the SAME perfect train accuracy, a larger
    mdl_lambda widens the gap in favor of the shorter program."""
    g = _grid_2x2()
    # Identity perfectly fits (g, g). A Reflect(H) program does NOT fit
    # (it would flip the grid). To test "same accuracy / different length"
    # we pick a target that BOTH fit perfectly: the all-zero grid.
    #
    # Identity(zeros) -> zeros. Reflect(H)(zeros) -> zeros. Same accuracy = 1.0.
    z = np.zeros((3, 3), dtype=np.int32)
    pairs = [(z, z.copy())]
    p_short = _identity_program()
    p_long = make_program(Reflect(axis="H"), make_hole(Grid))

    # Both fit perfectly under the data.
    assert train_cell_accuracy(p_short, pairs) == 1.0
    assert train_cell_accuracy(p_long, pairs) == 1.0

    # encoding_bits(long) > encoding_bits(short) — Reflect adds axis param.
    bits_short = encoding_bits(p_short)
    bits_long = encoding_bits(p_long)
    assert bits_long > bits_short

    gap_small_lambda = (
        score(p_short, pairs, mdl_lambda=0.001)
        - score(p_long, pairs, mdl_lambda=0.001)
    )
    gap_big_lambda = (
        score(p_short, pairs, mdl_lambda=1.0)
        - score(p_long, pairs, mdl_lambda=1.0)
    )
    # The bigger lambda widens the gap in favor of the shorter program.
    assert gap_big_lambda > gap_small_lambda, (
        f"big-λ gap({gap_big_lambda}) must exceed small-λ gap({gap_small_lambda})"
    )
    # And both gaps are non-negative — shorter is never worse here.
    assert gap_small_lambda >= 0.0
    assert gap_big_lambda > 0.0


def test_score_handles_multiple_train_pairs():
    """Score over multiple pairs averages cell accuracies — verify it's the
    mean of per-pair accuracies, not the sum."""
    g1 = np.array([[1, 0], [0, 0]], dtype=np.int32)
    g2 = np.array([[0, 0], [0, 2]], dtype=np.int32)
    # Identity fits both pairs perfectly.
    pairs = [(g1, g1.copy()), (g2, g2.copy())]
    p = _identity_program()
    acc = train_cell_accuracy(p, pairs)
    assert math.isclose(acc, 1.0, abs_tol=1e-9)
    # Now flip ONE pair to mismatch — accuracy should drop to ~0.5
    # (one pair perfect + one pair 0.0 due to differing values).
    mismatched = np.array([[9, 9], [9, 9]], dtype=np.int32)
    pairs_mixed = [(g1, g1.copy()), (g1, mismatched)]
    acc_mixed = train_cell_accuracy(p, pairs_mixed)
    # First pair = 1.0, second pair = 0.0 (all cells differ) → mean = 0.5
    assert math.isclose(acc_mixed, 0.5, abs_tol=1e-9), (
        f"mean accuracy = {acc_mixed}, expected 0.5"
    )


def test_score_empty_train_pairs_is_pure_penalty():
    """With no training data, accuracy is 0.0 and score reduces to the
    negated MDL penalty — a clean floor that synthesis can rely on."""
    p = _identity_program()
    s = score(p, [], mdl_lambda=0.01)
    expected = 0.0 - 0.01 * encoding_bits(p)
    assert math.isclose(s, expected, abs_tol=1e-9)


def test_score_program_evaluation_error_counts_as_miss():
    """If evaluating the program raises (e.g. wrong grid shape for an op
    the program assumes), the offending pair contributes 0.0 accuracy."""
    # Build a program that performs a Recolor — perfectly valid on grids of
    # any shape. To force an error, hand it a non-grid input.
    p = make_program(Recolor(mapping={0: 1}), make_hole(Grid))
    target = np.zeros((2, 2), dtype=np.int32)
    # First pair: valid → 0.0 accuracy because all zeros become ones, but
    # target is zeros. Second pair: bogus input → caught as miss.
    pairs = [("not a grid", target)]
    acc = train_cell_accuracy(p, pairs)
    assert acc == 0.0
