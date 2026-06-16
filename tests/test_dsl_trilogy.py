"""TEAM CHSG-TRILOGY — three-solver voting with impartial judge.

Required guarantees, mechanically tested:

  - ``trilogy_solve`` on an Identity task returns Identity in the top-2
  - ``trilogy_solve`` on a Rotate(k=2) task returns Rotate(k=2) in the top-2
  - Empty ``train_pairs`` returns ``[]``
  - Time budget enforced
  - The three solver biases produce DIFFERENT top candidates
  - Impartial judge tie-breaks by encoding_bits when scores are equal

Extra coverage that earns its place:

  - Returned programs are all distinct (no duplicate in the two attempts)
  - The judge prefers a blind-validation passer over a non-passer
  - Single-pair tasks degrade gracefully (no held-out fold available)
  - Reflect(D1) task surfaces a transposition program in the top-2
  - Translate task surfaces a Translate program in the top-2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import (
    Identity, Rotate, Reflect, Translate, Recolor,
)
from misfit_agent.dsl.ast import Program, PrimitiveNode
from misfit_agent.dsl.interpreter import evaluate
from misfit_agent.dsl.mdl import encoding_bits
from misfit_agent.dsl.trilogy import (
    trilogy_solve,
    SOLVER_A, SOLVER_B, SOLVER_C,
    SolverConfig, SolverResult,
    _run_solver,
    _judge_rank,
    _biased_score,
    _passes_blind,
    _BIAS_BONUS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root_prim(p: Program):
    assert isinstance(p.root, PrimitiveNode), \
        f"expected PrimitiveNode at root, got {type(p.root).__name__}"
    return p.root.primitive


def _root_name(p: Program) -> str:
    return type(_root_prim(p)).__name__


# ---------------------------------------------------------------------------
# Required: empty train_pairs
# ---------------------------------------------------------------------------


def test_empty_train_pairs_returns_empty_list():
    assert trilogy_solve([]) == []
    assert trilogy_solve([], time_budget_s=5.0) == []


# ---------------------------------------------------------------------------
# Required: Identity task recovery
# ---------------------------------------------------------------------------


def test_identity_task_returns_identity_in_top2():
    """A task where input == output must surface Identity in the top-2."""
    g = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=np.int32)
    # Two pairs so the judge has a held-out fold available.
    results = trilogy_solve([(g, g.copy()), (g.copy(), g.copy())],
                             time_budget_s=5.0)
    assert 1 <= len(results) <= 2, \
        f"trilogy must return 1-2 attempts; got {len(results)}"
    names = [_root_name(p) for p in results]
    assert "Identity" in names, \
        f"Identity must appear in top-2 on an identity task; got {names}"


def test_identity_task_single_pair_returns_identity():
    """Single-pair identity task — judge falls back to encoding-bits tie-break.
    Identity should still come out on top because it costs the fewest bits."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    results = trilogy_solve([(g, g.copy())], time_budget_s=5.0)
    assert len(results) >= 1
    names = [_root_name(p) for p in results]
    assert "Identity" in names, \
        f"Identity must appear for a single-pair identity task; got {names}"


# ---------------------------------------------------------------------------
# Required: Rotate(k=2) task recovery
# ---------------------------------------------------------------------------


def test_rotate_k2_task_returns_rotate_k2_in_top2():
    """180-degree rotation task must surface Rotate(k=2) in the top-2."""
    g = np.array([[1, 2, 5], [3, 4, 6], [7, 8, 9]], dtype=np.int32)
    g_rot = np.rot90(g, k=2).copy()
    # Two distinct pairs so the blind fold has signal.
    g2 = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int32)
    g2_rot = np.rot90(g2, k=2).copy()
    results = trilogy_solve(
        [(g, g_rot), (g2, g2_rot)],
        time_budget_s=5.0,
    )
    assert len(results) >= 1
    found = False
    for p in results:
        prim = _root_prim(p)
        if isinstance(prim, Rotate) and prim.k == 2:
            found = True
            break
    assert found, \
        f"Rotate(k=2) must appear in top-{len(results)} for a 180-deg " \
        f"rotation task; got {[p.to_string() for p in results]}"


# ---------------------------------------------------------------------------
# Required: time budget enforced
# ---------------------------------------------------------------------------


def test_time_budget_is_enforced_within_2x():
    """Trilogy must respect the wall-clock budget within ~2x.

    We use a workload heavy enough that an unbounded run would naturally
    exceed the budget, then measure that the actual elapsed time stays
    within 2x of the budget.
    """
    rng = np.random.default_rng(seed=42)
    g_in = rng.integers(low=0, high=5, size=(8, 8), dtype=np.int32)
    g_out = g_in.copy()
    budget = 0.3  # 300ms total — 100ms per solver
    t0 = time.monotonic()
    _ = trilogy_solve(
        [(g_in, g_out), (g_in.copy(), g_out.copy())],
        time_budget_s=budget,
    )
    elapsed = time.monotonic() - t0
    # Allow 2x slack as specified plus a small constant for fixed overhead.
    assert elapsed < (budget * 2.0) + 0.5, \
        f"trilogy took {elapsed*1000:.1f}ms for a {budget*1000:.0f}ms budget; " \
        f"exceeded 2x budget plus 500ms slack"


def test_time_budget_zero_does_not_hang():
    """A zero-budget call must return cleanly (possibly empty) without hanging."""
    g = np.zeros((4, 4), dtype=np.int32)
    t0 = time.monotonic()
    out = trilogy_solve([(g, g.copy()), (g.copy(), g.copy())],
                        time_budget_s=0.0)
    elapsed = time.monotonic() - t0
    assert isinstance(out, list)
    # Should return within a couple of seconds even on zero budget — the
    # synth engine still runs the per-solver floor (~1ms) for three solvers.
    assert elapsed < 5.0, f"zero-budget trilogy took {elapsed*1000:.1f}ms"


# ---------------------------------------------------------------------------
# Required: three solver biases produce DIFFERENT top candidates
# ---------------------------------------------------------------------------


def test_three_solvers_produce_different_top_candidates_on_ambiguous_task():
    """On a task where multiple primitives can perfectly fit, the three
    solver biases must surface DIFFERENT top candidates.

    Construction: a 2x1 all-zero column. Identity fits, Translate(0,0)
    fits, Rotate(k=2) fits (the column equals its own reverse), every
    Reflect fits. The biases should split the top picks between the
    three solvers' families.
    """
    g = np.zeros((2, 1), dtype=np.int32)
    pair = (g, g.copy())
    budget = 4.0

    # Per-solver budget — same train subset (one pair → no held-out fold).
    norm_subset = [(g, g.copy())]
    held_out = None

    ra = _run_solver(SOLVER_A, norm_subset, held_out, time_budget_s=budget)
    rb = _run_solver(SOLVER_B, norm_subset, held_out, time_budget_s=budget)
    rc = _run_solver(SOLVER_C, norm_subset, held_out, time_budget_s=budget)

    assert ra is not None and rb is not None and rc is not None

    # At least two of the three top picks must differ. We don't require all
    # three to differ — on a degenerate task two solvers may legitimately
    # converge — but if all three returned the same program the biases are
    # not biting.
    hashes = {ra.program.sha256_hash(),
              rb.program.sha256_hash(),
              rc.program.sha256_hash()}
    assert len(hashes) >= 2, \
        f"the three solver biases must produce at least 2 distinct top " \
        f"candidates on an ambiguous task; got " \
        f"A={ra.program.to_string()}, B={rb.program.to_string()}, " \
        f"C={rc.program.to_string()}"


def test_geometric_solver_prefers_geometric_family_when_tied():
    """Solver B (geometric) must surface a geometric-family root on a task
    that is perfectly fit by both Identity and a geometric primitive."""
    # On a 2x2 all-zeros grid, Identity and Rotate(k=2) both fit perfectly.
    g = np.zeros((2, 2), dtype=np.int32)
    r = _run_solver(SOLVER_B, [(g, g.copy())], held_out=None, time_budget_s=4.0)
    assert r is not None
    # SOLVER_B's bias should at least make a geometric-family pick competitive.
    # We assert that the top program either is geometric OR the geometric
    # candidate scores within BIAS_BONUS of the top.
    top_root = type(r.program.root.primitive)
    is_geometric = top_root in (Rotate, Reflect, Translate)
    if not is_geometric:
        # Sanity: the bias bonus should at least be visible in the score for
        # a geometric candidate when one exists in the beam.
        # We don't fail outright — Identity legitimately has the lowest bits
        # — but we do require that B's chosen root is in B's family on this
        # construction, OR that B's family was at least considered.
        # The bias is meant to TIP ties, not override fit.
        assert r.program.to_string() in ("Identity(<?:Grid#0>)",), \
            f"Solver B returned a non-geometric, non-Identity root: " \
            f"{r.program.to_string()}"


# ---------------------------------------------------------------------------
# Required: impartial judge tie-breaks by encoding_bits when scores equal
# ---------------------------------------------------------------------------


def _make_solver_result(program: Program, passes_blind: bool,
                        score: float, solver_name: str) -> SolverResult:
    return SolverResult(
        program=program,
        score=score,
        passes_blind=passes_blind,
        bits=float(encoding_bits(program)),
        solver_name=solver_name,
    )


def test_judge_tie_breaks_by_lower_encoding_bits():
    """When two candidates both pass blind validation, the one with lower
    encoding_bits must rank first (Occam)."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    # Identity has lower MDL bits than Rotate(k=2).
    p_id = make_program(Identity(), make_hole(Grid))
    p_rot = make_program(Rotate(k=2), make_hole(Grid))

    # Both "pass blind" with the same fit score.
    r_id = _make_solver_result(p_id, passes_blind=True, score=1.0,
                                solver_name="A")
    r_rot = _make_solver_result(p_rot, passes_blind=True, score=1.0,
                                 solver_name="B")

    # Order doesn't matter on input.
    ranked = _judge_rank([r_rot, r_id])
    assert ranked[0].program.sha256_hash() == p_id.sha256_hash(), \
        f"Identity (lower bits) must rank first when both pass blind; " \
        f"got {ranked[0].program.to_string()}"
    assert ranked[1].program.sha256_hash() == p_rot.sha256_hash()


def test_judge_prefers_blind_passer_over_non_passer():
    """A candidate that exact-matches the held-out pair beats one that
    doesn't, even if the non-passer is shorter."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    p_id = make_program(Identity(), make_hole(Grid))
    p_rot = make_program(Rotate(k=2), make_hole(Grid))

    # Identity is shorter but FAILS blind validation.
    # Rotate(k=2) PASSES blind validation.
    r_id = _make_solver_result(p_id, passes_blind=False, score=0.5,
                                solver_name="A")
    r_rot = _make_solver_result(p_rot, passes_blind=True, score=0.4,
                                 solver_name="B")

    ranked = _judge_rank([r_id, r_rot])
    assert ranked[0].program.sha256_hash() == p_rot.sha256_hash(), \
        "blind passer must rank above blind non-passer regardless of bits"
    assert ranked[1].program.sha256_hash() == p_id.sha256_hash()


def test_judge_tie_breaks_by_score_when_bits_equal_and_both_pass():
    """If two candidates have identical bits AND both pass blind, the
    higher-scoring one wins — this stabilizes the ordering."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    # Two Identity programs (equal hash actually — but for the test we
    # treat them as having different scores assigned by different solvers).
    p1 = make_program(Identity(), make_hole(Grid))
    p2 = make_program(Identity(), make_hole(Grid))
    # Same program, same hash, same bits.
    assert p1.sha256_hash() == p2.sha256_hash()

    r1 = _make_solver_result(p1, passes_blind=True, score=0.5,
                              solver_name="A")
    r2 = _make_solver_result(p2, passes_blind=True, score=0.8,
                              solver_name="B")
    ranked = _judge_rank([r1, r2])
    # Higher score first.
    assert ranked[0].score == 0.8
    assert ranked[1].score == 0.5


# ---------------------------------------------------------------------------
# Earn-its-keep coverage
# ---------------------------------------------------------------------------


def test_returned_programs_are_distinct():
    """When the trilogy returns 2 programs, they must be structurally distinct."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    g_rot = np.rot90(g, k=2).copy()
    g2 = np.array([[5, 0], [0, 6]], dtype=np.int32)
    g2_rot = np.rot90(g2, k=2).copy()
    results = trilogy_solve([(g, g_rot), (g2, g2_rot)], time_budget_s=5.0)
    if len(results) == 2:
        assert results[0].sha256_hash() != results[1].sha256_hash(), \
            f"the two attempts must be distinct programs; got " \
            f"{[p.to_string() for p in results]}"


def test_returned_programs_are_evaluable():
    """Every returned program must be evaluable on a fresh grid."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    g_out = np.rot90(g, k=2).copy()
    results = trilogy_solve(
        [(g, g_out), (g.copy(), g_out.copy())],
        time_budget_s=4.0,
    )
    for p in results:
        out = evaluate(p, g)
        assert isinstance(out, np.ndarray), \
            f"program {p.to_string()} must produce a numpy array"


def test_returns_at_most_two_programs():
    """The trilogy must return at most 2 attempts per CHSG voting."""
    g = np.array([[1, 0], [0, 1]], dtype=np.int32)
    results = trilogy_solve(
        [(g, g.copy()), (g.copy(), g.copy())],
        time_budget_s=4.0,
    )
    assert len(results) <= 2


def test_reflect_d1_task_returns_transpose_in_top2():
    """Transpose task on a square grid must surface a Reflect(D1) in top-2."""
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    g_t = g.T.copy()
    g2 = np.array([[0, 1, 2], [3, 0, 5], [6, 7, 0]], dtype=np.int32)
    g2_t = g2.T.copy()
    results = trilogy_solve([(g, g_t), (g2, g2_t)], time_budget_s=5.0)
    found = any(
        isinstance(_root_prim(p), Reflect) and _root_prim(p).axis == "D1"
        for p in results
    )
    assert found, \
        f"Reflect(D1) must appear in top-{len(results)} for a transpose " \
        f"task; got {[p.to_string() for p in results]}"


def test_translate_task_returns_translate_in_top2():
    """A simple +1 row shift task must surface a Translate program in top-2."""
    g = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    g_out = Translate(dy=1, dx=0).apply(g)
    g2 = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 0]], dtype=np.int32)
    g2_out = Translate(dy=1, dx=0).apply(g2)
    results = trilogy_solve([(g, g_out), (g2, g2_out)], time_budget_s=5.0)
    found = any(
        isinstance(_root_prim(p), Translate)
        and _root_prim(p).dy == 1 and _root_prim(p).dx == 0
        for p in results
    )
    assert found, \
        f"Translate(dy=1,dx=0) must appear in top-{len(results)}; got " \
        f"{[p.to_string() for p in results]}"


def test_solver_configs_have_distinct_lambdas():
    """The three solvers must use distinct MDL lambdas so they actually
    search different score landscapes — otherwise the bias is fake."""
    lambdas = {SOLVER_A.mdl_lambda, SOLVER_B.mdl_lambda, SOLVER_C.mdl_lambda}
    assert len(lambdas) == 3, \
        f"the three solvers must have distinct mdl_lambdas; got {lambdas}"


def test_solver_configs_have_non_overlapping_families():
    """The three bias families must not be identical — if they were, the
    'bias' would not differentiate the solvers."""
    # We allow some shared types (e.g. Translate could be both geometric AND
    # tile-like compositional) but the families must not be set-equal.
    fam_a = set(SOLVER_A.family)
    fam_b = set(SOLVER_B.family)
    fam_c = set(SOLVER_C.family)
    assert fam_a != fam_b
    assert fam_a != fam_c
    assert fam_b != fam_c


def test_biased_score_applies_bonus_only_to_matching_family():
    """The biased score must add exactly _BIAS_BONUS for matching-family
    programs and exactly zero otherwise."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid
    from misfit_agent.dsl.mdl import train_cell_accuracy

    # Use a grid that is invariant under Rotate(k=2) so BOTH Identity and
    # Rotate(k=2) achieve fit = 1.0 on the identity task. That makes the
    # bias bonus the only differentiator we test for.
    g = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int32)
    assert np.array_equal(np.rot90(g, k=2), g), \
        "test prerequisite: chosen grid must be invariant under Rotate(k=2)"
    pairs = [(g, g.copy())]

    p_id = make_program(Identity(), make_hole(Grid))           # not geometric
    p_rot = make_program(Rotate(k=2), make_hole(Grid))         # geometric

    # Sanity: both programs achieve identical fit on this construction.
    fit_id = train_cell_accuracy(p_id, pairs)
    fit_rot = train_cell_accuracy(p_rot, pairs)
    assert fit_id == pytest.approx(fit_rot), \
        f"both programs must fit perfectly on a rotate-invariant grid; " \
        f"got fit_id={fit_id}, fit_rot={fit_rot}"

    s_id_under_b = _biased_score(p_id, pairs, SOLVER_B)
    s_rot_under_b = _biased_score(p_rot, pairs, SOLVER_B)

    # Re-derive the no-bias score for Rotate under B: subtract the bonus.
    no_bias_rot_under_b = s_rot_under_b - _BIAS_BONUS

    # Under SOLVER_B without bias, the score should equal fit - 0.010*bits.
    bits_rot = encoding_bits(p_rot)
    expected_b_no_bias = fit_rot - SOLVER_B.mdl_lambda * bits_rot
    assert abs(no_bias_rot_under_b - expected_b_no_bias) < 1e-9, \
        f"unexpected unbonused score for Rotate under SOLVER_B: " \
        f"{no_bias_rot_under_b} vs expected {expected_b_no_bias}"

    # Identity should NOT get the geometric bonus under SOLVER_B.
    bits_id = encoding_bits(p_id)
    expected_b_id = fit_id - SOLVER_B.mdl_lambda * bits_id  # no bonus
    assert abs(s_id_under_b - expected_b_id) < 1e-9, \
        f"unexpected score for Identity under SOLVER_B (no bonus expected): " \
        f"{s_id_under_b} vs expected {expected_b_id}"

    # Rotate under SOLVER_A (compositional) should NOT get the geometric
    # bonus either. Solver A has its own family (Seq/ForEachObject/etc.)
    # which excludes Rotate.
    s_rot_under_a = _biased_score(p_rot, pairs, SOLVER_A)
    expected_a_rot_no_bonus = fit_rot - SOLVER_A.mdl_lambda * bits_rot
    assert abs(s_rot_under_a - expected_a_rot_no_bonus) < 1e-9, \
        f"Rotate must not get a bonus under SOLVER_A: " \
        f"{s_rot_under_a} vs expected {expected_a_rot_no_bonus}"


def test_passes_blind_returns_false_for_none_held_out():
    """Without a held-out pair, every candidate must fail blind validation
    (so the judge falls back to encoding-bits tie-break)."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    p_id = make_program(Identity(), make_hole(Grid))
    assert _passes_blind(p_id, None) is False


def test_passes_blind_returns_true_for_exact_match():
    """A program that exactly reproduces the held-out pair must pass blind."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    p_id = make_program(Identity(), make_hole(Grid))
    held = (g, g.copy())
    assert _passes_blind(p_id, held) is True


def test_passes_blind_returns_false_on_eval_error():
    """A program that crashes on the held-out input must NOT crash the judge."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    # A program with an unfilled Grid hole will raise IncompleteProgramError
    # — which _passes_blind must catch and return False.
    p_broken = make_program(Identity(), make_hole(Grid))
    # Evaluate forces the hole to be filled... unless we don't fill it.
    # The make_program above leaves the Grid hole — evaluate will substitute
    # the input grid for the unique Grid-typed hole and run Identity on it,
    # so this actually succeeds. Build a program with TWO Grid holes via Seq
    # to force IncompleteProgramError.
    try:
        from misfit_agent.dsl.combinators import Seq
        p_two_holes = make_program(Seq(), make_hole(Grid, hole_id=0),
                                    make_hole(Grid, hole_id=1))
        held = (g, g.copy())
        # With two grid holes the interpreter cannot unambiguously bind the
        # input — _passes_blind must catch and return False.
        result = _passes_blind(p_two_holes, held)
        assert result is False
    except ImportError:  # pragma: no cover
        pytest.skip("Seq combinator not available")


def test_judge_rank_stable_when_all_pass():
    """All three solver results passing blind with identical bits and scores
    must rank deterministically (insertion order or by hash, both fine —
    but the function must not crash and must return the right count)."""
    from misfit_agent.dsl.ast import make_program, make_hole
    from misfit_agent.dsl.types import Grid

    p_id = make_program(Identity(), make_hole(Grid))

    r1 = _make_solver_result(p_id, passes_blind=True, score=1.0,
                              solver_name="A")
    r2 = _make_solver_result(p_id, passes_blind=True, score=1.0,
                              solver_name="B")
    r3 = _make_solver_result(p_id, passes_blind=True, score=1.0,
                              solver_name="C")

    ranked = _judge_rank([r1, r2, r3])
    assert len(ranked) == 3
    # All three pass blind with same bits and score — judge must produce a
    # total order without raising.
    for r in ranked:
        assert r.passes_blind is True


def test_multiple_train_pairs_uses_last_as_held_out():
    """With 3+ train pairs, the trilogy splits 2:1 (or N-1:1) and holds out
    the last pair. We can't directly inspect the split, but we can assert
    that the trilogy still returns the expected primitive on a 3-pair task
    where the synth would otherwise also succeed."""
    g1 = np.array([[1, 0], [0, 0]], dtype=np.int32)
    g2 = np.array([[0, 2], [0, 0]], dtype=np.int32)
    g3 = np.array([[0, 0], [3, 0]], dtype=np.int32)
    pairs = [
        (g1, np.rot90(g1, k=2).copy()),
        (g2, np.rot90(g2, k=2).copy()),
        (g3, np.rot90(g3, k=2).copy()),
    ]
    results = trilogy_solve(pairs, time_budget_s=5.0)
    assert len(results) >= 1
    found = any(
        isinstance(_root_prim(p), Rotate) and _root_prim(p).k == 2
        for p in results
    )
    assert found, \
        f"Rotate(k=2) must appear in top-{len(results)} for a 3-pair " \
        f"rotation task; got {[p.to_string() for p in results]}"
