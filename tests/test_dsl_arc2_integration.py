"""TEAM ARC2-INTEGRATION — wire the DSL synthesis engine into solve_task.

The integration contract (from the wave-2 brief):

  - Keep the existing hand-rule beam running.
  - After hand-rules finish, run dsl.synthesize + refine + seed_from_resonance.
  - Wrap each surviving DSL Program so it presents the same .predict /
    .signature interface as a hand-rule, then merge into a single beam.
  - Re-rank merged candidates by train-pair score.
  - Return top-2 distinct attempts.
  - DSL-found programs are surfaced so the resonance updater can record them.

Tests cover, mechanically:
  1. Rotate(k=2) task — solver returns the correct rotated grid AND the
     DSL engine independently produced a Program that reproduces the
     correct output on every train pair (proves DSL participated).
  2. Recolor regression — the legacy recolor case still solves correctly
     after the DSL leg is bolted on.
  3. Per-task time budget — hand-rule time + DSL time stays under the
     declared total budget (with a small tolerance for OS jitter).
  4. Two distinct attempts — when at least two non-equivalent candidates
     fit the train pairs, attempt_1 != attempt_2 element-wise.
  5. DSL programs surfaced for resonance updater — the
     solve_task_with_dsl_programs return value carries the list of DSL
     programs that survived into the merged beam.

Plus a few extra-coverage tests that earn their place:
  - DSL leg can be disabled via use_dsl=False (legacy hand-rule-only path).
  - DSL leg degrades gracefully when given an empty time budget.
  - Identity-only tasks still return identity attempts (honest-abstain).
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.arc2_solver import (
    solve_task,
    solve_task_with_dsl_programs,
    DSL_DEFAULT_TIME_BUDGET_S,
)
from misfit_agent.dsl.ast import Program
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers — small builders so each test reads as a single intent.
# ---------------------------------------------------------------------------


def _rotate_k2_task(n: int = 3, seed: int = 7):
    """Build a Rotate(k=2) task: every output = input rotated 180°."""
    rng = np.random.default_rng(seed=seed)
    train_pairs = []
    for i in range(n):
        g = rng.integers(low=0, high=5, size=(3 + i, 3 + i), dtype=np.int32)
        train_pairs.append((g.copy(), np.rot90(g, k=2).copy()))
    test_input = rng.integers(low=0, high=5, size=(4, 4), dtype=np.int32)
    expected = np.rot90(test_input, k=2)
    return train_pairs, test_input, expected


def _recolor_task():
    """Build a Recolor task: 1->2 across pairs. Regression sanity."""
    train_pairs = [
        (np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32),
         np.array([[2, 0, 2], [0, 2, 0]], dtype=np.int32)),
        (np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int32),
         np.array([[0, 2, 2], [2, 0, 0]], dtype=np.int32)),
    ]
    test_input = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    expected = np.array([[2, 2, 0], [0, 2, 2]], dtype=np.int32)
    return train_pairs, test_input, expected


def _identity_task():
    """Build an identity-only task: every output == input. Test input shares
    the bg=0 convention with the train pairs so BackgroundSwap fits as a
    no-op everywhere (avoiding spurious recolor under bg-disagreement)."""
    train_pairs = [
        (np.array([[1, 0], [0, 1]], dtype=np.int32),
         np.array([[1, 0], [0, 1]], dtype=np.int32)),
        (np.array([[2, 3, 0], [0, 4, 5]], dtype=np.int32),
         np.array([[2, 3, 0], [0, 4, 5]], dtype=np.int32)),
    ]
    # Test input also has bg=0 (the 0 cells dominate), matching the
    # train-pair convention so identity-equivalent rules apply uniformly.
    test_input = np.array([[7, 0], [0, 8]], dtype=np.int32)
    return train_pairs, test_input


# ---------------------------------------------------------------------------
# Required test 1: Rotate(k=2) — DSL engine produces the answer.
# ---------------------------------------------------------------------------


def test_solve_task_rotate_k2_uses_dsl_engine():
    """A Rotate(k=2) task: solve_task returns the correct rotated grid AND
    the DSL leg independently produced a Program that reproduces the
    correct training outputs (proving the DSL engine participated)."""
    train_pairs, test_input, expected = _rotate_k2_task()

    # Sanity: the public solve_task contract is preserved.
    a1, a2 = solve_task(train_pairs, test_input)
    assert isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray)
    assert np.array_equal(a1, expected), (
        f"Rotate(k=2): attempt_1 should be the 180-rotation; "
        f"got\n{a1}\nexpected\n{expected}"
    )

    # The extended API exposes the DSL programs that made it into the beam.
    a1b, a2b, dsl_progs = solve_task_with_dsl_programs(train_pairs, test_input)
    assert np.array_equal(a1, a1b)
    assert np.array_equal(a2, a2b)
    assert len(dsl_progs) > 0, (
        "DSL engine should have surfaced at least one program for a "
        "Rotate(k=2) task — synth covers Rotate {1,2,3}."
    )

    # At least one DSL program must reproduce the training outputs perfectly
    # — proving the DSL leg independently found the rotation, regardless
    # of whether the hand-rule beam also did.
    correct_dsl = []
    for p in dsl_progs:
        assert isinstance(p, Program)
        ok = True
        for inp, out in train_pairs:
            try:
                pred = evaluate(p, inp)
            except Exception:
                ok = False
                break
            if not isinstance(pred, np.ndarray) or not np.array_equal(pred, out):
                ok = False
                break
        if ok:
            correct_dsl.append(p)
    assert len(correct_dsl) >= 1, (
        "At least one DSL program must reproduce every Rotate(k=2) train "
        "pair exactly — that's the evidence the DSL engine solved this task."
    )


# ---------------------------------------------------------------------------
# Required test 2: Recolor regression — legacy case still solves.
# ---------------------------------------------------------------------------


def test_solve_task_recolor_still_picks_right_answer():
    """Regression: the Recolor case that solved before the DSL bolt-on must
    still solve afterward. The hand-rule beam covers this directly; the DSL
    bolt-on must not knock it off the top of the merged beam."""
    train_pairs, test_input, expected = _recolor_task()
    a1, a2 = solve_task(train_pairs, test_input)
    assert np.array_equal(a1, expected), (
        f"Recolor regression: attempt_1 must apply 1->2 to the test input. "
        f"got\n{a1}\nexpected\n{expected}"
    )


# ---------------------------------------------------------------------------
# Required test 3: per-task time budget.
# ---------------------------------------------------------------------------


def test_solve_task_respects_per_task_time_budget():
    """solve_task with total_time_budget_s must complete within roughly that
    budget — proving the hand-rule + DSL legs together honor the cap.

    Use an adversarial task (random unfittable pairs) so the DSL leg can't
    converge early and we actually test the budget enforcement path."""
    rng = np.random.default_rng(seed=99)
    train_pairs = [
        (rng.integers(0, 5, (4, 4), dtype=np.int32),
         rng.integers(0, 5, (4, 4), dtype=np.int32))
        for _ in range(3)
    ]
    test_input = rng.integers(0, 5, (4, 4), dtype=np.int32)

    total_budget = 0.5
    # We allow the underlying synth to want up to 5s; the integration must
    # squeeze it into the total budget.
    t0 = time.monotonic()
    a1, a2 = solve_task(
        train_pairs, test_input,
        dsl_time_budget_s=5.0,
        total_time_budget_s=total_budget,
    )
    elapsed = time.monotonic() - t0

    # Allow some slack for OS jitter / final scoring outside the deadline
    # window. The contract is hand_rule_time + dsl_time < total_budget,
    # so total_budget plus a small constant covers per-platform overhead.
    tolerance = 0.75
    assert elapsed < total_budget + tolerance, (
        f"per-task budget violation: elapsed={elapsed:.2f}s, "
        f"budget={total_budget}s, tolerance={tolerance}s"
    )
    # Sanity: the solver still returned valid grids.
    assert isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray)
    assert a1.shape == test_input.shape
    assert a2.shape == test_input.shape


# ---------------------------------------------------------------------------
# Required test 4: two distinct attempts when possible.
# ---------------------------------------------------------------------------


def test_solve_task_returns_two_distinct_attempts_when_possible():
    """A task where multiple non-equivalent programs fit the train pairs
    must yield two attempts that differ on the test input."""
    # Rotate(k=2) train pairs admit Rotate(k=2) AND also fit when the
    # underlying grid happens to be symmetric. Use a deliberately
    # asymmetric grid so attempt_2 cannot collapse to attempt_1.
    train_pairs, test_input, expected = _rotate_k2_task(n=3, seed=33)
    # Sanity: the test input is NOT 180-symmetric, so Rotate(k=2)(test) != test.
    assert not np.array_equal(np.rot90(test_input, k=2), test_input), (
        "Test setup expects an asymmetric test grid."
    )

    a1, a2 = solve_task(train_pairs, test_input)
    assert a1.shape == test_input.shape
    assert a2.shape == test_input.shape
    assert not np.array_equal(a1, a2), (
        f"two attempts must differ on this asymmetric test input. "
        f"got attempt_1=\n{a1}\nattempt_2=\n{a2}"
    )


# ---------------------------------------------------------------------------
# Required test 5: DSL programs recorded for resonance update.
# ---------------------------------------------------------------------------


def test_dsl_programs_surfaced_for_resonance_library_update():
    """The DSL-found programs are exposed via solve_task_with_dsl_programs
    so the resonance updater can record them as self-solved entries."""
    train_pairs, test_input, expected = _rotate_k2_task()
    a1, a2, dsl_progs = solve_task_with_dsl_programs(train_pairs, test_input)

    assert isinstance(dsl_progs, list), \
        f"dsl_programs must be a list, got {type(dsl_progs).__name__}"
    assert len(dsl_progs) > 0, (
        "Rotate(k=2) task: DSL leg must surface at least one program for "
        "the resonance updater to record."
    )
    for p in dsl_progs:
        assert isinstance(p, Program), \
            f"each entry must be a dsl.Program; got {type(p).__name__}"
        # Each surfaced program must be evaluable on the first train input
        # (the resonance library's record_solved path will evaluate it).
        out = evaluate(p, train_pairs[0][0])
        assert isinstance(out, np.ndarray), (
            "DSL program returned for resonance recording must be Grid->Grid; "
            f"got output of type {type(out).__name__}"
        )

    # The list should be deduplicated by AST hash — no two entries with the
    # same sha256_hash. (Dedup happens inside solve_task_with_dsl_programs.)
    hashes = [p.sha256_hash() for p in dsl_progs]
    assert len(hashes) == len(set(hashes)), \
        f"DSL programs must be deduplicated by AST hash; got {hashes}"


# ---------------------------------------------------------------------------
# Extra: ablation — use_dsl=False reproduces legacy behaviour.
# ---------------------------------------------------------------------------


def test_use_dsl_false_skips_dsl_leg_entirely():
    """Ablation switch: use_dsl=False reverts to pure hand-rule behaviour
    and must (a) still solve a Recolor task and (b) surface zero DSL
    programs in the extended API."""
    train_pairs, test_input, expected = _recolor_task()
    a1, a2 = solve_task(train_pairs, test_input, use_dsl=False)
    assert np.array_equal(a1, expected)

    a1b, a2b, dsl_progs = solve_task_with_dsl_programs(
        train_pairs, test_input, use_dsl=False
    )
    assert np.array_equal(a1, a1b)
    assert dsl_progs == [], (
        f"use_dsl=False must yield zero DSL programs; got {dsl_progs}"
    )


# ---------------------------------------------------------------------------
# Extra: zero DSL budget is honored.
# ---------------------------------------------------------------------------


def test_zero_dsl_time_budget_skips_dsl_leg():
    """dsl_time_budget_s=0 disables the DSL leg without crashing — useful
    for tight per-task budgets where the hand-rule beam alone is enough."""
    train_pairs, test_input, expected = _recolor_task()
    a1, a2, dsl_progs = solve_task_with_dsl_programs(
        train_pairs, test_input, dsl_time_budget_s=0.0
    )
    assert dsl_progs == []
    # Hand-rule beam still solves the recolor task.
    assert np.array_equal(a1, expected)


# ---------------------------------------------------------------------------
# Extra: identity-only task — honest-abstain still works.
# ---------------------------------------------------------------------------


def test_identity_only_task_returns_identity_attempts():
    """When the only fitting program is Identity, attempt_1 must equal the
    test input (the hand-rule beam's null hypothesis). The DSL leg also
    finds Identity, but the merged beam de-duplicates so this still ends in
    the same place — two attempts, both shape-correct, attempt_1 == input."""
    train_pairs, test_input = _identity_task()
    a1, a2 = solve_task(train_pairs, test_input)
    assert np.array_equal(a1, test_input), (
        f"identity task: attempt_1 should equal test input; got\n{a1}"
    )
    assert a2.shape == test_input.shape


# ---------------------------------------------------------------------------
# Extra: default budget constants are sane.
# ---------------------------------------------------------------------------


def test_default_dsl_time_budget_matches_brief():
    """The brief specifies time_budget_s=2.0 for the DSL leg. The default
    constant exported from arc2_solver must match — this catches accidental
    drift in the integration defaults."""
    assert DSL_DEFAULT_TIME_BUDGET_S == 2.0, (
        f"DSL default time budget drift: expected 2.0, "
        f"got {DSL_DEFAULT_TIME_BUDGET_S}"
    )
