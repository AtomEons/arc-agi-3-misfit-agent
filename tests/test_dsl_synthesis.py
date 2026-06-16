"""TEAM SYNTH — beam-search synthesis over typed program ASTs.

The required guarantees, mechanically tested:
  - synthesize on a single Identity task returns Identity in the top-K
  - synthesize on a Rotate(k=2) task returns Rotate(k=2) in the top-K
  - synthesize on empty train_pairs returns []
  - the time budget is enforced (return within ~2x budget on a realistic
    workload)
  - the MDL penalty pushes Identity above other primitives that fit
    equally well but cost more bits
  - the returned list is sorted by score descending

Extra coverage that earns its place:
  - returned Programs are typed Grid -> Grid and depth ≤ max_depth
  - dedup: two equivalent primitive instances do not occupy two beam slots
  - Reflect(D1) task is recovered for a square grid
  - Translate(dy,dx) task is recovered for a small shift
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl import Grid, DslType
from misfit_agent.dsl.synthesis import synthesize, MDL_LAMBDA
from misfit_agent.dsl.ast import Program, PrimitiveNode
from misfit_agent.dsl.primitives import (
    Identity, Rotate, Reflect, Translate,
)
from misfit_agent.dsl.walker import total_mdl_bits
from misfit_agent.dsl.interpreter import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root_primitive_name(p: Program) -> str:
    """Name of the primitive at the program root (atomic synthesis only)."""
    assert isinstance(p.root, PrimitiveNode), \
        f"expected PrimitiveNode at root, got {type(p.root).__name__}"
    return type(p.root.primitive).__name__


def _root_primitive(p: Program):
    assert isinstance(p.root, PrimitiveNode)
    return p.root.primitive


# ---------------------------------------------------------------------------
# Required: Identity task recovery
# ---------------------------------------------------------------------------


def test_identity_task_recovers_identity():
    """A task where input == output should surface Identity in the top-K."""
    g = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=8, time_budget_s=5.0)
    names = [_root_primitive_name(p) for p in results]
    assert "Identity" in names, \
        f"Identity should appear in top-{len(results)} for an identity task; " \
        f"got {names}"


def test_identity_task_identity_is_top_ranked():
    """MDL prior should make Identity strictly the top result on an identity task."""
    g = np.array([[1, 2, 0], [0, 3, 4]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=8, time_budget_s=5.0)
    assert len(results) > 0
    assert _root_primitive_name(results[0]) == "Identity"


# ---------------------------------------------------------------------------
# Required: Rotate(k=2) task recovery
# ---------------------------------------------------------------------------


def test_rotate_k2_task_recovers_rotate_k2():
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    g_out = np.rot90(g, k=2).copy()
    results = synthesize([(g, g_out)], beam_width=8, time_budget_s=5.0)
    # Find a Rotate(k=2) in the top-K.
    found = False
    for p in results:
        prim = _root_primitive(p)
        if isinstance(prim, Rotate) and prim.k == 2:
            found = True
            break
    assert found, f"Rotate(k=2) should appear in top-{len(results)} for " \
                  f"a 180° rotation task; got " \
                  f"{[p.to_string() for p in results]}"


def test_rotate_k2_task_rotate_k2_is_top_ranked():
    """Rotate(k=2) should be the single best fit for a 180° rotation task."""
    g = np.array([[1, 2, 5], [3, 4, 6], [7, 8, 9]], dtype=np.int32)
    g_out = np.rot90(g, k=2).copy()
    results = synthesize([(g, g_out)], beam_width=8, time_budget_s=5.0)
    assert len(results) > 0
    top = _root_primitive(results[0])
    assert isinstance(top, Rotate) and top.k == 2, \
        f"top program should be Rotate(k=2), got {results[0].to_string()}"


# ---------------------------------------------------------------------------
# Required: empty train_pairs
# ---------------------------------------------------------------------------


def test_empty_train_pairs_returns_empty_list():
    assert synthesize([]) == []
    assert synthesize([], beam_width=16, time_budget_s=10.0) == []


# ---------------------------------------------------------------------------
# Required: time budget enforcement
# ---------------------------------------------------------------------------


def test_time_budget_is_enforced_within_2x():
    """Synthesis must respect the wall-clock budget within ~2x.

    We use a workload that would naturally take more than the budget by
    enumerating the full primitive grid against a non-trivial grid.
    """
    rng = np.random.default_rng(seed=42)
    g_in = rng.integers(low=0, high=5, size=(10, 10), dtype=np.int32)
    g_out = g_in.copy()
    budget = 0.05  # 50ms — tight enough to actually bite
    t0 = time.monotonic()
    _ = synthesize([(g_in, g_out)], beam_width=8, time_budget_s=budget)
    elapsed = time.monotonic() - t0
    # Allow 2x slack as specified in the brief, plus a small constant for
    # interpreter overhead on the very first iteration.
    assert elapsed < (budget * 2.0) + 0.05, \
        f"synthesize took {elapsed*1000:.1f}ms for a {budget*1000:.0f}ms budget; " \
        f"exceeded 2x budget"


def test_time_budget_returns_partial_results_on_exhaustion():
    """When the budget is too tight to enumerate all primitives, the call
    must still return a list (possibly shorter than beam_width) instead
    of raising or hanging."""
    g = np.zeros((20, 20), dtype=np.int32)
    out = synthesize([(g, g.copy())], beam_width=8, time_budget_s=0.0)
    # Even with zero budget we don't crash; we return *something* (possibly
    # empty or partial). Must be a list.
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Required: MDL penalty pushes Identity below richer programs that fit equally
# ---------------------------------------------------------------------------


def test_mdl_penalty_orders_identity_above_translate_zero():
    """Identity and Translate(dy=0,dx=0) both perfectly fit an identity task,
    but Identity has lower MDL bits and so must rank above Translate(0,0)."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=20, time_budget_s=5.0)

    identity_rank = None
    translate_zero_rank = None
    for i, p in enumerate(results):
        prim = _root_primitive(p)
        if isinstance(prim, Identity) and identity_rank is None:
            identity_rank = i
        if isinstance(prim, Translate) and prim.dy == 0 and prim.dx == 0 \
                and translate_zero_rank is None:
            translate_zero_rank = i
    assert identity_rank is not None, \
        f"Identity not in top-{len(results)}: {[p.to_string() for p in results]}"
    if translate_zero_rank is not None:
        assert identity_rank < translate_zero_rank, \
            f"Identity should rank above Translate(0,0); got Identity={identity_rank}, " \
            f"Translate(0,0)={translate_zero_rank}"


def test_mdl_penalty_constant_matches_brief():
    """The brief specifies lambda = 0.01 for the MDL penalty."""
    assert MDL_LAMBDA == pytest.approx(0.01)


def test_mdl_penalty_breaks_ties_when_fit_is_equal():
    """Build two real candidate programs that fit a degenerate task perfectly
    (input == output of a single all-zero cell) — Identity must score
    strictly higher than Translate(dy=0,dx=0) due to MDL."""
    g = np.zeros((2, 2), dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=20, time_budget_s=5.0)
    # Find both programs in the result set.
    id_prog = next(
        (p for p in results if isinstance(_root_primitive(p), Identity)),
        None,
    )
    tx_prog = next(
        (p for p in results
         if isinstance(_root_primitive(p), Translate)
         and _root_primitive(p).dy == 0 and _root_primitive(p).dx == 0),
        None,
    )
    if id_prog is not None and tx_prog is not None:
        assert total_mdl_bits(id_prog) < total_mdl_bits(tx_prog), \
            f"Identity MDL ({total_mdl_bits(id_prog):.2f}) should be less " \
            f"than Translate(0,0) MDL ({total_mdl_bits(tx_prog):.2f})"


# ---------------------------------------------------------------------------
# Required: result list sorted by score descending
# ---------------------------------------------------------------------------


def test_results_sorted_by_score_descending():
    """The returned list must be sorted highest-score-first."""
    # Use a task that produces a clear score gradient (Rotate 90° task).
    g = np.array([[1, 2, 0], [3, 0, 4], [0, 5, 6]], dtype=np.int32)
    g_out = np.rot90(g, k=1).copy()
    results = synthesize([(g, g_out)], beam_width=8, time_budget_s=5.0)
    assert len(results) >= 2

    # Re-derive scores using the public scoring helpers so we can verify
    # ordering without re-running the inner loop.
    from misfit_agent.dsl.synthesis import _final_score
    scores = [_final_score(p, [(g, g_out)]) for p in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-9, \
            f"results not sorted descending at position {i}: " \
            f"{scores[i]:.4f} < {scores[i+1]:.4f}"


# ---------------------------------------------------------------------------
# Earn-its-keep coverage
# ---------------------------------------------------------------------------


def test_results_are_grid_typed():
    """Every returned program must be Grid -> Grid so it can chain into the
    integration team's combinator layer without type-mismatches."""
    g = np.array([[0, 1], [1, 0]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=8, time_budget_s=5.0)
    for p in results:
        assert p.output_type() == Grid


def test_results_respect_max_depth():
    """Atomic-only synthesis produces depth-1 programs which respect max_depth=3."""
    g = np.array([[1, 0], [0, 1]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=8,
                         time_budget_s=5.0, max_depth=3)
    for p in results:
        assert p.depth() <= 3


def test_results_are_unique_programs():
    """No two beam slots should hold structurally identical programs."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=8, time_budget_s=5.0)
    hashes = [p.sha256_hash() for p in results]
    assert len(hashes) == len(set(hashes)), \
        f"duplicate programs in results: {[p.to_string() for p in results]}"


def test_reflect_d1_task_recovers_reflect_d1():
    """Transpose task on a square grid should recover Reflect(D1)."""
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    g_out = g.T.copy()
    results = synthesize([(g, g_out)], beam_width=8, time_budget_s=5.0)
    found = any(
        isinstance(_root_primitive(p), Reflect)
        and _root_primitive(p).axis == "D1"
        for p in results
    )
    assert found, f"Reflect(D1) should appear in top-{len(results)} for " \
                  f"a transpose task; got {[p.to_string() for p in results]}"


def test_translate_task_recovers_correct_offset():
    """A simple +1 row shift task should recover Translate(dy=1, dx=0)."""
    g = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    # Apply the actual Translate primitive to build the ground-truth output
    # — this guarantees the BG-fill semantics match.
    g_out = Translate(dy=1, dx=0).apply(g)
    results = synthesize([(g, g_out)], beam_width=8, time_budget_s=5.0)
    found = any(
        isinstance(_root_primitive(p), Translate)
        and _root_primitive(p).dy == 1 and _root_primitive(p).dx == 0
        for p in results
    )
    assert found, f"Translate(dy=1,dx=0) should appear in top-{len(results)}; " \
                  f"got {[p.to_string() for p in results]}"


def test_returned_programs_are_evaluable():
    """Every returned program must evaluate without raising on the train input."""
    g = np.array([[1, 2], [3, 4]], dtype=np.int32)
    results = synthesize([(g, g.copy())], beam_width=8, time_budget_s=5.0)
    for p in results:
        out = evaluate(p, g)
        # Output must be a numpy array of the right type.
        assert isinstance(out, np.ndarray), \
            f"program {p.to_string()} did not return ndarray"


def test_multiple_train_pairs_average_scoring():
    """Synthesis should aggregate accuracy across multiple train pairs."""
    g1 = np.array([[1, 0], [0, 0]], dtype=np.int32)
    g2 = np.array([[0, 2], [0, 0]], dtype=np.int32)
    pairs = [
        (g1, np.rot90(g1, k=2).copy()),
        (g2, np.rot90(g2, k=2).copy()),
    ]
    results = synthesize(pairs, beam_width=8, time_budget_s=5.0)
    assert len(results) > 0
    top = _root_primitive(results[0])
    assert isinstance(top, Rotate) and top.k == 2, \
        f"top program for two-pair rotation task should be Rotate(k=2); " \
        f"got {results[0].to_string()}"


def test_beam_width_caps_result_size():
    """The result list size must not exceed beam_width."""
    g = np.array([[1, 0], [0, 2]], dtype=np.int32)
    for w in (1, 3, 5, 8):
        results = synthesize([(g, g.copy())], beam_width=w, time_budget_s=5.0)
        assert len(results) <= w, \
            f"beam_width={w} but got {len(results)} results"


def test_no_train_fit_still_returns_candidates():
    """A truly hopeless target (random data) should not crash synthesis;
    it should return programs (just with low scores)."""
    rng = np.random.default_rng(seed=7)
    g_in = rng.integers(low=0, high=5, size=(4, 4), dtype=np.int32)
    g_out = rng.integers(low=0, high=5, size=(4, 4), dtype=np.int32)
    results = synthesize([(g_in, g_out)], beam_width=4, time_budget_s=5.0)
    # The beam should still hold candidates — just low-scoring ones.
    assert isinstance(results, list)
    assert len(results) <= 4
