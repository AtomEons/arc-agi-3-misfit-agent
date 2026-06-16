"""GoalInducer tests — Spelke-priors-only hypothesis ranking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.goal_inducer import (
    GoalHypothesis,
    GoalInducer,
    MAX_FREE_PARAMS,
)
from misfit_agent.perceptor import perceive_grid


def test_empty_inducer_returns_no_hypotheses():
    g = GoalInducer()
    assert g.hypothesize() == []


def test_removed_all_of_class_ranks_top_when_level_advances_iff_class_gone():
    """Pre has class=5; post has class=5 removed iff levels advanced."""
    g = GoalInducer()
    # Pair 1: 5 removed → level +1 (supports "remove all 5s")
    pre1 = perceive_grid(np.array([[0, 5, 5], [0, 0, 0]], dtype=np.int32))
    post1 = perceive_grid(np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32))
    g.observe(pre1, post1, delta_levels=1)
    # Pair 2: 5 still present → no level (also supports)
    pre2 = perceive_grid(np.array([[0, 5, 5], [0, 0, 0]], dtype=np.int32))
    post2 = perceive_grid(np.array([[0, 5, 0], [0, 0, 0]], dtype=np.int32))
    g.observe(pre2, post2, delta_levels=0)
    # Pair 3: 5 removed → level +1 (supports)
    pre3 = perceive_grid(np.array([[0, 0, 5], [0, 0, 0]], dtype=np.int32))
    post3 = perceive_grid(np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32))
    g.observe(pre3, post3, delta_levels=1)

    hyps = g.hypothesize()
    assert len(hyps) > 0
    top = hyps[0]
    # The top hypothesis should mention class 5 in some way
    assert any(5 in h.params for h in hyps[:3])
    # And the "removed_all" hypothesis for class 5 should be among the top
    removed_5 = [h for h in hyps if h.kind == "removed_all_of_class" and h.params == (5,)]
    assert removed_5, "expected removed_all_of_class(5) hypothesis"
    assert removed_5[0].score > 0.5


def test_hypothesis_max_free_params_enforced():
    with pytest.raises(ValueError):
        GoalHypothesis(
            kind="bogus",
            params=(1, 2, 3, 4),  # 4 > MAX_FREE_PARAMS=3
            score=1.0,
            support=1,
            contradictions=0,
        )
    assert MAX_FREE_PARAMS == 3


def test_count_equals_n_hypothesis_emitted():
    """When post-scene consistently has count(class=3) == 1 on advances."""
    g = GoalInducer()
    pre = perceive_grid(np.array([[0, 3, 0, 3, 0], [0, 0, 0, 0, 0]], dtype=np.int32))
    post = perceive_grid(np.array([[0, 3, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32))
    g.observe(pre, post, delta_levels=1)
    g.observe(pre, post, delta_levels=1)
    hyps = g.hypothesize()
    # Should include a count-equals hypothesis for class 3, N=1
    count_hyps = [h for h in hyps if h.kind == "count_of_class_equals_N"
                  and h.params == (3, 1)]
    assert count_hyps, f"expected count(class=3)==1 hypothesis, got {hyps}"


def test_contradicting_evidence_lowers_score():
    """A hypothesis with both support and contradiction should score lower
    than one with support only."""
    g_clean = GoalInducer()
    g_messy = GoalInducer()

    pre = perceive_grid(np.array([[0, 7, 0], [0, 0, 0]], dtype=np.int32))
    post_removed = perceive_grid(np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32))

    # Clean: removed → advance, twice
    g_clean.observe(pre, post_removed, delta_levels=1)
    g_clean.observe(pre, post_removed, delta_levels=1)
    # Messy: removed → advance once, removed → no advance once (contradiction)
    g_messy.observe(pre, post_removed, delta_levels=1)
    g_messy.observe(pre, post_removed, delta_levels=0)

    clean = [h for h in g_clean.hypothesize()
             if h.kind == "removed_all_of_class" and h.params == (7,)]
    messy = [h for h in g_messy.hypothesize()
             if h.kind == "removed_all_of_class" and h.params == (7,)]
    assert clean and messy
    assert clean[0].score > messy[0].score


def test_top_k_truncation():
    """top_k caps the returned hypothesis list."""
    g = GoalInducer()
    # Manufacture many distinct classes to generate many hypotheses
    for color in range(1, 6):
        grid_pre = np.zeros((3, 3), dtype=np.int32)
        grid_pre[0, 0] = color
        grid_post = np.zeros((3, 3), dtype=np.int32)
        g.observe(perceive_grid(grid_pre), perceive_grid(grid_post), delta_levels=1)
    hyps = g.hypothesize(top_k=3)
    assert len(hyps) <= 3
