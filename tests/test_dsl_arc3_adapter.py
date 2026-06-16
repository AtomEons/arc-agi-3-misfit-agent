"""Tests for the ARC-AGI-3 DSL adapter (Arc3DslAdapter).

Coverage:
  - empty adapter initialises with empty buffer + no cached programs
  - observations accumulate before the distinct-action threshold
  - cached programs appear once >= min_distinct_actions distinct actions seen
  - predict_next_state returns (np.ndarray, float-confidence)
  - predict_next_state on uncached action falls back to identity, conf 0
  - best_action returns an int that IS in available_actions
  - best_action falls back to random when no programs cached
  - time budget is enforced (resynth bounded by groups * budget)
  - identity action observation produces an identity-like program
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from misfit_agent.dsl.arc3_adapter import (
    ACTION_IDS,
    RESET_ACTION,
    Arc3DslAdapter,
)


@pytest.fixture
def base_grid() -> np.ndarray:
    return np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ],
        dtype=np.int64,
    )


# ----------------------------------------------------------------- init


def test_adapter_initializes_empty():
    adapter = Arc3DslAdapter()
    assert adapter.buffer_size == 0
    assert adapter.cached_action_ids == []


def test_adapter_accepts_custom_time_budget():
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.05)
    assert adapter.time_budget_per_action_s == pytest.approx(0.05)


# ----------------------------------------------------------------- observe


def test_single_observation_does_not_trigger_synthesis(base_grid):
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.05)
    adapter.observe(base_grid, RESET_ACTION, base_grid)
    assert adapter.buffer_size == 1
    # one distinct action < min_distinct_actions=3 by default
    assert adapter.cached_action_ids == []


def test_below_threshold_distinct_actions_no_synthesis(base_grid):
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.05, min_distinct_actions=3)
    adapter.observe(base_grid, 0, base_grid)
    adapter.observe(base_grid, 1, base_grid)
    assert adapter.buffer_size == 2
    assert adapter.cached_action_ids == []


def test_five_observations_with_distinct_actions_caches_programs(base_grid):
    """After 5 observations covering >=3 distinct actions, programs are cached."""
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2, min_distinct_actions=3)
    # identity transitions for actions 0, 1, 2 (three distinct types)
    adapter.observe(base_grid, 0, base_grid)
    adapter.observe(base_grid, 1, base_grid)
    adapter.observe(base_grid, 2, base_grid)
    adapter.observe(base_grid, 0, base_grid)
    adapter.observe(base_grid, 1, base_grid)
    assert adapter.buffer_size == 5
    cached = adapter.cached_action_ids
    assert len(cached) >= 3, f"expected >=3 cached programs, got {cached}"
    assert 0 in cached
    assert 1 in cached
    assert 2 in cached


# ----------------------------------------------------------------- predict


def test_predict_next_state_returns_grid_and_confidence(base_grid):
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2)
    # train: action 0/1/2 = identity
    for a in (0, 1, 2):
        for _ in range(2):
            adapter.observe(base_grid, a, base_grid)
    pred, conf = adapter.predict_next_state(base_grid, 0)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == base_grid.shape
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_predict_uncached_action_returns_identity_zero_conf(base_grid):
    adapter = Arc3DslAdapter()
    pred, conf = adapter.predict_next_state(base_grid, 7)
    assert np.array_equal(pred, base_grid)
    assert conf == 0.0


def test_identity_transitions_yield_high_confidence(base_grid):
    """When every transition for action a is identity, the cached program
    for action a must predict the input grid with high confidence."""
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.3)
    for a in (0, 1, 2):
        for _ in range(2):
            adapter.observe(base_grid, a, base_grid)
    pred, conf = adapter.predict_next_state(base_grid, 0)
    assert np.array_equal(pred, base_grid)
    assert conf >= 0.99


# ----------------------------------------------------------------- best_action


def test_best_action_picks_from_available(base_grid):
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2)
    for a in (0, 1, 2):
        for _ in range(2):
            adapter.observe(base_grid, a, base_grid)
    available = [0, 1, 2]
    chosen = adapter.best_action(base_grid, available)
    assert isinstance(chosen, int)
    assert chosen in available


def test_best_action_falls_back_to_random_when_no_cache(base_grid):
    adapter = Arc3DslAdapter(rng_seed=42)
    available = [3, 4, 5]
    chosen = adapter.best_action(base_grid, available)
    assert chosen in available


def test_best_action_random_fallback_is_deterministic(base_grid):
    a1 = Arc3DslAdapter(rng_seed=7).best_action(base_grid, [1, 2, 3])
    a2 = Arc3DslAdapter(rng_seed=7).best_action(base_grid, [1, 2, 3])
    assert a1 == a2


def test_best_action_empty_raises(base_grid):
    adapter = Arc3DslAdapter()
    with pytest.raises(ValueError):
        adapter.best_action(base_grid, [])


# ----------------------------------------------------------------- budget


def test_time_budget_enforced_on_resynth(base_grid):
    """Adapter must not spend more than groups * budget on resynth.

    Three distinct actions @ 0.1s each => wall-clock for the triggering
    observe() call should be < ~1.5s with generous slack.
    """
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.1, min_distinct_actions=3)
    # prime first two distinct actions without triggering synth
    adapter.observe(base_grid, 0, base_grid)
    adapter.observe(base_grid, 1, base_grid)
    t0 = time.monotonic()
    adapter.observe(base_grid, 2, base_grid)  # triggers resynth across 3 groups
    dt = time.monotonic() - t0
    # 3 groups * 0.1s budget = 0.3s soft cap; allow 5x slack for CI noise
    assert dt < 1.5, f"resynth took {dt:.3f}s, exceeds 5x soft cap"


def test_predict_respects_time_budget(base_grid):
    """predict_next_state on a cached program must return quickly."""
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2)
    for a in (0, 1, 2):
        for _ in range(2):
            adapter.observe(base_grid, a, base_grid)
    t0 = time.monotonic()
    adapter.predict_next_state(base_grid, 0)
    dt = time.monotonic() - t0
    # interpreter eval on a 3x3 grid is microseconds; cap at 1s
    assert dt < 1.0, f"predict took {dt:.3f}s"


# ----------------------------------------------------------------- introspect


def test_buffer_size_tracks_observations(base_grid):
    adapter = Arc3DslAdapter()
    assert adapter.buffer_size == 0
    adapter.observe(base_grid, 0, base_grid)
    adapter.observe(base_grid, 1, base_grid)
    assert adapter.buffer_size == 2


def test_cached_action_ids_sorted(base_grid):
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2)
    for a in (5, 2, 4):
        for _ in range(2):
            adapter.observe(base_grid, a, base_grid)
    cached = adapter.cached_action_ids
    assert cached == sorted(cached)


# ----------------------------------------------------------------- robustness


def test_observe_accepts_lists_not_just_arrays(base_grid):
    """The agent loop hands us nested lists from FrameData.frame — adapter
    must normalise to ndarray internally."""
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2)
    grid_list = base_grid.tolist()
    for a in (0, 1, 2):
        for _ in range(2):
            adapter.observe(grid_list, a, grid_list)
    pred, conf = adapter.predict_next_state(grid_list, 0)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == base_grid.shape


def test_predict_handles_unknown_action_gracefully(base_grid):
    adapter = Arc3DslAdapter(time_budget_per_action_s=0.2)
    for a in (0, 1, 2):
        for _ in range(2):
            adapter.observe(base_grid, a, base_grid)
    # action 99 was never observed
    pred, conf = adapter.predict_next_state(base_grid, 99)
    assert np.array_equal(pred, base_grid)
    assert conf == 0.0
