"""HungarianTracker tests — assignment, birth/death, fallback, idempotence."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.perceptor import perceive_grid
from misfit_agent.tracker_hungarian import HungarianTracker, TrackingCosts


def test_track_identity_when_objects_unchanged():
    """Two identical scenes → every prev maps to itself (or at least to
    some curr index with no births/deaths)."""
    g = np.array([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
    ], dtype=np.int32)
    scene = perceive_grid(g)
    tr = HungarianTracker()
    mapping = tr.track(scene, scene)
    # Every prev should be matched (no None values)
    assert all(v is not None for v in mapping.values())
    # No spawned, no destroyed
    assert tr.spawned_indices(scene, scene, mapping) == []
    assert tr.destroyed_indices(mapping) == []


def test_track_handles_destroyed_object():
    """An object present in prev but absent in curr → destroyed (None)."""
    g_pre = np.array([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    g_post = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    pre = perceive_grid(g_pre)
    post = perceive_grid(g_post)
    tr = HungarianTracker()
    mapping = tr.track(pre, post)
    # Exactly one prev object should be unmatched (the 2)
    destroyed = tr.destroyed_indices(mapping)
    assert len(destroyed) == 1
    # The destroyed object should have been color=2
    assert pre.objects[destroyed[0]].color == 2


def test_track_handles_spawned_object():
    """An object in curr that wasn't in prev → spawned (not in mapping values)."""
    g_pre = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    g_post = np.array([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    pre = perceive_grid(g_pre)
    post = perceive_grid(g_post)
    tr = HungarianTracker()
    mapping = tr.track(pre, post)
    spawned = tr.spawned_indices(pre, post, mapping)
    assert len(spawned) == 1
    assert post.objects[spawned[0]].color == 2


def test_track_is_idempotent():
    """track(a, b) twice yields identical results; no side effects on inputs."""
    g_pre = np.array([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    g_post = np.array([
        [0, 0, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    pre = perceive_grid(g_pre)
    post = perceive_grid(g_post)
    tr = HungarianTracker()
    m1 = tr.track(pre, post)
    m2 = tr.track(pre, post)
    assert m1 == m2
    # And inputs unmodified
    assert np.array_equal(pre.grid, g_pre)
    assert np.array_equal(post.grid, g_post)


def test_greedy_fallback_matches_scipy_for_simple_case():
    """Force greedy path and verify it produces a valid assignment."""
    g_pre = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0],
    ], dtype=np.int32)
    g_post = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 2],
    ], dtype=np.int32)
    pre = perceive_grid(g_pre)
    post = perceive_grid(g_post)

    tr_scipy = HungarianTracker(force_greedy=False)
    tr_greedy = HungarianTracker(force_greedy=True)
    m_scipy = tr_scipy.track(pre, post)
    m_greedy = tr_greedy.track(pre, post)
    # Both should match every prev object (no destroyed) since each is
    # close to exactly one curr object of the same color.
    assert all(v is not None for v in m_greedy.values())
    # Both solvers should match same-color objects to same-color objects.
    for prev_i, curr_j in m_greedy.items():
        assert curr_j is not None
        assert pre.objects[prev_i].color == post.objects[curr_j].color
    for prev_i, curr_j in m_scipy.items():
        assert curr_j is not None
        assert pre.objects[prev_i].color == post.objects[curr_j].color


def test_empty_prev_returns_empty_mapping():
    g_empty = np.zeros((3, 3), dtype=np.int32)
    g_full = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int32)
    pre = perceive_grid(g_empty)
    post = perceive_grid(g_full)
    tr = HungarianTracker()
    mapping = tr.track(pre, post)
    assert mapping == {}
    # All curr objects are spawned
    assert len(tr.spawned_indices(pre, post, mapping)) == len(post.objects)
