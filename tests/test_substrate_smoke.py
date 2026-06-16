"""Smoke tests for the priors-only substrate modules.

These tests do NOT import arcengine (only available in the Kaggle eval env).
They exercise perceptor / episode / fingerprint / resonance via direct calls
to confirm the substrate is internally consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.perceptor import perceive_grid, perceive_frame, grid_diff
from misfit_agent.episode import EpisodeTracker
from misfit_agent.fingerprint import fingerprint_episode, cosine, FINGERPRINT_DIM
from misfit_agent.resonance import ResonanceLibrary


def test_perceive_grid_single_object():
    grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    scene = perceive_grid(grid)
    assert scene.rows == 4
    assert scene.cols == 4
    assert scene.background_color == 0
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.color == 1
    assert obj.area == 4
    assert obj.bbox == (1, 1, 2, 2)
    assert obj.touches_edge is False
    assert obj.v_symmetric is True
    assert obj.h_symmetric is True


def test_perceive_grid_multiple_objects_sorted_by_area():
    grid = np.array([
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    scene = perceive_grid(grid)
    assert len(scene.objects) == 2
    # Largest first — the 3-colored 2x3 block
    assert scene.objects[0].color == 3
    assert scene.objects[0].area == 6
    assert scene.objects[1].color == 2
    assert scene.objects[1].area == 1


def test_perceive_grid_touches_edge():
    grid = np.array([
        [5, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    scene = perceive_grid(grid)
    assert len(scene.objects) == 1
    assert scene.objects[0].touches_edge is True


def test_perceive_frame_handles_3d_input():
    frame = np.array([
        [[0, 1], [1, 0]],
    ], dtype=np.int32)
    scene = perceive_frame(frame)
    assert scene.rows == 2
    assert scene.cols == 2


def test_grid_diff_same_grid_is_zero():
    a = np.array([[0, 1], [1, 0]], dtype=np.int32)
    changed, bg_to_fg, fg_to_bg = grid_diff(a, a)
    assert changed == 0
    assert bg_to_fg == 0
    assert fg_to_bg == 0


def test_grid_diff_detects_changes():
    a = np.array([[0, 0], [1, 1]], dtype=np.int32)
    b = np.array([[1, 0], [1, 0]], dtype=np.int32)
    changed, bg_to_fg, fg_to_bg = grid_diff(a, b)
    assert changed == 2
    assert bg_to_fg == 1
    assert fg_to_bg == 1


def test_fingerprint_empty_tracker_is_zeros():
    tracker = EpisodeTracker(game_id="empty")
    fp = fingerprint_episode(tracker)
    assert fp.shape == (FINGERPRINT_DIM,)
    assert float(np.abs(fp).sum()) == 0.0


def test_fingerprint_identical_episodes_have_high_cosine():
    grid = np.array([[0, 1], [1, 0]], dtype=np.int32)
    t1 = EpisodeTracker(game_id="g1")
    t2 = EpisodeTracker(game_id="g2")
    for t in (t1, t2):
        scene = perceive_grid(grid)
        class FakeFrame:
            levels_completed = 0
            state = "PLAY"
        t.scenes.append(scene)
        t.scenes.append(scene)
    fp1 = fingerprint_episode(t1)
    fp2 = fingerprint_episode(t2)
    sim = cosine(fp1, fp2)
    assert sim > 0.999


def test_fingerprint_distinct_episodes_have_lower_cosine():
    g1 = np.array([[0, 1], [1, 0]], dtype=np.int32)
    g2 = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.int32)
    t1 = EpisodeTracker(game_id="g1")
    t2 = EpisodeTracker(game_id="g2")
    t1.scenes.append(perceive_grid(g1))
    t2.scenes.append(perceive_grid(g2))
    fp1 = fingerprint_episode(t1)
    fp2 = fingerprint_episode(t2)
    sim = cosine(fp1, fp2)
    assert sim < 0.95


def test_resonance_library_roundtrip_with_tmp(tmp_path):
    from misfit_agent.episode import ActionRecord
    lib_path = tmp_path / "lib.jsonl"
    lib = ResonanceLibrary.load_or_create(str(lib_path))
    assert lib.entries == []
    fp = np.linspace(0.1, 0.5, FINGERPRINT_DIM, dtype=np.float32)
    policy = [
        ActionRecord("ACTION1", 1, {}, pre_levels_completed=0,
                     post_levels_completed=0, cells_changed=2),
        ActionRecord("ACTION6", 6, {"x": 10, "y": 20}, pre_levels_completed=0,
                     post_levels_completed=1, cells_changed=4, triggered_win=True),
    ]
    lib.record_solved(fp, policy, composite_score=0.75, game_id="testgame")
    assert lib.flush_to_disk() == 1
    lib2 = ResonanceLibrary.load_or_create(str(lib_path))
    assert len(lib2.entries) == 1
    assert lib2.entries[0].game_id == "testgame"
    assert lib2.entries[0].source == "self-solved"


def test_resonance_library_rejects_non_self_solved():
    import pytest
    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    lib = ResonanceLibrary.load_or_create("/tmp/never_written.jsonl")
    with pytest.raises(ValueError):
        lib.record_solved(fp, [], composite_score=0.0, source="public-arc-seeded")


def test_resonance_library_returns_top_k_seeds():
    import tempfile
    from misfit_agent.episode import ActionRecord
    with tempfile.TemporaryDirectory() as td:
        lib_path = Path(td) / "lib.jsonl"
        lib = ResonanceLibrary.load_or_create(str(lib_path))
        for i in range(3):
            fp = np.full(FINGERPRINT_DIM, float(i + 1) * 0.1, dtype=np.float32)
            policy = [ActionRecord(f"ACTION{i+1}", i + 1, {}, pre_levels_completed=0)]
            lib.record_solved(fp, policy, composite_score=1.0, game_id=f"g{i}")
        query = np.full(FINGERPRINT_DIM, 0.2, dtype=np.float32)
        seeds = lib.retrieve_policy_seeds(query, k=2)
        assert len(seeds) == 2
