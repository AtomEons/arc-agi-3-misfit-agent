"""Click quantizer tests — 4096-cell collapse to priors-derived candidates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.click_quantizer import click_candidates, best_click_candidate
from misfit_agent.perceptor import perceive_grid


def test_click_candidates_use_object_centroids_first():
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ], dtype=np.int32)
    scene = perceive_grid(grid)
    cands = click_candidates(scene)
    # First candidate should be the LARGER object's centroid
    assert cands[0].source == "centroid"
    # 2x2 block at (1,1)-(2,2) has centroid (1.5, 1.5) → click (2, 2) after round
    assert cands[0].x == 2
    assert cands[0].y == 2


def test_click_candidates_clip_to_grid_bounds():
    grid = np.zeros((64, 64), dtype=np.int32)
    grid[63, 63] = 7
    scene = perceive_grid(grid)
    cands = click_candidates(scene)
    for c in cands:
        assert 0 <= c.x <= 63
        assert 0 <= c.y <= 63


def test_click_candidates_emit_quadrant_fallback_for_empty_grid():
    grid = np.zeros((10, 10), dtype=np.int32)
    scene = perceive_grid(grid)
    cands = click_candidates(scene)
    # No objects detected → quadrant fallbacks present
    sources = {c.source for c in cands}
    assert "quadrant_fallback" in sources
    # 9 quadrants → at least 9 candidates after dedup
    quadrant_cands = [c for c in cands if c.source == "quadrant_fallback"]
    assert len(quadrant_cands) >= 4


def test_click_candidates_dedup_by_coordinate():
    grid = np.zeros((5, 5), dtype=np.int32)
    scene = perceive_grid(grid)
    cands = click_candidates(scene)
    coords = [(c.x, c.y) for c in cands]
    assert len(coords) == len(set(coords))


def test_click_candidates_collapse_factor_at_least_50x():
    """The whole point: collapse 4096 candidates to a sane fraction."""
    grid = np.zeros((64, 64), dtype=np.int32)
    grid[10:13, 10:13] = 4    # one 3x3 object
    grid[30:31, 50:51] = 7    # one single cell
    scene = perceive_grid(grid)
    cands = click_candidates(scene)
    assert len(cands) < 4096 / 50, (
        f"expected <82 candidates for 64x64 collapse, got {len(cands)}"
    )


def test_best_click_candidate_picks_centroid_when_no_seeds():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 3
    scene = perceive_grid(grid)
    best = best_click_candidate(scene)
    assert best.source == "centroid"


def test_best_click_candidate_biases_toward_seed_xy():
    grid = np.zeros((20, 20), dtype=np.int32)
    grid[2:4, 2:4] = 1     # small object near top-left
    grid[15:18, 15:18] = 2  # bigger object near bottom-right
    scene = perceive_grid(grid)
    # Top-left object centroid ≈ (2,2), bottom-right ≈ (16,16)
    # Seed near top-left should pull selection away from bigger centroid.
    seed = [(3, 3)]
    best = best_click_candidate(scene, policy_seeds_xy=seed)
    # Distance from chosen to (3,3) should be < distance to bottom-right object
    chosen_dist_to_seed = (best.x - 3) ** 2 + (best.y - 3) ** 2
    chosen_dist_to_br = (best.x - 16) ** 2 + (best.y - 16) ** 2
    assert chosen_dist_to_seed < chosen_dist_to_br
