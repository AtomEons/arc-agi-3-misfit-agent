"""AbstainPolicy tests — derived floor, plateau, and WM-variance triggers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.abstain_policy import AbstainPolicy, HUMAN_BASELINE_MULTIPLIER
from misfit_agent.config import CONFIG
from misfit_agent.episode import ActionRecord, EpisodeTracker
from misfit_agent.fingerprint import FINGERPRINT_DIM
from misfit_agent.perceptor import perceive_grid
from misfit_agent.world_model import WorldModel


def _make_tracker_with_actions(n: int) -> EpisodeTracker:
    t = EpisodeTracker(game_id="test")
    grid = np.zeros((3, 3), dtype=np.int32)
    scene = perceive_grid(grid)
    t.scenes.append(scene)
    for i in range(n):
        t.scenes.append(scene)
        rec = ActionRecord(
            action_name="ACTION1",
            action_value=1,
            data={},
            pre_levels_completed=0,
            post_levels_completed=0,
            cells_changed=0,
            triggered_win=False,
        )
        t.action_history.append(rec)
    return t


def test_min_actions_uses_config_floor_when_no_baseline():
    p = AbstainPolicy()
    assert p.min_actions == CONFIG.abstain.min_actions_before_abstain


def test_min_actions_derived_from_human_baseline():
    p = AbstainPolicy(estimated_human_baseline=20)
    # Derived = 2 * 20 = 40; floor = 25; max(40, 25) = 40
    expected = max(
        CONFIG.abstain.min_actions_before_abstain,
        HUMAN_BASELINE_MULTIPLIER * 20,
    )
    assert p.min_actions == expected
    assert HUMAN_BASELINE_MULTIPLIER == 2  # documents the scoring-math derivation


def test_does_not_abstain_before_min_actions():
    p = AbstainPolicy()
    t = _make_tracker_with_actions(3)  # well below floor
    wm = WorldModel()
    assert p.should_abstain(t, wm) is False
    assert "min_actions" in p.reason(t, wm)


def test_does_not_abstain_without_plateau():
    p = AbstainPolicy(plateau_window_k=3, plateau_delta_threshold=0.01)
    t = _make_tracker_with_actions(CONFIG.abstain.min_actions_before_abstain + 5)
    wm = WorldModel()
    # Push fingerprints that change a LOT — no plateau
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        p.push_fingerprint(rng.normal(size=FINGERPRINT_DIM).astype(np.float32))
    assert p.should_abstain(t, wm) is False
    assert "novelty" in p.reason(t, wm).lower()


def test_abstains_when_all_three_conditions_hold():
    """Actions past floor + plateau + WM variance high → abstain."""
    p = AbstainPolicy(plateau_window_k=3, plateau_delta_threshold=0.01)
    t = _make_tracker_with_actions(CONFIG.abstain.min_actions_before_abstain + 5)
    # Plateau: identical fingerprints
    flat = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    for _ in range(5):
        p.push_fingerprint(flat.copy())

    # WM that always predicts no-change but observations all show change.
    # We force this via a fake WM whose predict() returns the input unchanged
    # at confidence=1.0, while every ActionRecord has cells_changed > 0.
    class _FakeWM:
        def predict(self, grid, action_name):
            return grid.copy(), 1.0

    # Mutate the tracker's action records to show change observed.
    for r in t.action_history:
        r.cells_changed = 3  # observed change, predicted no change → disagreement

    fake_wm = _FakeWM()
    assert p.should_abstain(t, fake_wm) is True
    assert "abstain" in p.reason(t, fake_wm)
