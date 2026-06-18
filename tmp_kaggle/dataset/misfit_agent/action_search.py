"""Action selection — Tier-1 priors-only policy with world-model 1-step lookahead.

Day-3 → Day-4 upgrade from the substrate scaffold:
  - Honor `available_actions` strictly (action alphabet from the engine).
  - **NEW: World model 1-step lookahead** — for each candidate action, ask
    the world model to predict the next grid. Score by progress proxy
    (cells_changed > 0, perceived novelty, prior winning-policy match).
  - Prefer actions with positive level-advance rate in episode history.
  - Penalize known-dud actions (cells_changed == 0 in all observations).
  - **NEW: ClickQuantizer** for ACTION6 — collapse 4096-cell space to
    ~5-20 priors-derived candidates (object centroids + bbox corners +
    edge midpoints + quadrant fallback).
  - When resonance seeds are available, bias toward the action that prior
    winning policies took at this step index.
  - Otherwise fall back to weighted-random among non-dud actions.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

import numpy as np

from arcengine import GameAction  # type: ignore[import-not-found]

from .click_quantizer import best_click_candidate
from .episode import EpisodeTracker
from .perceptor import SceneObservation
from .world_model import WorldModel


def _name_of(action: Any) -> str:
    return getattr(action, "name", str(action))


def _action_dud_score(tracker: EpisodeTracker, name: str) -> float:
    """0.0 = no evidence of duddiness, 1.0 = certain dud (no effect ever)."""
    bucket = tracker.transition_signals.get(name)
    if not bucket or bucket["total"] == 0:
        return 0.0
    if bucket["cells_changed_sum"] == 0 and bucket["level_advances"] == 0:
        return 1.0
    return 0.0


def _action_advance_rate(tracker: EpisodeTracker, name: str) -> float:
    bucket = tracker.transition_signals.get(name)
    if not bucket or bucket["total"] == 0:
        return 0.0
    return bucket["level_advances"] / bucket["total"]


def _world_model_lookahead_score(
    world_model: Optional[WorldModel],
    grid: np.ndarray,
    action_name: str,
    tracker: EpisodeTracker,
) -> tuple[float, float]:
    """Return (score_bonus, confidence) for one-step lookahead under WM.

    Score bonus components (all priors-only):
      + 0.5 if the predicted grid differs from current (action is not a no-op)
      + 0.3 if predicted cells_changed > 0 AND action has zero observed advance
        (novel transition — worth probing under exploration)
      + 0.2 if the action has been seen before at high confidence and
        succeeded (cells_changed > 0 in prior observations)
    """
    if world_model is None:
        return 0.0, 0.0
    try:
        predicted, conf = world_model.predict(grid, action_name)
    except Exception:
        return 0.0, 0.0
    if conf <= 0.0:
        return 0.0, 0.0
    bonus = 0.0
    if not np.array_equal(predicted, grid):
        bonus += 0.5
        bucket = tracker.transition_signals.get(action_name)
        if not bucket or bucket["level_advances"] == 0:
            bonus += 0.3
        elif bucket["cells_changed_sum"] > 0:
            bonus += 0.2
    return bonus, conf


def select_action(
    scene: SceneObservation,
    tracker: EpisodeTracker,
    available_actions: Sequence[Any],
    policy_seeds: Sequence[Sequence[dict]],
    action_budget_remaining: int,
    world_model: Optional[WorldModel] = None,
) -> GameAction:
    """Return a GameAction from `available_actions` under Tier-1 priors only."""

    if not available_actions:
        tracker.record_action("RESET", int(GameAction.RESET.value), {},
                              pre_levels_completed=0)
        tracker.set_rationale("no available_actions reported; resetting")
        return GameAction.RESET

    # Step index → resonance seed votes.
    step_idx = len(tracker.action_history)
    seed_votes: dict[str, int] = {}
    seed_xy_hints: list[tuple[int, int]] = []
    for policy in policy_seeds:
        if step_idx < len(policy):
            entry = policy[step_idx]
            n = entry.get("action_name", "")
            if n:
                seed_votes[n] = seed_votes.get(n, 0) + 1
            data = entry.get("data") or {}
            if "x" in data and "y" in data:
                seed_xy_hints.append((int(data["x"]), int(data["y"])))

    # Score each available action.
    scored: list[tuple[float, Any, float]] = []  # (weight, action, wm_conf)
    for action in available_actions:
        name = _name_of(action)
        if _action_dud_score(tracker, name) >= 1.0 and action_budget_remaining > 5:
            continue
        weight = 1.0
        weight += 2.0 * _action_advance_rate(tracker, name)
        weight += 1.5 * seed_votes.get(name, 0)
        wm_bonus, wm_conf = _world_model_lookahead_score(
            world_model, scene.grid, name, tracker
        )
        weight += wm_bonus
        scored.append((weight, action, wm_conf))

    if not scored:
        action = random.choice(list(available_actions))
        chosen_name = _name_of(action)
        rationale = "all-actions-known-dud; random tiebreak"
        wm_conf_chosen = 0.0
    else:
        max_w = max(w for w, _, _ in scored)
        top = [(a, c) for w, a, c in scored if w == max_w]
        action, wm_conf_chosen = random.choice(top)
        chosen_name = _name_of(action)
        rationale = (
            f"priors+wm pick: advance={_action_advance_rate(tracker, chosen_name):.2f}, "
            f"seed_votes={seed_votes.get(chosen_name, 0)}, "
            f"wm_conf={wm_conf_chosen:.2f}"
        )

    # ACTION6 — quantize click coordinate via objectness prior, biased by
    # seed (x,y) hints if any.
    data: dict = {}
    if hasattr(action, "is_complex") and action.is_complex():
        cand = best_click_candidate(scene, policy_seeds_xy=seed_xy_hints or None)
        data = {"x": cand.x, "y": cand.y}
        action.set_data(data)
        rationale += f"; click=({cand.x},{cand.y}) {cand.source} [{cand.rationale}]"

    pre_lv = (
        tracker.action_history[-1].post_levels_completed
        if tracker.action_history
        and tracker.action_history[-1].post_levels_completed is not None
        else 0
    )
    tracker.record_action(chosen_name, int(getattr(action, "value", 0)), data,
                          pre_levels_completed=pre_lv)
    tracker.set_rationale(rationale)
    return action
