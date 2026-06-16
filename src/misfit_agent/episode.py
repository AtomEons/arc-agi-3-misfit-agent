"""EpisodeTracker — per-episode state log for rule induction.

We track:
  - the sequence of perceived scenes
  - the sequence of actions emitted (with optional x,y data)
  - the sequence of observed transitions (Δgrid, Δlevels)
  - the *current* hypothesis about what advances the win condition

No cross-episode memory lives here — that's the resonance library's job.
This tracker resets per game.

Also exposes `observe_hungarian(prev_scene, curr_scene)` which produces an
(s, a, s') correspondence dict using `HungarianTracker`. The dict is meant
to be folded into the WorldModel.fit observation list so per-class rule
templates see object identity across the transition (a CONTINUITY prior),
not just unordered before/after object summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .perceptor import SceneObservation, grid_diff


@dataclass
class ActionRecord:
    action_name: str         # e.g. "ACTION1", "ACTION6"
    action_value: int        # numeric enum value
    data: dict               # {"x": int, "y": int} for complex actions
    pre_levels_completed: int
    post_levels_completed: Optional[int] = None
    cells_changed: Optional[int] = None
    triggered_win: bool = False


@dataclass
class EpisodeTracker:
    game_id: str
    scenes: list[SceneObservation] = field(default_factory=list)
    action_history: list[ActionRecord] = field(default_factory=list)
    last_action_rationale: str = ""

    # Lightweight induced rule beliefs (Day 2+ will fill these from transitions).
    # All keys are *observed* dynamics, not hand-crafted task families.
    transition_signals: dict[str, Any] = field(default_factory=dict)

    def observe(self, latest_frame: Any, scene: SceneObservation) -> None:
        """Record a perceived frame + close the previous action record."""
        prior_scene = self.scenes[-1] if self.scenes else None
        self.scenes.append(scene)

        # If we have a prior action record awaiting a follow-up, close it.
        if self.action_history and self.action_history[-1].post_levels_completed is None:
            rec = self.action_history[-1]
            rec.post_levels_completed = int(latest_frame.levels_completed)
            if prior_scene is not None and scene.grid.shape == prior_scene.grid.shape:
                changed, _, _ = grid_diff(prior_scene.grid, scene.grid)
                rec.cells_changed = changed
            rec.triggered_win = bool(getattr(latest_frame, "state", None) and
                                     str(latest_frame.state).endswith("WIN"))
            self._update_transition_signals(rec, prior_scene, scene)

    def observe_terminal(self, latest_frame: Any) -> None:
        """Game-over / not-played state — clear last-action linkage."""
        if self.action_history and self.action_history[-1].post_levels_completed is None:
            self.action_history[-1].post_levels_completed = int(latest_frame.levels_completed)

    def record_action(self, action_name: str, action_value: int, data: dict,
                      pre_levels_completed: int) -> None:
        self.action_history.append(ActionRecord(
            action_name=action_name,
            action_value=action_value,
            data=dict(data) if data else {},
            pre_levels_completed=pre_levels_completed,
        ))

    def set_rationale(self, rationale: str) -> None:
        self.last_action_rationale = rationale

    def winning_actions(self) -> list[ActionRecord]:
        return [a for a in self.action_history if a.triggered_win]

    def level_advancing_actions(self) -> list[ActionRecord]:
        out = []
        for a in self.action_history:
            if a.post_levels_completed is not None and a.post_levels_completed > a.pre_levels_completed:
                out.append(a)
        return out

    def _update_transition_signals(self, rec: ActionRecord,
                                    prior: SceneObservation,
                                    after: SceneObservation) -> None:
        """Accumulate *observed* transition signals — never hardcoded priors."""
        if prior is None:
            return
        action_key = rec.action_name
        bucket = self.transition_signals.setdefault(action_key, {
            "total": 0,
            "cells_changed_sum": 0,
            "level_advances": 0,
            "object_count_delta_sum": 0,
        })
        bucket["total"] += 1
        bucket["cells_changed_sum"] += rec.cells_changed or 0
        if rec.post_levels_completed is not None and rec.post_levels_completed > rec.pre_levels_completed:
            bucket["level_advances"] += 1
        bucket["object_count_delta_sum"] += len(after.objects) - len(prior.objects)


def observe_hungarian(prev_scene: SceneObservation,
                       curr_scene: SceneObservation,
                       tracker: Optional[Any] = None) -> dict:
    """Build an (s, a, s') correspondence dict via HungarianTracker.

    Returns a dict with keys:
      - "mapping": {prev_obj_idx -> curr_obj_idx | None} from HungarianTracker.track
      - "spawned": list of curr-scene indices unmatched (born objects)
      - "destroyed": list of prev-scene indices with no successor
      - "matched_pairs": list of (prev_idx, curr_idx) tuples for matched persistents
      - "prev_objects": [{"centroid": (r,c), "area": int, "color": int}, ...]
      - "post_objects": [{"centroid": (r,c), "area": int, "color": int}, ...]

    The `tracker` parameter is accepted for API symmetry with the existing
    tracker.observe() methods, but is NOT used — correspondences are a pure
    function of the two scenes under Spelke priors (CONTINUITY, COHESION,
    SOLIDITY). Callers will typically fold this dict into the WorldModel
    observation list so per-class rules see object identity, not just
    summary counts.
    """
    # Imported here to keep top-level import surface minimal and to avoid
    # any chance of cycle (HungarianTracker imports perceptor types too).
    from .tracker_hungarian import HungarianTracker

    _ = tracker  # accepted for API symmetry; correspondences are pure of scenes
    ht = HungarianTracker()
    mapping = ht.track(prev_scene, curr_scene)
    spawned = ht.spawned_indices(prev_scene, curr_scene, mapping)
    destroyed = ht.destroyed_indices(mapping)
    matched_pairs = [(int(i), int(j)) for i, j in mapping.items() if j is not None]

    def _serialize(obj: Any) -> dict:
        return {
            "centroid": obj.centroid,
            "area": int(obj.area),
            "color": int(obj.color),
        }

    return {
        "mapping": {int(k): (int(v) if v is not None else None)
                    for k, v in mapping.items()},
        "spawned": [int(j) for j in spawned],
        "destroyed": [int(i) for i in destroyed],
        "matched_pairs": matched_pairs,
        "prev_objects": [_serialize(o) for o in prev_scene.objects],
        "post_objects": [_serialize(o) for o in curr_scene.objects],
    }
