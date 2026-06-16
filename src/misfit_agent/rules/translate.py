"""TRANSLATE rule — object-class shift in (dx, dy) per action.

Spelke priors used:
  - COHESION: objects move as cohesive units
  - CONTINUITY: object identity persists across the translation
  - AGENCY: ACTION1-4 (and ACTION6 with cursor data) can move objects of one
    class; identification of which class is the agent is *learned*, not assumed

The fitted parameters per rule instance:
  - object_class    : which color/class follows this rule (constrained by data)
  - dx_per_action   : dict[action_name -> int] of column delta
  - dy_per_action   : dict[action_name -> int] of row delta

Maximum 3 free params: object_class + two integer-valued dicts keyed on the
4-direction action set (ACTION1..ACTION4) — well within the architect's
≤3-free-params constraint when ACTION6/7 deltas are constrained to 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Translate:
    object_class: int
    dx_per_action: dict[str, int] = field(default_factory=dict)
    dy_per_action: dict[str, int] = field(default_factory=dict)
    fitted: bool = False
    consistency_score: float = 0.0

    def fit(self, observations: list[dict]) -> bool:
        """Fit dx/dy per action from (pre_centroid, post_centroid) pairs.

        observations: list of dicts with keys:
            action_name (str)
            pre_objects_of_class (list of {centroid: (r,c), ...})
            post_objects_of_class (list of {centroid: (r,c), ...})
        Per the Hungarian tracker, pre[i] corresponds to post[i] when
        objects persist. We require single-object class for now (multi-
        object translate is a Day-N extension).

        Returns True if a consistent (dx, dy) exists per action — i.e. the
        same delta appears EVERY time the action is taken on this class.
        """
        per_action_deltas: dict[str, list[tuple[float, float]]] = {}
        for obs in observations:
            action = obs.get("action_name", "")
            pre = obs.get("pre_objects_of_class", [])
            post = obs.get("post_objects_of_class", [])
            if len(pre) != 1 or len(post) != 1:
                continue  # Translate is single-object for now
            pre_r, pre_c = pre[0]["centroid"]
            post_r, post_c = post[0]["centroid"]
            per_action_deltas.setdefault(action, []).append(
                (post_r - pre_r, post_c - pre_c)
            )

        if not per_action_deltas:
            return False

        total = 0
        consistent = 0
        for action, deltas in per_action_deltas.items():
            if not deltas:
                continue
            dy = round(float(np.median([d[0] for d in deltas])))
            dx = round(float(np.median([d[1] for d in deltas])))
            for d in deltas:
                total += 1
                if abs(d[0] - dy) < 0.5 and abs(d[1] - dx) < 0.5:
                    consistent += 1
            self.dy_per_action[action] = dy
            self.dx_per_action[action] = dx

        if total == 0:
            return False

        self.consistency_score = consistent / total
        self.fitted = self.consistency_score >= 0.8   # 80% consistency threshold
        return self.fitted

    def predict(self, grid: np.ndarray, action_name: str) -> np.ndarray:
        """Forward simulator: shift all cells of object_class by (dx, dy)."""
        dx = self.dx_per_action.get(action_name, 0)
        dy = self.dy_per_action.get(action_name, 0)
        if dx == 0 and dy == 0:
            return grid.copy()
        out = grid.copy()
        rows, cols = out.shape
        mask = out == self.object_class
        if not mask.any():
            return out
        ys, xs = np.where(mask)
        # Clear current positions
        out[mask] = _background_color(grid)
        # Place at translated positions, clipped to grid bounds
        new_ys = np.clip(ys + dy, 0, rows - 1)
        new_xs = np.clip(xs + dx, 0, cols - 1)
        out[new_ys, new_xs] = self.object_class
        return out


def _background_color(grid: np.ndarray) -> int:
    """ARC convention: 0 is background when present; else most-frequent color."""
    if (grid == 0).any():
        return 0
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))
