"""NoOp rule — predicts no change.

This is the trivial null hypothesis every other rule must beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoOp:
    """The null rule: action has no effect on this object class."""

    object_class: int   # which color/class this rule applies to
    fitted: bool = False

    def fit(self, observations: list[dict]) -> bool:
        """Fit: confirm that ALL observed transitions for this class show no change."""
        for obs in observations:
            pre = obs.get("pre_objects_of_class", [])
            post = obs.get("post_objects_of_class", [])
            if len(pre) != len(post):
                return False
            for p, q in zip(pre, post):
                if p["centroid"] != q["centroid"]:
                    return False
                if p["area"] != q["area"]:
                    return False
        self.fitted = True
        return True

    def predict(self, grid: np.ndarray, action_name: str) -> np.ndarray:
        """Predict: nothing changes."""
        return grid.copy()
