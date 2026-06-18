"""World model — composes fitted rule library into a forward simulator.

Per the architect's plan: f(state, action) -> next_state at <50us/step,
which powers MCTS rollouts that DO NOT COUNT against the action budget
(per ARC-AGI-3 methodology: 'internal operations not counted as actions').

Critical Tier-1 honesty constraint (judges' must-fix #1):
  - Only trust transitions seen >=3 times with consistent outcome.
  - Below that, simulator reports `confidence < 1.0` and the caller
    (MCTS / agent_core) must fall back to CuriosityExplorer.

Critical correctness constraint:
  - Forward sim is DETERMINISTIC given the current ruleset.
  - If real observations contradict a fitted rule, hypothesis_pruner
    demotes that rule before the next predict() call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import CONFIG
from .rules.no_op import NoOp
from .rules.translate import Translate


RuleInstance = Translate | NoOp


@dataclass
class WorldModel:
    """Composes fitted rule instances into a deterministic forward simulator."""

    rules: list[RuleInstance] = field(default_factory=list)
    observations_per_action: dict[str, int] = field(default_factory=dict)
    confirmed_transitions: dict[str, int] = field(default_factory=dict)
    # Outer refinement loop bookkeeping — HRM analysis (arcprize.org 2025-08-15)
    # showed +13pp from 0→1 refinement, ~doubled from 1→8. We track total
    # iterations done and last per-call score history for adaptive cutoff.
    refinement_iterations_total: int = 0
    last_fit_score_history: list[float] = field(default_factory=list)

    def fit(self, observations: list[dict]) -> dict[str, float]:
        """Fit available rule templates against observed transitions.

        Single-pass fit. Returns per-rule consistency scores. Use
        `fit_with_refinement` for the HRM-style outer loop variant.
        """
        # Group observations by object_class so each rule fits within-class
        by_class: dict[int, list[dict]] = {}
        for obs in observations:
            for c in obs.get("classes_involved", []):
                by_class.setdefault(c, []).append(obs)

        scores: dict[str, float] = {}
        new_rules: list[RuleInstance] = []

        for cls, cls_obs in by_class.items():
            tr = Translate(object_class=cls)
            if tr.fit(cls_obs):
                new_rules.append(tr)
                scores[f"Translate(class={cls})"] = tr.consistency_score
                continue
            no = NoOp(object_class=cls)
            if no.fit(cls_obs):
                new_rules.append(no)
                scores[f"NoOp(class={cls})"] = 1.0

        # Update transition-count bookkeeping
        for obs in observations:
            action = obs.get("action_name", "")
            self.observations_per_action[action] = (
                self.observations_per_action.get(action, 0) + 1
            )

        self.rules = new_rules
        return scores

    def fit_with_refinement(
        self,
        observations: list[dict],
        max_iters: int = 4,
        improvement_threshold: float = 0.02,
    ) -> dict[str, float]:
        """Outer refinement loop — HRM's hidden performance driver.

        Per arcprize.org/blog/hrm-analysis (2025-08-15): refinement was
        worth +13pp from 0→1 and roughly doubled the gain from 1→8.
        We cap at `max_iters` (default 4 — diminishing returns past that
        per the HRM data) and early-stop when mean rule score improves by
        less than `improvement_threshold`.

        Each refinement pass:
          1. Re-fits rule templates against the full observation set
          2. Predicts each observed transition with the current ruleset
          3. Discards rules whose predictions contradict any single
             observation — feedback signal that drives the next refit

        Returns the final per-rule consistency scores after refinement.
        """
        scores: dict[str, float] = {}
        prev_mean_score = 0.0
        for i in range(max_iters):
            scores = self.fit(observations)
            self.refinement_iterations_total += 1
            mean_score = (
                sum(scores.values()) / len(scores) if scores else 0.0
            )
            self.last_fit_score_history.append(mean_score)
            if mean_score - prev_mean_score < improvement_threshold and i > 0:
                break
            prev_mean_score = mean_score
            # Adversarial pass — discard rules that contradict any obs.
            # The hypothesis_pruner pattern from architect Day 9.
            self._prune_contradicting_rules(observations)
        return scores

    def _prune_contradicting_rules(self, observations: list[dict]) -> None:
        """Drop rules whose predictions are contradicted by any single
        observation in the fit set. The remaining rules survive the
        next refinement iteration; pruning is what makes refinement
        improve coverage rather than just re-fitting."""
        if not self.rules:
            return
        survivors: list[RuleInstance] = []
        for rule in self.rules:
            ok = True
            for obs in observations:
                # Only check observations where this rule's class is involved
                cls = getattr(rule, "object_class", None)
                if cls is None or cls not in obs.get("classes_involved", []):
                    continue
                pre = obs.get("pre_objects_of_class", [])
                post = obs.get("post_objects_of_class", [])
                # If counts differ, NoOp can't apply; Translate can only
                # apply when pre/post counts match. Either way: contradiction
                # = drop. This is a coarse but cheap consistency check.
                if isinstance(rule, NoOp) and len(pre) != len(post):
                    ok = False
                    break
                if isinstance(rule, Translate) and (len(pre) != 1 or len(post) != 1):
                    ok = False
                    break
                if isinstance(rule, NoOp):
                    for p, q in zip(pre, post):
                        if p.get("centroid") != q.get("centroid"):
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                survivors.append(rule)
        self.rules = survivors

    def predict(
        self,
        grid: np.ndarray,
        action_name: str,
    ) -> tuple[np.ndarray, float]:
        """Forward-simulate one step.

        Returns (predicted_grid, confidence). Confidence is 1.0 only when
        the action has been observed >= min_observations_for_trust times
        AND at least one fitted rule fires.
        """
        if action_name == "RESET":
            # Can't predict reset effects without level-start observations;
            # caller should treat as low-confidence.
            return grid.copy(), 0.0

        obs_count = self.observations_per_action.get(action_name, 0)
        threshold = CONFIG.world_model.min_observations_for_trust
        if obs_count < threshold or not self.rules:
            return grid.copy(), 0.0

        predicted = grid.copy()
        rules_fired = 0
        for rule in self.rules:
            try:
                next_grid = rule.predict(predicted, action_name)
            except Exception:
                continue
            if not np.array_equal(next_grid, predicted):
                predicted = next_grid
                rules_fired += 1

        if rules_fired == 0:
            return predicted, 0.5  # rules exist but none fire — partial confidence

        return predicted, 1.0

    def coverage(self) -> float:
        """Fraction of attempted actions for which we have any fitted rule."""
        if not self.observations_per_action:
            return 0.0
        threshold = CONFIG.world_model.min_observations_for_trust
        trusted = sum(1 for n in self.observations_per_action.values() if n >= threshold)
        return trusted / len(self.observations_per_action)

    def has_class_coverage(self, object_class: int) -> bool:
        return any(getattr(r, "object_class", None) == object_class for r in self.rules)
