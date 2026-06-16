"""ARC-AGI-3 interactive DSL adapter — wave 2.

ARC-AGI-3 is a stream of (state, action, next_state) transitions rather than
the ARC-AGI-2 (input, output) pairs. This adapter buffers observed transitions
per game, groups them by action_id, and uses each per-action group as a
synthetic train set for the public `synthesize()` beam search.

The synthesised Program for a given (game_id, action_id) becomes a forward
world-model: it maps an arbitrary current_state to the predicted next_state if
that action were applied. The adapter uses these programs to:

  - predict_next_state(state, action) -> (predicted_grid, confidence)
  - best_action(state, available_actions) -> action_id

Tier-1 honest:
  - Pure search over the public hand-authored DSL grammar
  - No third-party ML runtimes or pretrained model fetchers of any kind
  - No learned parameters; cache + grammar are the entire state
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ast import Program
from .interpreter import evaluate
from .synthesis import synthesize


# Action id constants matching the ARC-AGI-3 GameAction enum the upstream
# substrate uses (RESET=0, ACTION1..ACTION7 = 1..7). Kept here as plain ints
# so the adapter doesn't pull the agents.actions module into Tier-1 scope.
RESET_ACTION = 0
ACTION_IDS = (0, 1, 2, 3, 4, 5, 6, 7)


@dataclass
class _Observation:
    """One observed (prev_state, action, next_state) transition."""

    prev_state: np.ndarray
    action: int
    next_state: np.ndarray


@dataclass
class Arc3DslAdapter:
    """Buffered per-game DSL world-model for ARC-AGI-3.

    Attributes:
        time_budget_per_action_s: Soft cap on per-action synthesise() and
            predict_next_state() work. Passed straight to synthesize().
        min_distinct_actions: Minimum number of distinct action ids that must
            appear in the buffer before any synthesis is attempted.
        min_pairs_per_action: Minimum train-pair count per action group before
            that action's program is synthesised.
        rng_seed: Seed for the fallback random action chooser (deterministic
            tests).
    """

    time_budget_per_action_s: float = 0.5
    min_distinct_actions: int = 3
    min_pairs_per_action: int = 1
    rng_seed: int = 0

    # observations buffered in arrival order; we keep them as a flat list and
    # group by action id at synthesis time
    _buffer: list[_Observation] = field(default_factory=list)
    # cache: action_id -> best Program
    _programs: dict[int, Program] = field(default_factory=dict)
    # confidence for each cached program (train-set cell accuracy on the
    # pseudo-pairs that produced it)
    _confidence: dict[int, float] = field(default_factory=dict)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    # ------------------------------------------------------------------ observe

    def observe(
        self,
        prev_state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> None:
        """Add one (prev_state, action, next_state) transition to the buffer.

        Triggers (re)synthesis when the buffer covers
        `min_distinct_actions` distinct action ids.
        """
        prev = np.asarray(prev_state)
        nxt = np.asarray(next_state)
        self._buffer.append(_Observation(prev, int(action), nxt))

        distinct_actions = {obs.action for obs in self._buffer}
        if len(distinct_actions) >= self.min_distinct_actions:
            self._resynthesize()

    # ------------------------------------------------------------------ synth

    def _pairs_by_action(self) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
        groups: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for obs in self._buffer:
            groups.setdefault(obs.action, []).append((obs.prev_state, obs.next_state))
        return groups

    def _resynthesize(self) -> None:
        """Re-run synthesise() for every action group with enough samples.

        Each call is bounded by self.time_budget_per_action_s; total wall-clock
        is bounded by len(groups) * time_budget_per_action_s.
        """
        groups = self._pairs_by_action()
        for action_id, pairs in groups.items():
            if len(pairs) < self.min_pairs_per_action:
                continue
            programs = synthesize(
                pairs,
                time_budget_s=self.time_budget_per_action_s,
            )
            if not programs:
                continue
            best = programs[0]
            self._programs[action_id] = best
            self._confidence[action_id] = self._train_accuracy(best, pairs)

    @staticmethod
    def _train_accuracy(
        program: Program,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Cell-accuracy on the pseudo-pairs that produced this program.

        Used as the per-program confidence score.
        """
        total_cells = 0
        matched_cells = 0
        for x, y in pairs:
            try:
                pred = evaluate(program, x)
            except Exception:
                pred = None
            if isinstance(pred, np.ndarray) and pred.shape == y.shape:
                matched_cells += int(np.sum(pred == y))
                total_cells += int(y.size)
            else:
                total_cells += int(y.size)
        if total_cells == 0:
            return 0.0
        return matched_cells / total_cells

    # ------------------------------------------------------------------ predict

    def predict_next_state(
        self,
        current_state: np.ndarray,
        action: int,
    ) -> tuple[np.ndarray, float]:
        """Predict next_state if `action` is applied to `current_state`.

        Returns:
            (predicted_grid, confidence). When no program is cached for this
            action, falls back to identity (the grid is unchanged) with
            confidence 0.0.
        """
        deadline = time.monotonic() + self.time_budget_per_action_s
        cur = np.asarray(current_state)
        program = self._programs.get(int(action))
        if program is None:
            return cur.copy(), 0.0
        try:
            pred = evaluate(program, cur)
        except Exception:
            return cur.copy(), 0.0
        # enforce the time budget on the evaluate() call itself; if interpreter
        # blew past it, downgrade confidence (still return the prediction)
        confidence = float(self._confidence.get(int(action), 0.0))
        if time.monotonic() > deadline:
            confidence *= 0.5
        if not isinstance(pred, np.ndarray):
            return cur.copy(), 0.0
        return pred, confidence

    # ------------------------------------------------------------------ choose

    def best_action(
        self,
        current_state: np.ndarray,
        available_actions: list[int],
    ) -> int:
        """Pick the action with the highest (novelty * confidence) score.

        Novelty here is a cheap, principled proxy: fraction of cells that the
        predicted grid changes relative to `current_state`. Action types that
        we have no cached program for are scored 0; if every available action
        is uncached we fall back to a deterministic random pick from the
        provided list.
        """
        if not available_actions:
            raise ValueError("available_actions must not be empty")

        cur = np.asarray(current_state)
        best_score = -1.0
        best_a: Optional[int] = None
        any_cached = False
        for a in available_actions:
            if int(a) not in self._programs:
                continue
            any_cached = True
            pred, conf = self.predict_next_state(cur, a)
            if pred.shape != cur.shape:
                novelty = 1.0
            else:
                changed = float(np.sum(pred != cur))
                total = float(cur.size) if cur.size else 1.0
                novelty = changed / total
            score = conf * novelty
            if score > best_score:
                best_score = score
                best_a = int(a)

        if not any_cached or best_a is None or best_score <= 0.0:
            return int(self._rng.choice(list(available_actions)))
        return best_a

    # ------------------------------------------------------------------ introspect

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def cached_action_ids(self) -> list[int]:
        return sorted(self._programs.keys())


__all__ = [
    "Arc3DslAdapter",
    "RESET_ACTION",
    "ACTION_IDS",
]
