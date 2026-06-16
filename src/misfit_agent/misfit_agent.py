"""Misfit — Tier-1 Spelke-priors agent for ARC-AGI-3.

NO LLM in the inference path. NO pretrained heuristics. NO pre-seeded library.
Pure Spelke Core Knowledge priors + experience-only resonance + bounded search.

See README.md for the honesty framework.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from .episode import EpisodeTracker
from .perceptor import perceive_frame, grid_diff
from .fingerprint import fingerprint_episode
from .resonance import ResonanceLibrary, default_library_path
from .action_search import select_action
from .world_model import WorldModel


class Misfit(Agent):
    """The misfit substrate agent.

    Per-frame loop (see Agent.main upstream):
        1. perceive_frame(latest_frame.frame) → SceneObservation
        2. tracker.observe(latest_frame, scene)
        3. world_model.fit(observed transitions) — refit rules each step
        4. fingerprint_episode(tracker) → 50-dim signature
        5. resonance.find_k_nearest(signature) → prior winning policies
        6. select_action(scene, tracker, available_actions, seeds, world_model)
    """

    # StochasticGoose-confirmed: server does NOT enforce per-game cap.
    # 8h55m wall-clock self-kill is the real budget owner.
    MAX_ACTIONS = float("inf")  # type: ignore[assignment]
    LIBRARY_PATH: Optional[str] = None  # None → use default_library_path()
    HARD_WALL_CLOCK_SECONDS = 8 * 3600 + 50 * 60   # 8h50m — judges' Kaggle-reality fix

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tracker = EpisodeTracker(game_id=self.game_id)
        path = self.LIBRARY_PATH or str(default_library_path())
        self.library = ResonanceLibrary.load_or_create(path)
        self.world_model = WorldModel()
        self._start_time = time.time()
        self._wm_observations: list[dict] = []

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def _has_time_elapsed(self) -> bool:
        """Wall-clock self-kill — judges' Kaggle-reality must-fix."""
        return (time.time() - self._start_time) >= self.HARD_WALL_CLOCK_SECONDS

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop on WIN or wall-clock budget exhausted."""
        try:
            if latest_frame.state is GameState.WIN:
                return True
            if self._has_time_elapsed():
                return True
        except Exception:
            return True  # bail out on unexpected state — never block the framework
        return False

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        try:
            # Game not started or game-over — reset.
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self.tracker.observe_terminal(latest_frame)
                action = GameAction.RESET
                action.reasoning = "game needs reset"
                return action

            # Perceive the current frame under Spelke priors.
            scene = perceive_frame(latest_frame.frame)
            self.tracker.observe(latest_frame, scene)

            # Refit the world model whenever a fresh transition just landed.
            self._maybe_refit_world_model()

            # Pull resonance seeds — prior winning policies for similar episodes.
            signature = fingerprint_episode(self.tracker)
            seeds = self.library.retrieve_policy_seeds(signature, k=5)

            # Resolve available actions. Per StochasticGoose: the gateway sends
            # raw ints [1,2,...,6,7] rather than GameAction enum members.
            raw_avail = getattr(latest_frame, "available_actions", None) or []
            available = []
            for a in raw_avail:
                aid = a.value if hasattr(a, "value") else int(a)
                try:
                    available.append(GameAction.from_id(aid))
                except Exception:
                    # If from_id is missing, try enum lookup by value.
                    for ga in GameAction:
                        if int(getattr(ga, "value", -1)) == aid:
                            available.append(ga)
                            break

            # Select via priors + world-model lookahead + click quantization.
            action = select_action(
                scene=scene,
                tracker=self.tracker,
                available_actions=available,
                policy_seeds=seeds,
                action_budget_remaining=10_000,  # wall-clock bound, not action bound
                world_model=self.world_model,
            )

            # Attach the rationale the substrate produced (for traceability).
            rationale = self.tracker.last_action_rationale or "priors-only fallback"
            if action.is_simple():
                action.reasoning = rationale
            elif action.is_complex():
                action.reasoning = {
                    "desired_action": str(getattr(action, "value", "")),
                    "rationale": rationale,
                }
            return action
        except Exception as e:
            # Never crash the framework — defensive fallback.
            try:
                action = GameAction.ACTION1
                action.reasoning = f"fallback after exception: {type(e).__name__}: {e}"
                return action
            except Exception:
                return GameAction.RESET

    def _maybe_refit_world_model(self) -> None:
        """Build a (s,a,s') observation from the latest two scenes + last action
        and refit rule templates. Called once per step after self.tracker.observe.
        """
        if len(self.tracker.scenes) < 2 or not self.tracker.action_history:
            return
        last_action = self.tracker.action_history[-1]
        prev_scene = self.tracker.scenes[-2]
        curr_scene = self.tracker.scenes[-1]
        if prev_scene.grid.shape != curr_scene.grid.shape:
            return

        classes_involved = sorted(set(int(o.color) for o in prev_scene.objects)
                                  | set(int(o.color) for o in curr_scene.objects))
        # Group object summaries by color for the rule.fit consumers.
        def _grouped(scene):
            by_color: dict[int, list[dict]] = {}
            for o in scene.objects:
                by_color.setdefault(int(o.color), []).append({
                    "centroid": o.centroid,
                    "area": int(o.area),
                })
            return by_color

        prev_by_color = _grouped(prev_scene)
        curr_by_color = _grouped(curr_scene)

        # Emit one observation per class so the composer can fit per-class rules.
        for cls in classes_involved:
            obs = {
                "action_name": last_action.action_name,
                "classes_involved": [cls],
                "pre_objects_of_class": prev_by_color.get(cls, []),
                "post_objects_of_class": curr_by_color.get(cls, []),
            }
            self._wm_observations.append(obs)

        # Refit every 5 steps to bound cost.
        if len(self.tracker.action_history) % 5 == 0:
            try:
                self.world_model.fit(self._wm_observations)
            except Exception:
                pass

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        """On win, persist the winning policy to the resonance library."""
        if self.frames and self.frames[-1].state is GameState.WIN:
            self.library.record_solved(
                fingerprint=fingerprint_episode(self.tracker),
                winning_policy=self.tracker.action_history.copy(),
                composite_score=self._compute_composite_score(),
                source="self-solved",
            )
            self.library.flush_to_disk()
        super().cleanup(scorecard)

    def _compute_composite_score(self) -> float:
        """Cheap proxy for the (human_baseline / ai_actions)^2 scoring rule.
        Without per-level human baselines on hand, we use a budget-fraction proxy.
        """
        if not self.action_counter:
            return 1.0
        budget_left = max(0, self.MAX_ACTIONS - self.action_counter)
        return budget_left / self.MAX_ACTIONS
