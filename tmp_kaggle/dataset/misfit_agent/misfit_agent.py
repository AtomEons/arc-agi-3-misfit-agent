"""Misfit — Tier-1 Spelke-priors agent for ARC-AGI-3.

NO LLM in the inference path. NO pretrained heuristics. NO pre-seeded library.
Pure Spelke Core Knowledge priors + experience-only resonance + bounded search.

Wave 4 integration (this file): the four substrate modules that landed under
Builder + Misfit teams are wired into the action loop here.

  - GoalInducer         → induces ranked goal hypotheses; top hypothesis
                          is persisted as a tag on the library entry at WIN
  - AbstainPolicy       → gates is_done (three-conjunction abstain trigger)
                          and receives a fingerprint push every step for
                          plateau detection
  - HungarianTracker    → flowed through episode.observe_hungarian into
                          the WorldModel.fit observation list (CONTINUITY
                          prior on object correspondences)
  - MCTSPUCT            → gated planner; activated only once WorldModel
                          coverage >= 0.3 (until then we keep the cheap
                          priors + 1-step lookahead + click quantizer path
                          from action_search.select_action)

Outer refinement loop: `_maybe_refit_world_model` now calls
`WorldModel.fit_with_refinement(..., max_iters=4)` per the HRM analysis
blog (arcprize.org 2025-08-15) — +13pp from 0→1 refinement pass.

See README.md for the honesty framework.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from .abstain_policy import AbstainPolicy
from .action_search import select_action
from .click_quantizer import click_candidates
from .episode import EpisodeTracker, observe_hungarian
from .fingerprint import fingerprint_episode
from .goal_inducer import GoalInducer
from .mcts_puct import MCTSPUCT, ActionHandle
from .perceptor import perceive_frame, grid_diff
from .resonance import ResonanceLibrary, default_library_path
from .tracker_hungarian import HungarianTracker
from .world_model import WorldModel


# Coverage threshold above which we gate up to MCTS planning. Below this
# we keep the cheap priors-only path — MCTS rollouts are wasted compute
# until the world model can predict at least some actions confidently.
# (b) BUDGET HEURISTIC — 0.3 = "at least one in three observed action types
# has crossed the trust threshold". Below that, rollouts mostly hit
# conf=0.0 leaves and degenerate to random walks.
MCTS_GATE_COVERAGE_THRESHOLD = 0.30


class Misfit(Agent):
    """The misfit substrate agent.

    Per-frame loop (see Agent.main upstream):
        1. perceive_frame(latest_frame.frame) → SceneObservation
        2. tracker.observe(latest_frame, scene)
        3. world_model.fit_with_refinement(observed transitions) — HRM outer loop
        4. fingerprint_episode(tracker) → 50-dim signature
        5. abstain_policy.push_fingerprint(signature) — plateau tracking
        6. resonance.find_k_nearest(signature) → prior winning policies
        7. If world_model.coverage() >= 0.3:
              MCTS-PUCT planner → ActionHandle
           Else:
              select_action(priors + 1-step lookahead + click quantizer)
        8. goal_inducer.observe(prev_scene, curr_scene, Δlevels) — every step
    """

    # StochasticGoose-confirmed: server does NOT enforce per-game cap.
    # 8h55m wall-clock self-kill is the real budget owner.
    MAX_ACTIONS = float("inf")  # type: ignore[assignment]
    LIBRARY_PATH: Optional[str] = None  # None → use default_library_path()
    HARD_WALL_CLOCK_SECONDS = 8 * 3600 + 50 * 60   # 8h50m — judges' Kaggle-reality fix

    # Gate above which MCTS-PUCT activates. Exposed as a class attribute so
    # tests can override it deterministically.
    MCTS_COVERAGE_GATE: float = MCTS_GATE_COVERAGE_THRESHOLD

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tracker = EpisodeTracker(game_id=self.game_id)
        path = self.LIBRARY_PATH or str(default_library_path())
        self.library = ResonanceLibrary.load_or_create(path)
        self.world_model = WorldModel()

        # Wave-4 substrate modules — instantiated up front so they accumulate
        # state across the full episode. MCTS is lazy (built on first use) so
        # tests that exercise only the priors-fallback path don't pay for it.
        self.goal_inducer = GoalInducer()
        self.abstain_policy = AbstainPolicy()
        self.tracker_hungarian = HungarianTracker()
        self.mcts: Optional[MCTSPUCT] = None  # built lazily in choose_action

        self._start_time = time.time()
        self._wm_observations: list[dict] = []

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def _has_time_elapsed(self) -> bool:
        """Wall-clock self-kill — judges' Kaggle-reality must-fix."""
        return (time.time() - self._start_time) >= self.HARD_WALL_CLOCK_SECONDS

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop on WIN, on abstain trigger, or when wall-clock budget exhausted."""
        try:
            if latest_frame.state is GameState.WIN:
                return True
            # Abstain check — three-conjunction trigger (min actions + novelty
            # plateau + world-model variance). False-positive abstain throws
            # away score; the policy is deliberately conservative.
            if self.abstain_policy.should_abstain(self.tracker, self.world_model):
                return True
            if self._has_time_elapsed():
                return True
        except Exception:
            return True  # bail out on unexpected state — never block the framework
        return False

    def _ensure_mcts(self) -> MCTSPUCT:
        """Lazy-build the MCTS planner on first use.

        Wired against `self.world_model.predict`, `click_candidates`, an empty
        progress path (resonance seeds are still consumed by select_action;
        MCTS uses its own PUCT prior over expanded children), and an advance-
        rate callable backed by the episode tracker.
        """
        if self.mcts is None:
            def _advance_rate(name: str) -> float:
                bucket = self.tracker.transition_signals.get(name)
                if not bucket or bucket["total"] == 0:
                    return 0.0
                return bucket["level_advances"] / bucket["total"]

            self.mcts = MCTSPUCT(
                world_model_predict=self.world_model.predict,
                click_candidates_fn=click_candidates,
                last_known_progress_path=(),
                historical_advance_rate=_advance_rate,
            )
        return self.mcts

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

            # Refit the world model whenever a fresh transition just landed
            # (uses HRM outer refinement loop, not plain .fit()).
            self._maybe_refit_world_model()

            # Pull resonance seeds — prior winning policies for similar episodes.
            signature = fingerprint_episode(self.tracker)
            seeds = self.library.retrieve_policy_seeds(signature, k=5)

            # Push the per-step fingerprint to AbstainPolicy for plateau tracking.
            # The abstain check itself fires in is_done, not here — choose_action
            # only feeds the signal.
            self.abstain_policy.push_fingerprint(signature)

            # Feed GoalInducer with the latest (s, a, s', Δlevels) pair so the
            # ranked hypothesis list stays current. This is the Spelke-numerosity
            # path — we don't act on the hypothesis directly here (that arrives
            # in a later wave), but we accumulate it for the cleanup tag.
            self._observe_goal_pair(latest_frame, scene)

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

            # ----------------------------------------------------------------
            # COVERAGE-GATED MCTS: only run the planner once the world model
            # can predict at least some actions confidently. Below the gate,
            # rollouts mostly hit conf=0.0 leaves and degenerate to random
            # walks — wasted compute. Above the gate, MCTS is the only path
            # to a quadratic score lift per the (h/N)^2 scoring rule.
            # ----------------------------------------------------------------
            coverage = self.world_model.coverage()
            if available and coverage >= self.MCTS_COVERAGE_GATE:
                action = self._choose_via_mcts(scene, available)
                # The tracker rationale and action record are populated by
                # _choose_via_mcts so we can fall through to the return.
                return action

            # Below coverage gate — fall back to priors + 1-step lookahead
            # + click quantizer (existing select_action path).
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

    def _choose_via_mcts(
        self,
        scene: Any,
        available: list[Any],
    ) -> GameAction:
        """Run MCTS-PUCT and bind the chosen handle to a canonical enum.

        Important: ActionHandle's `data` dict is deep-copied (Lane-A safety).
        We apply `set_data(chosen.data)` to the canonical enum exactly ONCE
        here — the same contract proven in test_action6_mutation_safety.
        """
        mcts = self._ensure_mcts()
        plan = mcts.plan(scene=scene, available_actions=available)
        chosen: ActionHandle = plan.chosen

        # Bind the chosen handle's data onto the canonical enum member.
        # If enum_ref is None (empty-plan degenerate path), fall back to
        # first available action; on truly empty avail, reset.
        enum_action = chosen.enum_ref
        if enum_action is None:
            if available:
                enum_action = available[0]
            else:
                action = GameAction.RESET
                action.reasoning = "mcts empty-plan + no available actions; reset"
                self.tracker.record_action(
                    "RESET", int(GameAction.RESET.value), {},
                    pre_levels_completed=self._pre_lv(),
                )
                self.tracker.set_rationale("mcts empty-plan; reset")
                return action

        # Deep-copy the data dict one more time before set_data — the handle
        # already deep-copies internally, but a second copy at the bind site
        # is the cheap belt-and-braces guarantee for Lane-A.
        bound_data = copy.deepcopy(chosen.data) if chosen.data else {}
        if chosen.is_complex and bound_data:
            try:
                enum_action.set_data(bound_data)
            except Exception:
                # If the enum rejects the data shape for any reason, leave it
                # un-set rather than crash the framework.
                pass

        # Build rationale and tracker bookkeeping.
        rationale = (
            f"mcts pick: name={chosen.action_name} "
            f"rollouts={plan.rollouts_run} ms={plan.wallclock_ms:.1f} "
            f"timed_out={plan.timed_out} coverage={self.world_model.coverage():.2f}"
        )
        if chosen.is_complex and bound_data:
            rationale += f" data={bound_data}"

        self.tracker.record_action(
            chosen.action_name,
            int(getattr(enum_action, "value", chosen.action_id)),
            bound_data,
            pre_levels_completed=self._pre_lv(),
        )
        self.tracker.set_rationale(rationale)

        if hasattr(enum_action, "is_simple") and enum_action.is_simple():
            enum_action.reasoning = rationale
        else:
            enum_action.reasoning = {
                "desired_action": str(getattr(enum_action, "value", "")),
                "rationale": rationale,
            }
        return enum_action

    def _pre_lv(self) -> int:
        """Pre-action level count for tracker.record_action."""
        if (self.tracker.action_history
                and self.tracker.action_history[-1].post_levels_completed is not None):
            return self.tracker.action_history[-1].post_levels_completed
        return 0

    def _observe_goal_pair(self, latest_frame: Any, scene: Any) -> None:
        """Feed GoalInducer with one (pre, post, Δlevels) pair if we have one.

        Requires at least two scenes and a closed prior action record.
        """
        if len(self.tracker.scenes) < 2:
            return
        prev_scene = self.tracker.scenes[-2]
        curr_scene = self.tracker.scenes[-1]
        # Δlevels: use the most recently closed action record's level delta.
        delta_levels = 0
        if self.tracker.action_history:
            rec = self.tracker.action_history[-1]
            if rec.post_levels_completed is not None:
                delta_levels = int(rec.post_levels_completed - rec.pre_levels_completed)
        try:
            self.goal_inducer.observe(prev_scene, curr_scene, delta_levels)
        except Exception:
            # GoalInducer is fail-soft — bad input must not crash the loop.
            pass

    def _maybe_refit_world_model(self) -> None:
        """Build a (s,a,s') observation from the latest two scenes + last action
        and refit rule templates. Called once per step after self.tracker.observe.

        Uses `fit_with_refinement(max_iters=4)` — the HRM outer-loop variant —
        instead of plain `.fit()`. Per the arcprize.org HRM analysis blog
        (2025-08-15), refinement was +13pp from 0→1 and roughly doubled the
        gain from 1→8. We also enrich each per-class observation with the
        HungarianTracker correspondence dict so per-class rules see object
        identity, not just summary counts.
        """
        if len(self.tracker.scenes) < 2 or not self.tracker.action_history:
            return
        last_action = self.tracker.action_history[-1]
        prev_scene = self.tracker.scenes[-2]
        curr_scene = self.tracker.scenes[-1]
        if prev_scene.grid.shape != curr_scene.grid.shape:
            return

        # Compute correspondences once for this transition. Folded into each
        # per-class obs below so the rule fitter has access if it wants it.
        try:
            correspondences = observe_hungarian(prev_scene, curr_scene, self.tracker)
        except Exception:
            correspondences = None

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
            if correspondences is not None:
                # Optional enrichment — does not change the existing
                # Translate/NoOp.fit contract (those keys are ignored if
                # unknown), but lets downstream rule families key on identity.
                obs["correspondences"] = correspondences
            self._wm_observations.append(obs)

        # Refit every 5 steps to bound cost — now via HRM outer loop.
        if len(self.tracker.action_history) % 5 == 0:
            try:
                self.world_model.fit_with_refinement(
                    self._wm_observations, max_iters=4,
                )
            except Exception:
                pass

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        """On win, persist the winning policy to the resonance library.

        Also persists the GoalInducer's top hypothesis as a tag on the library
        entry's game_id so future K-NN retrievals can correlate "what worked"
        with "what we believed the goal was at win time".
        """
        if self.frames and self.frames[-1].state is GameState.WIN:
            # Top goal hypothesis at win — tag for later analysis.
            top_hyp_tag = self._top_hypothesis_tag()

            # Library API accepts a `game_id`; we suffix the tag onto it so
            # we don't break the existing schema (which has no free-text tag
            # column). This is the smallest-surface change that preserves
            # both honesty and forward-compat with v2 schema.
            tagged_game_id = (
                f"{self.game_id}#{top_hyp_tag}" if top_hyp_tag else self.game_id
            )
            self.library.record_solved(
                fingerprint=fingerprint_episode(self.tracker),
                winning_policy=self.tracker.action_history.copy(),
                composite_score=self._compute_composite_score(),
                source="self-solved",
                game_id=tagged_game_id,
            )
            self.library.flush_to_disk()
        super().cleanup(scorecard)

    def _top_hypothesis_tag(self) -> str:
        """Best goal hypothesis at win, encoded as a short tag string.

        Returns empty string if the inducer has no scored hypothesis. The
        tag is `<kind>(<params>)`, e.g. `removed_all_of_class(2)` or
        `agent_reached_class(3,7)`. Underscores in `kind` are preserved.
        """
        try:
            top = self.goal_inducer.hypothesize(top_k=1)
        except Exception:
            return ""
        if not top:
            return ""
        h = top[0]
        params_str = ",".join(str(p) for p in h.params)
        return f"{h.kind}({params_str})"

    def _compute_composite_score(self) -> float:
        """Cheap proxy for the (human_baseline / ai_actions)^2 scoring rule.
        Without per-level human baselines on hand, we use a budget-fraction proxy.
        """
        if not self.action_counter:
            return 1.0
        budget_left = max(0, self.MAX_ACTIONS - self.action_counter)
        return budget_left / self.MAX_ACTIONS
