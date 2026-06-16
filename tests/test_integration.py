"""Integration tests — the four new substrate modules wired into Misfit.

These tests prove the wiring contract from the Wave-4 integration task:

  1. choose_action uses MCTS when world_model.coverage() >= the gate,
     and falls back to action_search.select_action below it.
  2. abstain_policy gates is_done correctly (a tracker past the action
     floor + a flat fingerprint plateau + a high-variance world model
     causes is_done to return True even when the state is not WIN).
  3. HungarianTracker correspondences are folded into the observation
     list that drives WorldModel.fit (via episode.observe_hungarian).
  4. The outer refinement loop is invoked from _maybe_refit_world_model
     (refinement_iterations_total > 0 after a refit).
  5. The Tier-1 attestation still passes after the wiring — no new LLM
     imports were introduced.

These tests do NOT import arcengine at the top level. They stub the
Misfit base-class constructor surface and the GameAction/GameState
identifiers needed by Misfit.choose_action / Misfit.is_done. This
keeps the integration suite runnable in the local dev env (same
convention as test_substrate_smoke.py and test_mcts_puct.py).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# arcengine + agents.agent stubs — installed before Misfit is imported.
# ---------------------------------------------------------------------------
#
# Misfit imports `from arcengine import FrameData, GameAction, GameState` and
# `from agents.agent import Agent`. Neither is available in the local dev env
# (they ship with the Kaggle eval container). We install lightweight stubs
# that match the surface Misfit actually uses.


class _FakeGameState:
    """Enum-ish — Misfit checks state via `is` and tuple membership."""
    NOT_PLAYED = "NOT_PLAYED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"
    PLAY = "PLAY"


class _FakeGameAction:
    """Mimics arcengine.GameAction enough for Misfit.choose_action."""
    _registry: dict[str, "_FakeGameAction"] = {}
    _by_id: dict[int, "_FakeGameAction"] = {}

    def __init__(self, name: str, value: int, is_complex_: bool):
        self.name = name
        self.value = value
        self._complex = is_complex_
        self.data: dict | None = None
        self.reasoning: Any = None

    def is_simple(self) -> bool:
        return not self._complex

    def is_complex(self) -> bool:
        return self._complex

    def set_data(self, data: dict) -> None:
        self.data = data

    @classmethod
    def from_id(cls, aid: int) -> "_FakeGameAction":
        if aid in cls._by_id:
            return cls._by_id[aid]
        raise KeyError(aid)

    def __iter__(self):  # pragma: no cover
        return iter([])


def _register_action(name: str, value: int, is_complex_: bool) -> _FakeGameAction:
    if name in _FakeGameAction._registry:
        return _FakeGameAction._registry[name]
    a = _FakeGameAction(name, value, is_complex_)
    _FakeGameAction._registry[name] = a
    _FakeGameAction._by_id[value] = a
    return a


# Pre-register the action surface Misfit references.
_RESET = _register_action("RESET", 0, False)
_A1 = _register_action("ACTION1", 1, False)
_A2 = _register_action("ACTION2", 2, False)
_A3 = _register_action("ACTION3", 3, False)
_A6 = _register_action("ACTION6", 6, True)


class _GameActionClass:
    """Class-like object exposing RESET/ACTION1/ACTION6 attributes + from_id +
    iteration over registered enum members. Behaves like the GameAction enum
    surface Misfit uses.
    """
    RESET = _RESET
    ACTION1 = _A1
    ACTION2 = _A2
    ACTION3 = _A3
    ACTION6 = _A6

    @staticmethod
    def from_id(aid: int) -> _FakeGameAction:
        return _FakeGameAction.from_id(aid)

    def __iter__(self):
        return iter(_FakeGameAction._registry.values())


_GAME_ACTION = _GameActionClass()


class _FakeFrameData:
    def __init__(self, frame=None, state=_FakeGameState.PLAY,
                 levels_completed: int = 0, available_actions=None,
                 guid: str = ""):
        self.frame = frame if frame is not None else np.zeros((3, 3), dtype=np.int32)
        self.state = state
        self.levels_completed = int(levels_completed)
        self.available_actions = available_actions or []
        self.guid = guid


# Install the stubs in sys.modules BEFORE importing misfit_agent.misfit_agent.
_arc_stub = types.ModuleType("arcengine")
_arc_stub.FrameData = _FakeFrameData  # type: ignore[attr-defined]
_arc_stub.GameAction = _GAME_ACTION  # type: ignore[attr-defined]
_arc_stub.GameState = _FakeGameState  # type: ignore[attr-defined]
sys.modules.setdefault("arcengine", _arc_stub)

_agents_pkg = types.ModuleType("agents")
_agents_agent = types.ModuleType("agents.agent")


class _StubAgent:
    """Minimal base — Misfit only touches self.game_id, self.action_counter,
    self.frames, and calls super().__init__ + super().cleanup."""

    MAX_ACTIONS: int = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.game_id = kwargs.get("game_id", "test_game")
        self.action_counter = 0
        self.frames: list[Any] = []
        self.agent_name = kwargs.get("agent_name", "misfit")
        self._cleanup = True

    @property
    def name(self) -> str:
        return f"{self.game_id}.misfit"

    def cleanup(self, scorecard=None) -> None:
        self._cleanup = False


_agents_agent.Agent = _StubAgent  # type: ignore[attr-defined]
sys.modules.setdefault("agents", _agents_pkg)
sys.modules.setdefault("agents.agent", _agents_agent)


# Now safe to import the wired Misfit class.
from misfit_agent.misfit_agent import Misfit, MCTS_GATE_COVERAGE_THRESHOLD  # noqa: E402
from misfit_agent.episode import (  # noqa: E402
    EpisodeTracker, ActionRecord, observe_hungarian,
)
from misfit_agent.perceptor import perceive_grid  # noqa: E402
from misfit_agent.world_model import WorldModel  # noqa: E402
from misfit_agent.config import CONFIG  # noqa: E402
from misfit_agent.fingerprint import FINGERPRINT_DIM  # noqa: E402
from misfit_agent.abstain_policy import AbstainPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _build_misfit(tmp_path: Path) -> Misfit:
    """Construct a Misfit with the stubbed Agent base and a tmp library."""
    Misfit.LIBRARY_PATH = str(tmp_path / "lib.jsonl")
    m = Misfit(game_id="integration_test")
    return m


def _two_object_grid() -> np.ndarray:
    """A 5x5 grid with two distinct colored objects — exercises perceptor
    + Hungarian correspondences with a non-trivial scene."""
    g = np.zeros((5, 5), dtype=np.int32)
    g[1, 1] = 2
    g[1, 2] = 2
    g[3, 3] = 5
    return g


def _shifted_two_object_grid() -> np.ndarray:
    """Same two objects but the color-2 block shifted one row down."""
    g = np.zeros((5, 5), dtype=np.int32)
    g[2, 1] = 2
    g[2, 2] = 2
    g[3, 3] = 5
    return g


# ---------------------------------------------------------------------------
# 1. choose_action: MCTS gated on coverage.
# ---------------------------------------------------------------------------


def test_choose_action_uses_mcts_when_coverage_at_or_above_gate(tmp_path):
    """With coverage stubbed >= gate, choose_action must dispatch via
    Misfit._choose_via_mcts (which calls MCTSPUCT.plan)."""
    m = _build_misfit(tmp_path)
    grid = _two_object_grid()
    frame = _FakeFrameData(
        frame=grid,
        state=_FakeGameState.PLAY,
        levels_completed=0,
        available_actions=[_A1, _A2],
    )

    # Force coverage above the gate. The stub returns a constant so MCTS
    # is guaranteed to run.
    with patch.object(WorldModel, "coverage",
                      return_value=MCTS_GATE_COVERAGE_THRESHOLD + 0.01):
        called = {"mcts": False, "select": False}

        def _spy_choose_via_mcts(self, scene, available):
            called["mcts"] = True
            # Return a benign action so the loop completes.
            _A1.reasoning = "spied"
            return _A1

        from misfit_agent import action_search as _as_mod

        original_select = _as_mod.select_action

        def _spy_select_action(*a, **kw):
            called["select"] = True
            return original_select(*a, **kw)

        with patch.object(Misfit, "_choose_via_mcts", _spy_choose_via_mcts), \
             patch.object(_as_mod, "select_action", _spy_select_action):
            action = m.choose_action(frames=[], latest_frame=frame)

        assert called["mcts"] is True, (
            "coverage >= gate must dispatch via MCTS, not select_action"
        )
        assert called["select"] is False, (
            "select_action must NOT be called when MCTS path fires"
        )
        assert action is _A1


def test_choose_action_falls_back_to_select_when_coverage_below_gate(tmp_path):
    """With coverage stubbed BELOW the gate, choose_action must fall back
    to action_search.select_action and NOT touch MCTS."""
    m = _build_misfit(tmp_path)
    grid = _two_object_grid()
    frame = _FakeFrameData(
        frame=grid,
        state=_FakeGameState.PLAY,
        levels_completed=0,
        available_actions=[_A1, _A2],
    )

    # Force coverage below the gate.
    with patch.object(WorldModel, "coverage",
                      return_value=MCTS_GATE_COVERAGE_THRESHOLD - 0.05):
        called = {"mcts": False, "select": False}

        def _spy_choose_via_mcts(self, scene, available):
            called["mcts"] = True
            return _A1

        from misfit_agent import misfit_agent as _ma_mod

        original_select = _ma_mod.select_action

        def _spy_select_action(*a, **kw):
            called["select"] = True
            return original_select(*a, **kw)

        with patch.object(Misfit, "_choose_via_mcts", _spy_choose_via_mcts), \
             patch.object(_ma_mod, "select_action", _spy_select_action):
            m.choose_action(frames=[], latest_frame=frame)

        assert called["select"] is True, (
            "coverage < gate must dispatch via select_action"
        )
        assert called["mcts"] is False, (
            "MCTS must NOT fire when coverage is below the gate"
        )


# ---------------------------------------------------------------------------
# 2. abstain_policy gates is_done.
# ---------------------------------------------------------------------------


def test_is_done_returns_true_when_abstain_policy_fires(tmp_path):
    """When the WIN flag is not set but AbstainPolicy.should_abstain returns
    True (action-counter past floor + plateau + high WM variance), is_done
    must return True. Mirrors test_abstain_policy.test_abstains_when_all_three."""
    m = _build_misfit(tmp_path)

    # Seed the tracker past the abstain floor.
    grid = np.zeros((3, 3), dtype=np.int32)
    scene = perceive_grid(grid)
    m.tracker.scenes.append(scene)
    n = CONFIG.abstain.min_actions_before_abstain + 5
    for _ in range(n):
        m.tracker.scenes.append(scene)
        rec = ActionRecord(
            action_name="ACTION1", action_value=1, data={},
            pre_levels_completed=0, post_levels_completed=0,
            cells_changed=3,  # observed change
            triggered_win=False,
        )
        m.tracker.action_history.append(rec)

    # Plateau: identical flat fingerprints pushed K+1 times.
    flat = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    m.abstain_policy = AbstainPolicy(plateau_window_k=3, plateau_delta_threshold=0.01)
    for _ in range(5):
        m.abstain_policy.push_fingerprint(flat.copy())

    # High WM variance: predict() always returns "no change at confidence 1.0"
    # while every record has cells_changed > 0 → 100% disagreement.
    class _ConfidentNoChangeWM:
        def predict(self, g, name):
            return g.copy(), 1.0

        def coverage(self):  # pragma: no cover - not called in this path
            return 0.0

    m.world_model = _ConfidentNoChangeWM()  # type: ignore[assignment]

    not_win_frame = _FakeFrameData(state=_FakeGameState.PLAY, levels_completed=0)
    assert m.is_done(frames=[], latest_frame=not_win_frame) is True, (
        "abstain trigger must surface as is_done() == True"
    )


def test_is_done_does_not_abstain_below_floor(tmp_path):
    """Sanity check the inverse: a fresh tracker with <floor actions and no
    plateau must return is_done == False on a non-WIN frame."""
    m = _build_misfit(tmp_path)
    grid = np.zeros((3, 3), dtype=np.int32)
    scene = perceive_grid(grid)
    m.tracker.scenes.append(scene)
    # Only 2 actions — well below the abstain floor (default 25).
    for _ in range(2):
        m.tracker.scenes.append(scene)
        m.tracker.action_history.append(ActionRecord(
            action_name="ACTION1", action_value=1, data={},
            pre_levels_completed=0, post_levels_completed=0,
        ))
    not_win_frame = _FakeFrameData(state=_FakeGameState.PLAY, levels_completed=0)
    assert m.is_done(frames=[], latest_frame=not_win_frame) is False


# ---------------------------------------------------------------------------
# 3. HungarianTracker correspondences flow through to WorldModel.fit.
# ---------------------------------------------------------------------------


def test_hungarian_correspondences_reach_world_model_fit(tmp_path):
    """The (s, a, s') observation list handed to WorldModel.fit must carry
    the HungarianTracker correspondence dict under the 'correspondences' key
    (mapping + spawned + destroyed + matched_pairs).

    We exercise this by driving Misfit._maybe_refit_world_model directly with
    a two-scene tracker, then capturing the observation list via a spy on
    WorldModel.fit_with_refinement.
    """
    m = _build_misfit(tmp_path)

    pre_scene = perceive_grid(_two_object_grid())
    post_scene = perceive_grid(_shifted_two_object_grid())

    m.tracker.scenes.append(pre_scene)
    m.tracker.scenes.append(post_scene)
    # Need exactly 5 actions to trip the "% 5 == 0" refit cadence.
    for _ in range(5):
        m.tracker.action_history.append(ActionRecord(
            action_name="ACTION1", action_value=1, data={},
            pre_levels_completed=0, post_levels_completed=0,
            cells_changed=2,
        ))

    captured: dict[str, Any] = {"obs": None}

    def _spy_fit_with_refinement(self, observations, max_iters=4, **kw):
        captured["obs"] = list(observations)
        # Don't actually fit — just record the input shape.
        return {}

    with patch.object(
        WorldModel, "fit_with_refinement", _spy_fit_with_refinement
    ):
        m._maybe_refit_world_model()

    assert captured["obs"] is not None, (
        "fit_with_refinement was not called by _maybe_refit_world_model"
    )
    assert len(captured["obs"]) > 0, "no observations were emitted"
    # Every emitted observation must carry the correspondences dict.
    for obs in captured["obs"]:
        assert "correspondences" in obs, (
            f"observation missing correspondences key: {obs}"
        )
        c = obs["correspondences"]
        assert "mapping" in c
        assert "matched_pairs" in c
        assert "spawned" in c
        assert "destroyed" in c

    # And it must be the same shape as a direct observe_hungarian call.
    direct = observe_hungarian(pre_scene, post_scene)
    assert direct["matched_pairs"] == captured["obs"][0]["correspondences"]["matched_pairs"]


# ---------------------------------------------------------------------------
# 4. Outer refinement loop is invoked.
# ---------------------------------------------------------------------------


def test_outer_refinement_loop_is_invoked_by_maybe_refit(tmp_path):
    """After _maybe_refit_world_model fires at the 5-step cadence,
    world_model.refinement_iterations_total must be > 0 — proving we
    went through fit_with_refinement, not the plain .fit() path."""
    m = _build_misfit(tmp_path)

    pre = perceive_grid(_two_object_grid())
    post = perceive_grid(_shifted_two_object_grid())
    m.tracker.scenes.append(pre)
    m.tracker.scenes.append(post)
    for _ in range(5):
        m.tracker.action_history.append(ActionRecord(
            action_name="ACTION1", action_value=1, data={},
            pre_levels_completed=0, post_levels_completed=0,
            cells_changed=2,
        ))

    assert m.world_model.refinement_iterations_total == 0, (
        "precondition: refinement counter starts at zero"
    )
    m._maybe_refit_world_model()
    assert m.world_model.refinement_iterations_total > 0, (
        "refinement_iterations_total must be > 0 after _maybe_refit_world_model "
        "(proves we went through fit_with_refinement, not plain .fit)"
    )


# ---------------------------------------------------------------------------
# 5. Tier-1 attestation still green after wiring.
# ---------------------------------------------------------------------------


def test_tier1_attestation_still_green_after_integration():
    """Re-run the Tier-1 attestation's forbidden-import scan over the
    integration changes to prove no new LLM dependencies leaked in."""
    # Import the attestation module's helpers and rerun the scan in-process.
    from tests.test_tier1_attestation import (
        _iter_source_files, _scan_file,
        FORBIDDEN_IMPORT_PATTERNS, FORBIDDEN_STRING_PATTERNS,
    )

    hits = []
    for f in _iter_source_files():
        for pat, line, lineno in _scan_file(f, FORBIDDEN_IMPORT_PATTERNS):
            hits.append((pat, str(f), line, lineno))
        for pat, line, lineno in _scan_file(f, FORBIDDEN_STRING_PATTERNS):
            hits.append((pat, str(f), line, lineno))

    assert not hits, (
        "Tier-1 violation after integration:\n"
        + "\n".join(f"  {fp}:{ln}: matched {pat!r}  ->  {line}"
                    for pat, fp, line, ln in hits)
    )
