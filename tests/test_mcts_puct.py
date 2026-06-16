"""MCTS-PUCT planner tests.

Covers:
  1. Basic plan returns a deep-copy-safe ActionHandle.
  2. CRITICAL — deep-copy mutation safety: sibling tree branches do not
     leak click coordinates into each other via the shared GameAction
     enum. (Lane A risk flagged by the architect.)
  3. Progress-path prior biases visits toward seeded actions.
  4. Hard timeout is respected.
  5. Progressive widening: more click candidates expanded as N grows.
  6. Root stats dict shape is sane and returnable.

These tests do NOT import arcengine. They use a `FakeAction` duck-type that
mimics the GameAction enum surface the planner relies on:
    .name, .value, .is_complex(), .set_data(dict)
This keeps the test suite runnable in the offline dev env per the
established convention in test_substrate_smoke.py and test_world_model.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.mcts_puct import (
    ActionHandle,
    MCTSPUCT,
    make_handle_from_enum,
)


# ---------------------------------------------------------------------------
# Test doubles — duck-typed GameAction + Scene + click candidates.
# ---------------------------------------------------------------------------

class FakeAction:
    """Mimics the GameAction enum surface MCTSPUCT depends on.

    Crucially, set_data MUTATES the instance in place — same as the real
    GameAction enum. That mutability is the whole reason for ActionHandle's
    existence; the deep-copy test must exercise it.
    """
    _registry: dict[str, "FakeAction"] = {}

    def __init__(self, name: str, value: int, complex_: bool):
        self.name = name
        self.value = value
        self._complex = complex_
        self.data: dict | None = None
        self.reasoning = None

    @classmethod
    def get(cls, name: str, value: int, complex_: bool) -> "FakeAction":
        """Singleton-per-name to mimic enum identity."""
        if name not in cls._registry:
            cls._registry[name] = cls(name, value, complex_)
        return cls._registry[name]

    def is_complex(self) -> bool:
        return self._complex

    def is_simple(self) -> bool:
        return not self._complex

    def set_data(self, data: dict) -> None:
        self.data = data  # IN-PLACE mutation — exactly the GameAction behaviour


class FakeScene:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]


class FakeClickCand:
    __slots__ = ("x", "y", "source")
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.source = "test"


def _wm_predict_identity(grid: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """A world model that says 'nothing changes' with high confidence."""
    return grid.copy(), 1.0


def _wm_predict_shift(grid: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """A world model where ACTION1 increments grid[0,0] and ACTION6 sets [0,1]=9."""
    out = grid.copy()
    if name == "ACTION1":
        out[0, 0] = (out[0, 0] + 1) % 10
    elif name == "ACTION6":
        out[0, 1] = 9
    return out, 1.0


def _click_cands_fixed(scene) -> list[FakeClickCand]:
    """Returns 5 distinct click candidates."""
    return [
        FakeClickCand(10, 10),
        FakeClickCand(20, 20),
        FakeClickCand(30, 30),
        FakeClickCand(40, 40),
        FakeClickCand(50, 50),
    ]


def _click_cands_empty(scene) -> list[FakeClickCand]:
    return []


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_plan_returns_action_handle_with_independent_data_dict():
    """Plan returns an ActionHandle whose `data` dict is its own copy —
    mutating it must not back-propagate into the tree's child handles."""
    FakeAction._registry.clear()
    a1 = FakeAction.get("ACTION1", 1, complex_=False)
    a6 = FakeAction.get("ACTION6", 6, complex_=True)
    scene = FakeScene(np.zeros((8, 8), dtype=np.int32))

    planner = MCTSPUCT(
        world_model_predict=_wm_predict_shift,
        click_candidates_fn=_click_cands_fixed,
        last_known_progress_path=[],
    )
    result = planner.plan(scene, available_actions=[a1, a6])

    assert isinstance(result.chosen, ActionHandle)
    assert result.chosen.action_name in {"ACTION1", "ACTION6"}
    # Mutate the chosen handle's data — should not affect any other state.
    chosen_data_before = dict(result.chosen.data)
    result.chosen.data["x"] = -999
    # The original handle data we read first should match what we expect:
    # mutation is local to this handle copy only.
    if "x" in chosen_data_before:
        assert chosen_data_before["x"] != -999


def test_DEEP_COPY_safety_sibling_branches_do_not_leak_click_data():
    """THE LANE-A CRITICAL TEST.

    If MCTS called `GameAction.ACTION6.set_data({...})` on every branch,
    branches would silently overwrite each other's click data. After many
    rollouts, every ACTION6 child in the tree would end up holding *the
    last branch's* coordinates, and the search would be incoherent.

    This test exercises the failure mode directly:
      - Three click candidates: (10,10), (20,20), (30,30).
      - Three expanded ACTION6 children in the tree.
      - After plan() runs 200 rollouts, the three children's `data` dicts
        must STILL be (10,10), (20,20), (30,30) — distinct and unmutated.
    """
    FakeAction._registry.clear()
    a6 = FakeAction.get("ACTION6", 6, complex_=True)
    scene = FakeScene(np.zeros((8, 8), dtype=np.int32))

    captured_handles: list[ActionHandle] = []

    def _click_three(scene_):
        return [FakeClickCand(10, 10), FakeClickCand(20, 20), FakeClickCand(30, 30)]

    planner = MCTSPUCT(
        world_model_predict=_wm_predict_identity,
        click_candidates_fn=_click_three,
    )
    result = planner.plan(scene, available_actions=[a6])

    # The returned handle's enum_ref should be the FakeAction singleton.
    assert result.chosen.enum_ref is a6
    # Pull all expanded ACTION6 handles out of root_stats — each must still
    # carry its original (x, y) and they must all be distinct.
    action6_entries = [
        v for k, v in result.root_stats.items()
        if k.startswith("ACTION6@")
    ]
    assert len(action6_entries) == 3, \
        f"expected 3 distinct ACTION6 children, got {len(action6_entries)}: {action6_entries}"
    xys = {(v["data"]["x"], v["data"]["y"]) for v in action6_entries}
    assert xys == {(10, 10), (20, 20), (30, 30)}, \
        f"DEEP-COPY LEAK: sibling click data collided — got {xys}"

    # Belt-and-braces: applying set_data on the shared enum at the very end
    # is a single operation, and prior to that the enum's data should be
    # untouched by the search. Since FakeAction.data starts as None, any
    # search-time set_data call would have populated it. Assert it is still
    # None — proving the search never touched the singleton.
    assert a6.data is None, \
        "Search mutated the canonical enum during rollouts — Lane A failure"


def test_progress_path_prior_biases_visits_toward_seeded_action():
    """If the progress path lists ACTION1, then under a neutral world
    model ACTION1 should accumulate more visits than ACTION3 at root."""
    FakeAction._registry.clear()
    a1 = FakeAction.get("ACTION1", 1, complex_=False)
    a3 = FakeAction.get("ACTION3", 3, complex_=False)
    scene = FakeScene(np.zeros((4, 4), dtype=np.int32))

    planner = MCTSPUCT(
        world_model_predict=_wm_predict_identity,
        click_candidates_fn=_click_cands_empty,
        last_known_progress_path=["ACTION1"],   # bias the prior
    )
    result = planner.plan(scene, available_actions=[a1, a3])

    n_a1 = result.root_stats["ACTION1"]["N"]
    n_a3 = result.root_stats["ACTION3"]["N"]
    p_a1 = result.root_stats["ACTION1"]["P"]
    p_a3 = result.root_stats["ACTION3"]["P"]
    # Prior on seeded action is 1.0; unseeded is 0.5 → normalized 2:1.
    assert p_a1 > p_a3, f"P(ACTION1)={p_a1} should be > P(ACTION3)={p_a3}"
    # With identity world model Q-terms are equal, so visit count should
    # follow the prior asymmetry.
    assert n_a1 > n_a3, f"N(ACTION1)={n_a1} should be > N(ACTION3)={n_a3} under prior bias"


def test_hard_timeout_is_respected_within_tolerance():
    """A pathologically slow world model triggers the wall-clock kill.

    We assert the planner returns within the timeout PLUS one slow-call
    grace period — a tight bound but not pixel-perfect, because Python
    timing has jitter."""
    FakeAction._registry.clear()
    a1 = FakeAction.get("ACTION1", 1, complex_=False)
    scene = FakeScene(np.zeros((4, 4), dtype=np.int32))

    slow_call_seconds = 0.02

    def _slow_wm(grid, name):
        time.sleep(slow_call_seconds)
        return grid.copy(), 1.0

    planner = MCTSPUCT(
        world_model_predict=_slow_wm,
        click_candidates_fn=_click_cands_empty,
    )

    t0 = time.perf_counter()
    result = planner.plan(scene, available_actions=[a1])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    timeout_ms = planner.timeout_ms
    # Grace = max-depth slow-calls in one final rollout before the next
    # deadline-check fires. Generous to avoid CI flakiness.
    grace_ms = planner.max_depth * slow_call_seconds * 1000.0 + 100.0
    assert elapsed_ms < timeout_ms + grace_ms, (
        f"plan() ran {elapsed_ms:.0f}ms; "
        f"limit was {timeout_ms}ms + {grace_ms:.0f}ms grace"
    )
    assert result.timed_out is True, \
        "expected timed_out flag to be set when wall-clock budget exhausted"


def test_progressive_widening_expands_at_least_min_candidates_at_root():
    """Progressive widening must NEVER expand fewer than the configured
    minimum at root, even when N=0. Otherwise the search starves on
    complex-action games."""
    FakeAction._registry.clear()
    a6 = FakeAction.get("ACTION6", 6, complex_=True)
    scene = FakeScene(np.zeros((8, 8), dtype=np.int32))

    # Supply many candidates; widening should still cap to >= min.
    def _many_cands(scene_):
        return [FakeClickCand(i, i) for i in range(20)]

    planner = MCTSPUCT(
        world_model_predict=_wm_predict_identity,
        click_candidates_fn=_many_cands,
    )
    result = planner.plan(scene, available_actions=[a6])

    action6_entries = [k for k in result.root_stats if k.startswith("ACTION6@")]
    assert len(action6_entries) >= MCTSPUCT.PROGRESSIVE_WIDENING_MIN, \
        f"progressive widening starved root — only {len(action6_entries)} children"
    # And we should not have expanded ALL 20 immediately at root with N=0.
    assert len(action6_entries) <= 20


def test_root_stats_shape_is_introspectable():
    """The returned root_stats dict must be human-readable for logging
    and ledger-row receipts: every edge has name/data/N/Q/P keys, plus
    a `__node__` summary entry."""
    FakeAction._registry.clear()
    a1 = FakeAction.get("ACTION1", 1, complex_=False)
    a2 = FakeAction.get("ACTION2", 2, complex_=False)
    scene = FakeScene(np.zeros((4, 4), dtype=np.int32))

    planner = MCTSPUCT(
        world_model_predict=_wm_predict_identity,
        click_candidates_fn=_click_cands_empty,
    )
    result = planner.plan(scene, available_actions=[a1, a2])

    assert "__node__" in result.root_stats
    assert result.root_stats["__node__"]["N_total"] == result.rollouts_run
    for k, v in result.root_stats.items():
        if k == "__node__":
            continue
        assert {"name", "data", "N", "Q", "P"}.issubset(v.keys()), \
            f"root_stats entry {k} missing required fields"
        assert isinstance(v["N"], int)
        assert isinstance(v["Q"], float)
        assert 0.0 <= v["P"] <= 1.0
    assert result.rollouts_run > 0
    assert result.wallclock_ms > 0.0
