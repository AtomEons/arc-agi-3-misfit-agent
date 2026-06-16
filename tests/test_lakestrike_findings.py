"""LakeStrike v4 review — regression tests for the most important findings.

Each test reproduces a Blocker or Must-Fix finding from
`docs/LAKESTRIKE_REVIEW_V4.md`. Tests are marked `xfail(strict=True)` so:

  - the suite stays GREEN as long as the bug exists (xfail = expected failure)
  - the moment a bug is FIXED, the test will XPASS, and strict=True turns that
    into a hard failure prompting the implementer to remove the xfail marker
    (or, ideally, write a non-xfail positive assertion).

That way the markers cannot rot silently after a fix lands.

Cover:
  1. G1 (Blocker) — `select_action` mutates the GameAction enum singleton.
  2. G2 (Blocker) — `fit_with_refinement` does NOT carry pruned rules into
     the next iteration (refinement is theater under current implementation).
  3. G3 (Blocker) — `fingerprint_episode` returns vectors that are NOT
     L2-normalized, contradicting the AbstainPolicy plateau-threshold
     calibration claim.

Each test depends only on the local dev environment (stubs arcengine where
needed, same convention as `test_mcts_puct.py` and `test_integration.py`).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# arcengine stub — installed before importing action_search.
# Same surface as test_integration.py uses, scoped local so we don't depend
# on import order between sibling test files.
# ---------------------------------------------------------------------------


class _FakeGameAction:
    _reg: dict[str, "_FakeGameAction"] = {}

    def __init__(self, name: str, value: int, is_complex_: bool):
        self.name = name
        self.value = value
        self._cx = is_complex_
        self.data: dict | None = None
        self.reasoning = None

    def is_complex(self) -> bool:
        return self._cx

    def is_simple(self) -> bool:
        return not self._cx

    def set_data(self, d: dict) -> None:
        # IN-PLACE mutation — exactly the real GameAction enum behavior.
        self.data = d

    @classmethod
    def get(cls, name: str, value: int, is_complex_: bool) -> "_FakeGameAction":
        if name not in cls._reg:
            cls._reg[name] = cls(name, value, is_complex_)
        return cls._reg[name]


_RESET = _FakeGameAction.get("RESET", 0, False)
_A6 = _FakeGameAction.get("ACTION6", 6, True)


class _GameActionClass:
    """Class-like surface that action_search imports as `from arcengine import GameAction`."""
    RESET = _RESET
    ACTION6 = _A6


_arc_stub = types.ModuleType("arcengine")
_arc_stub.GameAction = _GameActionClass  # type: ignore[attr-defined]
sys.modules.setdefault("arcengine", _arc_stub)


# Safe to import now that arcengine is stubbed.
from misfit_agent.action_search import select_action  # noqa: E402
from misfit_agent.episode import EpisodeTracker  # noqa: E402
from misfit_agent.fingerprint import fingerprint_episode, FINGERPRINT_DIM  # noqa: E402
from misfit_agent.perceptor import perceive_grid  # noqa: E402
from misfit_agent.world_model import WorldModel  # noqa: E402


# ===========================================================================
# G1 — `select_action` mutates the GameAction enum singleton.
# ===========================================================================
#
# Finding (docs/LAKESTRIKE_REVIEW_V4.md §G1):
#   `select_action` is the priors-fallback path that runs whenever world-model
#   coverage is below the MCTS gate (most of the early game). On line 155 of
#   `action_search.py`, the complex action's `set_data(...)` is called on the
#   GameAction enum singleton directly. This is the exact Lane-A failure mode
#   that the MCTS module documented as having structurally fixed via
#   ActionHandle (see `mcts_puct.py:45-61`).
#
# The fix should mirror the MCTS contract — wrap the GameAction in a handle,
# or copy.copy(action) before calling set_data — so that two consecutive
# select_action calls do NOT return objects that alias each other's `.data`.


@pytest.mark.xfail(
    reason=(
        "LakeStrike G1 (Blocker): select_action mutates the GameAction enum "
        "singleton — two consecutive calls return the same Python object and "
        "the second call's set_data overwrites the first call's data. Fix: "
        "wrap action in a handle or copy.copy(action) before set_data. "
        "Remove this xfail marker after fixing."
    ),
    strict=True,
)
def test_g1_select_action_does_not_alias_enum_singleton_across_calls():
    """Two consecutive select_action calls on the SAME ACTION6 enum singleton
    must NOT both return the same object — otherwise the caller cannot rely
    on `returned_action.data` being stable across the next call.

    Reproducer also documented in §G1 of the LakeStrike review. The current
    implementation returns the GameAction singleton itself, so `a1 is a2`
    holds and any change to A6.data is visible through a1.
    """
    tracker = EpisodeTracker(game_id="g1")
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2, 3] = 2
    grid[5, 6] = 5
    scene = perceive_grid(grid)
    tracker.scenes.append(scene)

    a_first = select_action(
        scene=scene, tracker=tracker, available_actions=[_A6],
        policy_seeds=[], action_budget_remaining=100, world_model=None,
    )
    data_first = dict(a_first.data) if a_first.data else {}

    # Simulate engine processing the step and a downstream cache happening:
    # the caller wants to inspect a_first.data AFTER another action runs.
    # If select_action returned a fresh handle (not the singleton), the
    # second call cannot affect what a_first sees.
    a_second = select_action(
        scene=scene, tracker=tracker, available_actions=[_A6],
        policy_seeds=[], action_budget_remaining=100, world_model=None,
    )

    # The handles MUST be distinct Python objects — that is the structural
    # safety guarantee `ActionHandle` enforces inside MCTS, and the same
    # guarantee should apply to the priors-fallback path.
    assert a_first is not a_second, (
        "LakeStrike G1: select_action returned the same enum singleton twice "
        "— Lane-A mutation hazard. Wrap with a handle or copy.copy first."
    )

    # Belt-and-braces: a_first.data must still match what we read at call 1,
    # even after the second call mutated its return value's data.
    assert dict(a_first.data) == data_first, (
        "LakeStrike G1: a_first.data was overwritten by a_second's "
        "set_data call (singleton aliasing)."
    )


# ===========================================================================
# G2 — Outer refinement loop has no algorithmic basis to gain HRM's +13pp.
# ===========================================================================
#
# Finding (docs/LAKESTRIKE_REVIEW_V4.md §G2):
#   `WorldModel.fit()` does `self.rules = new_rules` unconditionally at the
#   end of every call. `_prune_contradicting_rules` runs at the end of each
#   refinement iteration and drops contradicting rules — but the NEXT
#   iteration's `fit()` rebuilds `self.rules` from scratch, ignoring the
#   prune output. Therefore the refinement loop does no work beyond a single
#   fit and the claimed HRM-style +13pp gain is unsupported by the code.
#
# The fix should make refinement actually carry information across
# iterations — either by feeding survivors back into fit() as seed rules,
# or by excluding the observation classes whose rules were pruned, or by
# any other mechanism that makes iteration N+1 do something DIFFERENT from
# iteration N when the same observations are passed twice.


@pytest.mark.xfail(
    reason=(
        "LakeStrike G2 (Blocker): fit_with_refinement does not carry pruned "
        "rules into the next iteration. fit() replaces self.rules wholesale, "
        "so _prune_contradicting_rules output is informationally inert. "
        "Either rename to fit_then_prune and retract the HRM +13pp claim, "
        "or implement real cross-iteration carry-over. Remove this xfail "
        "marker after fixing."
    ),
    strict=True,
)
def test_g2_refinement_iterations_actually_improve_or_change_rules():
    """A refinement loop worthy of the HRM-analysis +13pp claim must, on
    contradiction-bearing observations, produce a DIFFERENT rule set after
    iteration 2 than after iteration 1. If iter-1 and iter-2 produce
    identical scores AND identical rules, refinement is theater.

    Reproducer summary (also in review §G2):
      fit_with_refinement(obs, max_iters=3) on contradictory observations
      currently yields score history [1.0, 1.0] and identical rules across
      iterations.
    """
    wm = WorldModel()

    # Two A1 observations for class 2 that disagree:
    # - one shows the object stationary (favors NoOp)
    # - one shows the object moving one row down (favors Translate)
    obs = [
        {
            "action_name": "A1", "classes_involved": [2],
            "pre_objects_of_class": [{"centroid": (0.0, 0.0), "area": 1},
                                     {"centroid": (1.0, 1.0), "area": 1}],
            "post_objects_of_class": [{"centroid": (0.0, 0.0), "area": 1},
                                      {"centroid": (1.0, 1.0), "area": 1}],
        },
        {
            "action_name": "A1", "classes_involved": [2],
            "pre_objects_of_class": [{"centroid": (5.0, 5.0), "area": 1}],
            "post_objects_of_class": [{"centroid": (6.0, 5.0), "area": 1}],
        },
    ]

    # Run refinement enough times that, if it were real, iter 2 would
    # see the pruned state from iter 1 and behave differently.
    wm.fit_with_refinement(obs, max_iters=3)
    rules_after_first_call = list(wm.rules)
    history_after_first_call = list(wm.last_fit_score_history)

    # Snapshot, then re-run. Real refinement should be IDEMPOTENT on
    # converged input (same final rules), but the SCORE HISTORY across
    # iterations within one call should reflect that pruning changed
    # what got fit — i.e., the per-iteration scores should not be a
    # flat repeat of the first iter's score.
    wm2 = WorldModel()
    wm2.fit_with_refinement(obs, max_iters=3)
    history_within_one_call = list(wm2.last_fit_score_history)

    # The core G2 assertion: within ONE call of fit_with_refinement with
    # contradiction-bearing observations, the iteration scores must vary —
    # at least one strictly different value across iterations, OR the
    # surviving rule set must change between iter 1 and iter 2.
    assert (
        len(set(history_within_one_call)) > 1
        or rules_after_first_call != list(wm2.rules)
    ), (
        "LakeStrike G2: fit_with_refinement score history within one call is "
        f"flat ({history_within_one_call}) and rule set unchanged "
        "— refinement is theater. Either rename to fit_then_prune (and retract "
        "the HRM +13pp claim) or implement real cross-iteration carry-over."
    )


# ===========================================================================
# G3 — Fingerprint scale assumption is violated.
# ===========================================================================
#
# Finding (docs/LAKESTRIKE_REVIEW_V4.md §G3):
#   `abstain_policy.py:75-76` claims fingerprints are L2-normalized so the
#   0.01 plateau threshold is "one percent of the unit ball". They are not.
#   `fingerprint_episode` performs no normalization. dim 9 alone (log scenes)
#   reaches ~3.9 by step 50 of a constant-scene episode. Plateau detection
#   therefore almost never fires, which silently disables abstain.
#
# Fix is one of:
#   (a) normalize fingerprint before storage
#   (b) drop the docstring lie and re-derive plateau_delta_threshold from
#       observed scale
#   (c) switch plateau check to cosine distance
# The test below pins option (a) because it is the simplest invariant to
# assert and matches the explicit docstring claim. The implementer may
# instead pick (b) or (c) — but they must change either the docstring or
# the threshold, OR pass this test, before this xfail can be removed.


@pytest.mark.xfail(
    reason=(
        "LakeStrike G3 (Blocker): fingerprint_episode is documented elsewhere "
        "as L2-normalized but is not — AbstainPolicy's plateau threshold "
        "(calibrated for unit-ball vectors) silently disables abstain. Fix: "
        "normalize in fingerprint_episode OR re-derive plateau_delta_threshold "
        "OR switch plateau check to cosine distance. Remove this xfail marker "
        "after fixing."
    ),
    strict=True,
)
def test_g3_fingerprint_is_unit_norm_as_abstain_policy_assumes():
    """A 50-step single-scene tracker should produce a fingerprint with L2
    norm ~1.0 (the abstain plateau threshold of 0.01 is calibrated for unit
    vectors). Currently the L2 norm reaches ~4.9 because dim 9 alone scales
    as log(1+N).
    """
    tracker = EpisodeTracker(game_id="g3")
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[1, 1] = 2
    grid[1, 2] = 2
    grid[3, 3] = 5
    scene = perceive_grid(grid)
    for _ in range(50):
        tracker.scenes.append(scene)

    fp = fingerprint_episode(tracker)
    assert fp.shape == (FINGERPRINT_DIM,), (
        f"fingerprint shape changed: {fp.shape} vs ({FINGERPRINT_DIM},)"
    )

    norm = float(np.linalg.norm(fp))
    # Generous tolerance: anywhere in [0.5, 2.0] would be "approximately unit"
    # under any sensible normalization scheme. Currently norm is ~4.94.
    assert 0.5 <= norm <= 2.0, (
        f"LakeStrike G3: fingerprint L2 norm = {norm:.3f}, expected ~1.0 "
        "(abstain plateau threshold of 0.01 is calibrated for unit-norm "
        "vectors; un-normalized fingerprints silently disable plateau detection)."
    )
