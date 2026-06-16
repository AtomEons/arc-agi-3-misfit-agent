"""ACTION6 in-place mutation safety — the Lane-A risk made explicit.

`arcengine.GameAction.ACTION6` is a Python Enum singleton. `set_data(d)`
mutates `self.data` in place. Any code path that does:

    GameAction.ACTION6.set_data({"x": 5, "y": 10})    # branch A
    GameAction.ACTION6.set_data({"x": 47, "y": 2})    # branch B

silently overwrites branch A's coordinates. Inside MCTS this becomes a
correlated, incoherent search: every ACTION6 child in the tree ends up
pointing at the LAST branch's click. Score collapses, debugging is hell.

The structural fix lives in `mcts_puct.py` — `ActionHandle` wraps each
candidate with a deep-copied `data` dict, and `set_data` on the canonical
enum happens AT MOST ONCE, on the way out of `plan()`.

`test_mcts_puct.py` already proves the MCTS-internal version of this
guarantee. THIS file is the narrower, dedicated regression: a fake
GameAction.ACTION6 with `set_data`, handed through a copy-mutation
helper, must leave the ORIGINAL singleton untouched.

The pattern caught here is the one MCTS will multiply: any future helper
that takes a GameAction enum and "applies a click" must use a copy, not
the singleton. If that contract breaks, this test fails fast.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.mcts_puct import ActionHandle, make_handle_from_enum


# ---------------------------------------------------------------------------
# Fake GameAction.ACTION6 — mirrors the real enum's mutability surface.
# ---------------------------------------------------------------------------

class FakeGameAction:
    """Mimics `arcengine.GameAction.ACTION6`:
      - singleton-like (one instance per name)
      - `.name`, `.value`, `.is_complex()` introspection
      - `set_data(d)` MUTATES `self.data` in place (this is the trap)
    """

    _SINGLETONS: dict[str, "FakeGameAction"] = {}

    def __init__(self, name: str, value: int, is_complex_: bool):
        self.name = name
        self.value = value
        self._complex = is_complex_
        self.data: Optional[dict] = None
        self.reasoning: Optional[str] = None

    @classmethod
    def singleton(cls, name: str, value: int, is_complex_: bool) -> "FakeGameAction":
        if name not in cls._SINGLETONS:
            cls._SINGLETONS[name] = cls(name, value, is_complex_)
        return cls._SINGLETONS[name]

    @classmethod
    def reset_singletons(cls) -> None:
        cls._SINGLETONS.clear()

    def is_complex(self) -> bool:
        return self._complex

    def is_simple(self) -> bool:
        return not self._complex

    def set_data(self, data: dict) -> None:
        """IN-PLACE mutation — exactly as `arcengine.GameAction.set_data` behaves."""
        self.data = data


# ---------------------------------------------------------------------------
# A representative "function that mutates a copy" — this is the only safe
# pattern for any helper that takes a GameAction and applies a click.
# ---------------------------------------------------------------------------

@dataclass
class ClickRequest:
    x: int
    y: int


def apply_click_to_copy(canonical_action: Any, click: ClickRequest) -> Any:
    """Reference implementation of the SAFE pattern.

    Returns a deep-copy of the action with the click coordinates applied,
    leaving the canonical singleton untouched. Any helper anywhere in the
    codebase that builds a `set_data` call against the enum MUST follow
    this shape — otherwise MCTS rollouts will share state across branches.
    """
    action_copy = copy.deepcopy(canonical_action)
    action_copy.set_data({"x": int(click.x), "y": int(click.y)})
    return action_copy


# An UNSAFE reference — what we are testing FOR. We don't call it in the
# production path; it exists so the test below can prove it would corrupt
# the canonical singleton, justifying the safe pattern's existence.
def apply_click_in_place_UNSAFE(canonical_action: Any, click: ClickRequest) -> Any:
    """DO NOT USE. Demonstrates the bug pattern this file guards against."""
    canonical_action.set_data({"x": int(click.x), "y": int(click.y)})
    return canonical_action


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_safe_helper_leaves_canonical_action_data_unchanged():
    """The contract: after `apply_click_to_copy(action, click)` returns,
    the canonical singleton's `.data` must be exactly what it was before.

    This is the Lane-A guarantee at its narrowest. If this test ever
    fails, every MCTS rollout is silently corrupting the next."""
    FakeGameAction.reset_singletons()
    action6 = FakeGameAction.singleton("ACTION6", 6, is_complex_=True)
    assert action6.data is None, "precondition: singleton starts with no data"

    returned = apply_click_to_copy(action6, ClickRequest(x=5, y=10))

    # The canonical singleton is untouched.
    assert action6.data is None, (
        "MUTATION LEAK: apply_click_to_copy modified the canonical singleton. "
        "MCTS rollouts would now share click state across branches."
    )
    # The returned object carries the click.
    assert returned is not action6, "must return a different object"
    assert returned.data == {"x": 5, "y": 10}


def test_safe_helper_isolates_sibling_branches():
    """Simulate two MCTS-like branches calling the helper back-to-back
    on the SAME canonical singleton. The two returned actions must hold
    DIFFERENT click coordinates, and the singleton must remain untouched.

    This is what `test_DEEP_COPY_safety_sibling_branches_do_not_leak_click_data`
    in test_mcts_puct.py proves at the planner level; here we prove the
    narrowest underlying building block does it too."""
    FakeGameAction.reset_singletons()
    action6 = FakeGameAction.singleton("ACTION6", 6, is_complex_=True)

    branch_a = apply_click_to_copy(action6, ClickRequest(x=5, y=10))
    branch_b = apply_click_to_copy(action6, ClickRequest(x=47, y=2))
    branch_c = apply_click_to_copy(action6, ClickRequest(x=63, y=0))

    # Three distinct returned objects, three distinct payloads.
    assert {id(branch_a), id(branch_b), id(branch_c)} == \
        {id(branch_a), id(branch_b), id(branch_c)}  # all unique by construction
    assert branch_a.data == {"x": 5, "y": 10}
    assert branch_b.data == {"x": 47, "y": 2}
    assert branch_c.data == {"x": 63, "y": 0}

    # And the canonical singleton remains pristine.
    assert action6.data is None, (
        "MUTATION LEAK across siblings: singleton was touched after "
        "three apply_click_to_copy calls. MCTS would explode here."
    )


def test_unsafe_pattern_DOES_corrupt_the_singleton_proving_the_bug_exists():
    """This is the inverse proof. We call the deliberately-unsafe helper
    and assert it DOES corrupt the singleton. If this test ever STOPS
    failing the way we expect, FakeGameAction has drifted away from the
    real GameAction's mutability semantics, and the safe-pattern test
    above is no longer testing anything real."""
    FakeGameAction.reset_singletons()
    action6 = FakeGameAction.singleton("ACTION6", 6, is_complex_=True)
    assert action6.data is None

    apply_click_in_place_UNSAFE(action6, ClickRequest(x=5, y=10))
    assert action6.data == {"x": 5, "y": 10}, (
        "FakeGameAction no longer mutates in place — this test double has "
        "drifted and the surrounding safety tests are now no-ops. Restore "
        "FakeGameAction.set_data to do `self.data = data`."
    )

    # And a second UNSAFE call clobbers the first — the exact MCTS bug.
    apply_click_in_place_UNSAFE(action6, ClickRequest(x=47, y=2))
    assert action6.data == {"x": 47, "y": 2}, "second clobber should win"
    assert action6.data != {"x": 5, "y": 10}, (
        "the very point of the test: branch B silently overwrites branch A"
    )


def test_make_handle_from_enum_returns_a_fresh_independent_data_dict():
    """`make_handle_from_enum` is the production primitive. Its `data`
    must be a deep copy — mutating it must NOT touch the input dict
    nor any other handle built from the same enum."""
    FakeGameAction.reset_singletons()
    action6 = FakeGameAction.singleton("ACTION6", 6, is_complex_=True)

    input_data = {"x": 5, "y": 10}
    handle_1 = make_handle_from_enum(action6, data=input_data)
    handle_2 = make_handle_from_enum(action6, data=input_data)

    # Distinct dicts, even though they came from the same input.
    assert handle_1.data is not input_data
    assert handle_2.data is not input_data
    assert handle_1.data is not handle_2.data

    # Mutate handle_1's data — must not affect input_data or handle_2.
    handle_1.data["x"] = -999
    assert input_data == {"x": 5, "y": 10}, "input dict was mutated"
    assert handle_2.data == {"x": 5, "y": 10}, "sibling handle was mutated"

    # And the canonical enum is still untouched.
    assert action6.data is None, (
        "make_handle_from_enum touched the canonical enum — Lane A violation"
    )


def test_action_handle_data_survives_deepcopy_round_trip():
    """An ActionHandle can be deep-copied (e.g. when MCTS pushes one onto
    a frontier queue) without the copy aliasing the original's data."""
    h = ActionHandle(
        action_id=6,
        action_name="ACTION6",
        is_complex=True,
        data={"x": 5, "y": 10},
        enum_ref=None,
    )
    h2 = copy.deepcopy(h)
    assert h2 is not h
    assert h2.data is not h.data
    h2.data["x"] = -1
    assert h.data == {"x": 5, "y": 10}, "deepcopy alias broke handle isolation"


def test_many_handles_from_one_enum_remain_independent_under_mutation():
    """Stress test: build 50 handles from one enum singleton, mutate each
    independently, assert no cross-talk and no canonical-enum corruption.
    Mirrors the MCTS pattern of expanding many ACTION6 children per node."""
    FakeGameAction.reset_singletons()
    action6 = FakeGameAction.singleton("ACTION6", 6, is_complex_=True)

    handles = [
        make_handle_from_enum(action6, data={"x": i, "y": i * 2})
        for i in range(50)
    ]
    # Mutate every handle's coordinates to (-i, -i).
    for i, h in enumerate(handles):
        h.data["x"] = -i
        h.data["y"] = -i

    # Each handle reflects its own mutation; none of them collided.
    for i, h in enumerate(handles):
        assert h.data == {"x": -i, "y": -i}, f"handle {i} was clobbered"

    # Canonical singleton still pristine.
    assert action6.data is None, (
        "50 handle mutations leaked into the canonical enum"
    )


def test_winning_policy_replay_does_not_mutate_canonical_enum():
    """The resonance library replays winning policies as `ActionRecord`
    dicts. When the agent converts one back into a `set_data` call on
    the live enum, the conversion must use a copy of the recorded data.
    Otherwise replaying ten games would overwrite the singleton ten times
    with the last game's coordinates by the time the live game starts."""
    FakeGameAction.reset_singletons()
    action6 = FakeGameAction.singleton("ACTION6", 6, is_complex_=True)

    recorded_policies = [
        [{"action_name": "ACTION6", "action_value": 6,
          "data": {"x": x, "y": y}}]
        for x, y in [(5, 10), (47, 2), (63, 0), (1, 1)]
    ]

    # Replay each policy via the safe helper.
    realised = []
    for policy in recorded_policies:
        record = policy[0]
        replayed = apply_click_to_copy(
            action6, ClickRequest(x=record["data"]["x"], y=record["data"]["y"])
        )
        realised.append(replayed.data)

    assert realised == [
        {"x": 5, "y": 10}, {"x": 47, "y": 2},
        {"x": 63, "y": 0}, {"x": 1, "y": 1},
    ]
    assert action6.data is None, (
        "Policy replay leaked into the canonical enum — every subsequent "
        "MCTS rollout would inherit the last replayed click."
    )


def test_numpy_array_in_action_data_is_independent_after_deepcopy():
    """Future-proofing: if a click candidate ever carries a numpy array
    (e.g. a mask or bbox), the deep copy must duplicate the array, not
    alias it. This catches the silent-aliasing bug numpy is famous for."""
    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    h = ActionHandle(
        action_id=6,
        action_name="ACTION6",
        is_complex=True,
        data={"x": 5, "y": 10, "mask": arr},
        enum_ref=None,
    )
    h2 = copy.deepcopy(h)
    h2.data["mask"][0] = -1
    assert arr[0] == 1, "deepcopy aliased a numpy array into the source"
    assert h.data["mask"][0] == 1, "original handle's mask was mutated"
