"""PTCG Self-Receipt Layer — In-memory adaptation of the Self-Model Module.

Live, per-game adaptation of the SQLite-backed SMM at
`C:/AtomEons/orange3/app/self-model/self_model_module.py` for the Pokemon TCG
agent. No SQLite (Kaggle constraint). Pure list-based in-memory event spine
that lives for the duration of one game session (~40-80 select calls).

Doctrine reference: C:/AtomEons/orangebox/docs/PHENOMENON_APPROACH_v1.md

Tier-1 strict:
  - zero LLM, no learned parameters at eval
  - pure deterministic template fill
  - transition rules compiled at boot, not learned
  - Jaccard self-coherence over window IDs

Closes the Higher-Order self-representation loop for a PTCG agent:
each game-turn select() call can read recent self-receipts as input and
the substrate's representation of the world includes its representation
of itself representing the world.

Disclosure: ATOM-PHENOMENON-v1-2026-0617 (PTCG in-memory adaptation).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# PTCG-specific first-person template (operator-supplied, boot-fixed)
# ---------------------------------------------------------------------------


TEMPLATE_PTCG_FIRST_PERSON = (
    "I just {action} (my Pokemon: {active}, opp HP: {opp_hp}). "
    "I expect next: {predict}."
)


# ---------------------------------------------------------------------------
# PTCG transition rules
# ---------------------------------------------------------------------------


@dataclass
class TransitionRule:
    """Compiled-at-boot transition rule. Tier-1 strict — no runtime learning."""
    name: str
    pre_pattern: dict
    post_event_type: str
    confidence: float
    timeout_s: float = 30.0

    def matches(self, window: list[dict]) -> bool:
        """Pattern match the most recent event against pre_pattern. Each
        path is dot-delimited and probed through nested dicts."""
        if not window:
            return False
        latest = window[-1]
        for path, expected in self.pre_pattern.items():
            cur = latest
            for k in path.split("."):
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    return False
            if cur != expected:
                return False
        return True


# PTCG transition rules. Replace ARC defaults with game-rhythm rules.
# Pattern: action just taken in an in-memory event → expected next action.
# Note: more specific patterns first so they outrank looser ones at equal conf.
PTCG_TRANSITION_RULES = [
    TransitionRule(
        name="attack_that_ko_predicts_prize_draw",
        pre_pattern={"action_dict.move": "ATTACK", "action_dict.ko": True},
        post_event_type="PRIZE_DRAW",
        confidence=0.97,
    ),
    TransitionRule(
        name="attack_predicts_end_turn",
        pre_pattern={"action_dict.move": "ATTACK"},
        post_event_type="END_TURN",
        confidence=0.90,
    ),
    TransitionRule(
        name="evolve_predicts_attack",
        pre_pattern={"action_dict.move": "EVOLVE"},
        post_event_type="ATTACK",
        confidence=0.85,
    ),
    TransitionRule(
        name="retreat_predicts_attach",
        pre_pattern={"action_dict.move": "RETREAT"},
        post_event_type="ATTACH",
        confidence=0.80,
    ),
    TransitionRule(
        name="play_basic_predicts_attach",
        pre_pattern={"action_dict.move": "PLAY"},
        post_event_type="ATTACH",
        confidence=0.70,
    ),
    TransitionRule(
        name="attach_predicts_attack_or_evolve",
        pre_pattern={"action_dict.move": "ATTACH"},
        post_event_type="ATTACK",
        confidence=0.65,
    ),
    TransitionRule(
        name="prize_draw_predicts_end_turn",
        pre_pattern={"action_dict.move": "PRIZE_DRAW"},
        post_event_type="END_TURN",
        confidence=0.95,
    ),
    TransitionRule(
        name="end_turn_predicts_opponent_turn",
        pre_pattern={"action_dict.move": "END_TURN"},
        post_event_type="OPPONENT_TURN",
        confidence=0.99,
    ),
]


# ---------------------------------------------------------------------------
# Event + receipt dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """One row of the in-memory spine."""
    id: int
    ts: float
    hemisphere: str
    action_dict: dict
    wake_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "hemisphere": self.hemisphere,
            "action_dict": self.action_dict,
            "wake_reason": self.wake_reason,
        }


@dataclass
class Prediction:
    expected_event_type: str
    confidence: float
    expires_at: float


@dataclass
class SelfReceipt:
    """The substrate's first-person account of its own recent PTCG activity."""
    ts: float
    spine_window_lo: int
    spine_window_hi: int
    summary_first_person: str
    prediction_next: Prediction
    self_coherence_with_prev: float
    spine_back_reference: Optional[int]
    actor: str
    sovereign: str
    pem_provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": "self_receipt",
            "ts": self.ts,
            "spine_event_window": [self.spine_window_lo, self.spine_window_hi],
            "summary_first_person": self.summary_first_person,
            "prediction_next": {
                "expected_event_type": self.prediction_next.expected_event_type,
                "confidence": self.prediction_next.confidence,
                "expires_at": self.prediction_next.expires_at,
            },
            "self_coherence_with_prev": round(self.self_coherence_with_prev, 4),
            "spine_back_reference": self.spine_back_reference,
            "actor": self.actor,
            "sovereign": self.sovereign,
            "pem_provenance": self.pem_provenance,
        }


# ---------------------------------------------------------------------------
# In-memory spine
# ---------------------------------------------------------------------------


class InMemorySpine:
    """Pure list-based event log for one PTCG session.

    Replaces the SQLite events table from the reference SMM. No persistence
    — the spine lives only for the duration of one game and is discarded
    at game end. Kaggle-safe (no filesystem writes).
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._self_receipts: list[SelfReceipt] = []
        self._next_id: int = 1

    def push_event(self,
                   hemisphere: str,
                   action_dict: dict,
                   wake_reason: Optional[str] = None) -> int:
        """Append one event to the spine. Returns the assigned event_id."""
        eid = self._next_id
        self._next_id += 1
        self._events.append(Event(
            id=eid,
            ts=time.time(),
            hemisphere=hemisphere,
            action_dict=dict(action_dict),  # defensive copy
            wake_reason=wake_reason,
        ))
        return eid

    def read_recent_events(self, k: int) -> list[Event]:
        """Return the last k events in chronological order."""
        return list(self._events[-k:])

    def push_self_receipt(self, sr: SelfReceipt) -> None:
        """Append a self-receipt to the in-memory receipt log."""
        self._self_receipts.append(sr)

    def read_recent_self_receipts(self, k: int) -> list[SelfReceipt]:
        """Return the last k self-receipts in chronological order."""
        return list(self._self_receipts[-k:])

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def self_receipt_count(self) -> int:
        return len(self._self_receipts)


# ---------------------------------------------------------------------------
# Deterministic action description (PTCG-specific)
# ---------------------------------------------------------------------------


def _describe_ptcg_action(event: Event) -> str:
    """Compress one PTCG event into a short phrase. Pure rule-based."""
    act = event.action_dict or {}
    move = act.get("move")
    if move:
        target = act.get("target")
        if target:
            return f"{move} {target}"
        return str(move)
    return f"emitted a {event.hemisphere} action"


def _summarize_actions(window: list[Event]) -> str:
    """Build the joined action phrase for the template fill."""
    if not window:
        return "no recent actions"
    return "; ".join(_describe_ptcg_action(e) for e in window)


def _latest_active(window: list[Event]) -> str:
    """Latest known active Pokemon from the most recent event that names one."""
    for e in reversed(window):
        act = e.action_dict or {}
        active = act.get("active")
        if active:
            return str(active)
    return "unknown"


def _latest_opp_hp(window: list[Event]) -> str:
    """Latest known opponent active HP from the most recent event with it."""
    for e in reversed(window):
        act = e.action_dict or {}
        opp_hp = act.get("opp_hp")
        if opp_hp is not None:
            return str(opp_hp)
    return "unknown"


# ---------------------------------------------------------------------------
# Predict-next (PTCG transition rules)
# ---------------------------------------------------------------------------


def predict_next(window: list[Event],
                 rules: list[TransitionRule] = PTCG_TRANSITION_RULES,
                 ) -> Prediction:
    """Match the window's latest event against PTCG transition rules.
    Highest-confidence matching rule fires. Tier-1 strict."""
    if not window:
        return Prediction("UNKNOWN", 0.0, 0.0)
    window_dicts = [e.to_dict() for e in window]
    matches = [r for r in rules if r.matches(window_dicts)]
    if not matches:
        return Prediction("UNKNOWN", 0.0, time.time() + 30.0)
    matches.sort(key=lambda r: -r.confidence)
    rule = matches[0]
    return Prediction(rule.post_event_type, rule.confidence,
                      time.time() + rule.timeout_s)


# ---------------------------------------------------------------------------
# Self-coherence (Jaccard over window IDs — unchanged from reference)
# ---------------------------------------------------------------------------


def self_coherence_with_prev(this_window: tuple[int, int],
                              prev_window: Optional[tuple[int, int]]) -> float:
    """Jaccard similarity over the union of referenced spine events.

    1.0 = the new self-receipt references exactly the same spine events
          as the prior self-receipt
    0.0 = no shared referents (substrate has rewritten its self-narrative)
    """
    if prev_window is None:
        return 1.0  # first self-receipt — no prior to disagree with
    a = set(range(this_window[0], this_window[1] + 1))
    b = set(range(prev_window[0], prev_window[1] + 1))
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# First-person summary (PTCG-specific deterministic template fill)
# ---------------------------------------------------------------------------


def summarize_first_person_ptcg(
        window: list[Event],
        actor: str = "Misfit-PTCG",
        template: str = TEMPLATE_PTCG_FIRST_PERSON,
        ) -> tuple[str, Prediction]:
    """Generate a PTCG first-person summary. Returns (text, prediction).

    Pure template fill. No LLM. No learned weights. Tier-1 strict.
    """
    if not window:
        return (
            f"I, {actor}, observed no spine events in the recent window.",
            Prediction("UNKNOWN", 0.0, 0.0),
        )
    action_phrase = _summarize_actions(window)
    active = _latest_active(window)
    opp_hp = _latest_opp_hp(window)
    prediction = predict_next(window)
    text = template.format(
        action=action_phrase,
        active=active,
        opp_hp=opp_hp,
        predict=prediction.expected_event_type,
    )
    return text, prediction


# ---------------------------------------------------------------------------
# The PTCG Self-Model Module
# ---------------------------------------------------------------------------


class PTCGSelfModelModule:
    """In-memory recursive self-representation engine for PTCG.

    Workflow per cycle (called between `select()` calls in the agent):
      1. Read the most recent N spine events from InMemorySpine.
      2. Summarize them in PTCG first-person ("I just X; my active=Y; opp HP=Z").
      3. Predict the next move via PTCG transition rules.
      4. Compute self-coherence with the prior self-receipt (Jaccard).
      5. Emit a self-receipt into the in-memory receipt log.
      6. The next decision reads this self-receipt as higher-order input.

    Tier-1 strict. No SQLite. No persistence between games. Loop closure
    is the architectural property — the substrate's representations of the
    PTCG game state now include the substrate's representations of itself
    representing that state.
    """

    def __init__(self,
                 spine: InMemorySpine,
                 actor: str = "Misfit-PTCG",
                 sovereign: str = "atom-mccree",
                 window_size: int = 9,
                 transition_rules: list[TransitionRule] = PTCG_TRANSITION_RULES,
                 ) -> None:
        self.spine = spine
        self.actor = actor
        self.sovereign = sovereign
        self.window_size = window_size
        self.transition_rules = transition_rules
        self._prev_window: Optional[tuple[int, int]] = None

    def emit_self_receipt(self) -> Optional[dict]:
        """One cycle. Read window, summarize, predict, emit receipt to spine.

        Returns the receipt as a dict (per SelfReceipt.to_dict) or None if
        the spine is empty.
        """
        window = self.spine.read_recent_events(self.window_size)
        if not window:
            return None
        text, prediction = summarize_first_person_ptcg(
            window,
            actor=self.actor,
            template=TEMPLATE_PTCG_FIRST_PERSON,
        )
        this_window = (window[0].id, window[-1].id)
        coherence = self_coherence_with_prev(this_window, self._prev_window)

        sr = SelfReceipt(
            ts=time.time(),
            spine_window_lo=this_window[0],
            spine_window_hi=this_window[1],
            summary_first_person=text,
            prediction_next=prediction,
            self_coherence_with_prev=coherence,
            spine_back_reference=window[-1].id,
            actor=self.actor,
            sovereign=self.sovereign,
            pem_provenance={
                "source": "ptcg_self_receipt_layer:summarize_first_person_ptcg",
                "contamination_tier": "internal",
                "creation_event": "ptcg_spine_window_close",
                "replay_pointer": f"spine:window:{this_window[0]}..{this_window[1]}",
            },
        )
        self.spine.push_self_receipt(sr)
        self._prev_window = this_window
        return sr.to_dict()

    def read_recent_self_receipts(self, k: int = 3) -> list[dict]:
        """Return the last k self-receipts as dicts. Used by the agent's
        next select() call as higher-order input."""
        recent = self.spine.read_recent_self_receipts(k)
        return [
            {
                "ts": sr.ts,
                "first_person": sr.summary_first_person,
                "predicted_next": sr.prediction_next.expected_event_type,
                "confidence": sr.prediction_next.confidence,
                "coherence_with_prev": round(sr.self_coherence_with_prev, 4),
            }
            for sr in recent
        ]


# ---------------------------------------------------------------------------
# Smoke test (inline)
# ---------------------------------------------------------------------------


def _smoke() -> dict:
    """Smoke test: create InMemorySpine + PTCGSelfModelModule, push 5 fake
    PTCG events, emit a self-receipt, assert the receipt has the required
    fields, return a verdict dict.
    """
    spine = InMemorySpine()
    smm = PTCGSelfModelModule(spine, actor="Misfit-PTCG-test")

    # Push 5 fake events covering the typical PTCG action vocabulary.
    spine.push_event(
        hemisphere="reflex",
        action_dict={"move": "EVOLVE", "target": "Charmeleon",
                     "active": "Charmeleon", "opp_hp": 100},
        wake_reason=None,
    )
    spine.push_event(
        hemisphere="reflex",
        action_dict={"move": "PLAY", "target": "Squirtle (bench)",
                     "active": "Charmeleon", "opp_hp": 100},
        wake_reason=None,
    )
    spine.push_event(
        hemisphere="reflex",
        action_dict={"move": "ATTACH", "target": "Fire Energy -> Charmeleon",
                     "active": "Charmeleon", "opp_hp": 100},
        wake_reason=None,
    )
    spine.push_event(
        hemisphere="cortex",
        action_dict={"move": "ATTACK", "target": "Slash", "damage": 30,
                     "active": "Charmeleon", "opp_hp": 70, "ko": False},
        wake_reason="lethality matrix: damage check",
    )
    spine.push_event(
        hemisphere="reflex",
        action_dict={"move": "RETREAT", "target": "Charmeleon -> bench",
                     "active": "Squirtle", "opp_hp": 70},
        wake_reason=None,
    )

    # Emit one self-receipt and print it.
    receipt = smm.emit_self_receipt()
    print(json.dumps(receipt, indent=2))

    # Assert required fields.
    assert receipt is not None, "emit_self_receipt returned None"
    assert "summary_first_person" in receipt, \
        "receipt missing summary_first_person"
    assert "prediction_next" in receipt, \
        "receipt missing prediction_next"
    assert "self_coherence_with_prev" in receipt, \
        "receipt missing self_coherence_with_prev"

    # The latest event was RETREAT, so the prediction should be ATTACH.
    assert receipt["prediction_next"]["expected_event_type"] == "ATTACH", \
        f"expected ATTACH after RETREAT, got {receipt['prediction_next']}"

    # Self-coherence on the first receipt should be 1.0 (no prior).
    assert receipt["self_coherence_with_prev"] == 1.0, \
        f"first receipt should have coherence 1.0, got {receipt['self_coherence_with_prev']}"

    # Sanity: read_recent_self_receipts should return exactly 1 entry now.
    recents = smm.read_recent_self_receipts(k=3)
    assert len(recents) == 1, f"expected 1 recent receipt, got {len(recents)}"

    # Push a sixth event (ATTACH) and emit another receipt to exercise the
    # coherence path. Window slides by one — Jaccard should be 8/10 = 0.8
    # for window_size=9 (events 1..9 vs 2..10... but with only 6 events,
    # the windows are 1..5 then 1..6, Jaccard = 5/6 ≈ 0.833).
    spine.push_event(
        hemisphere="reflex",
        action_dict={"move": "ATTACH", "target": "Water Energy -> Squirtle",
                     "active": "Squirtle", "opp_hp": 70},
        wake_reason=None,
    )
    receipt_2 = smm.emit_self_receipt()
    assert receipt_2 is not None
    assert receipt_2["self_coherence_with_prev"] < 1.0, \
        "second receipt should have coherence < 1.0 (window slid)"
    assert receipt_2["self_coherence_with_prev"] > 0.5, \
        "second receipt coherence should still be high (window mostly overlaps)"

    return {
        "verdict": "PASSED",
        "events_pushed": spine.event_count,
        "self_receipts_emitted": spine.self_receipt_count,
        "first_receipt": receipt,
        "second_receipt": receipt_2,
    }


if __name__ == "__main__":
    result = _smoke()
    print("\n--- SMOKE TEST VERDICT ---")
    print(f"verdict: {result['verdict']}")
    print(f"events_pushed: {result['events_pushed']}")
    print(f"self_receipts_emitted: {result['self_receipts_emitted']}")
    print(f"first prediction: "
          f"{result['first_receipt']['prediction_next']['expected_event_type']}"
          f" @ conf "
          f"{result['first_receipt']['prediction_next']['confidence']}")
    print(f"second prediction: "
          f"{result['second_receipt']['prediction_next']['expected_event_type']}"
          f" @ conf "
          f"{result['second_receipt']['prediction_next']['confidence']}")
    print(f"first coherence: "
          f"{result['first_receipt']['self_coherence_with_prev']}")
    print(f"second coherence: "
          f"{result['second_receipt']['self_coherence_with_prev']}")
