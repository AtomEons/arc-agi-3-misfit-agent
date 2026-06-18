"""AtomEons local arena — head-to-head agent evaluation harness.

Runs N games of agentA vs agentB on the real cg engine. Returns
win/loss/draw counts and 95% CI on win rate. This lets us A/B test
agent variants and deck variants at AI-scale BEFORE burning Kaggle
submission slots.

Usage:
    python local_arena.py agentA agentB deckA deckB num_games

Each agent module must expose:
    agent(obs_dict: dict) -> list[int]
    read_deck_csv() -> list[int]  (for initial selection — passed externally)

The arena bypasses read_deck_csv (we pass decks explicitly) and drives
the game loop manually via cg.game.battle_start / battle_select.

Tier-1 honest: deterministic engine call sequence. The only randomness
is the engine's internal shuffle and any agent-internal randomness
(our v1-v4 agents are fully deterministic).
"""

import importlib
import os
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, Observation


def _normalize_game_result(obs) -> int:
    """Return winner index from a game-over observation.
    -1 if undetermined (e.g., game not over yet).
    0 / 1 = winner player index. 2 = draw.
    Try first via obs.current.result, then via final RESULT log.
    """
    o = to_observation_class(obs)
    if o.current is not None and o.current.result is not None and o.current.result >= 0:
        return o.current.result
    # Scan logs for RESULT
    for log in (o.logs or [])[::-1]:
        if log.type is not None and int(log.type) == 23:  # LogType.RESULT
            return int(log.result) if log.result is not None else -1
    return -1


def play_one_game(agent_a, deck_a: list[int], agent_b, deck_b: list[int],
                  max_turns: int = 500) -> int:
    """Return 0 if agent_a wins, 1 if agent_b wins, 2 for draw, -1 for error."""
    obs, sd = battle_start(deck_a, deck_b)
    if sd.battlePtr is None or sd.battlePtr == 0:
        return -1
    try:
        steps = 0
        while True:
            o = to_observation_class(obs)
            if o.select is None:
                # Game over
                if o.current is not None and o.current.result is not None:
                    return int(o.current.result)
                return -1
            # Determine which player selects
            your_idx = o.select  # placeholder
            your_player = o.current.yourIndex if o.current else 0
            current_agent = agent_a if your_player == 0 else agent_b
            try:
                selection = current_agent(obs)
            except Exception as exc:
                # Treat agent crash as automatic loss
                return 1 - your_player
            if not isinstance(selection, list):
                return 1 - your_player
            # Cap selection to maxCount distinct indices in range
            try:
                obs = battle_select(selection)
            except Exception:
                return 1 - your_player
            steps += 1
            if steps > max_turns:
                # Draw on too-long game
                return 2
    finally:
        battle_finish()


def evaluate_pair(agent_a_path: str, agent_b_path: str,
                  deck_a: list[int], deck_b: list[int],
                  num_games: int = 30) -> dict:
    """Play num_games of agent_a vs agent_b. Returns stats."""
    mod_a = importlib.import_module(agent_a_path)
    mod_b = importlib.import_module(agent_b_path)
    agent_fn_a = mod_a.agent
    agent_fn_b = mod_b.agent

    wins_a = wins_b = draws = errors = 0
    t0 = time.time()
    for g in range(num_games):
        # Alternate first-player to balance positional advantage
        if g % 2 == 0:
            result = play_one_game(agent_fn_a, deck_a, agent_fn_b, deck_b)
            if result == 0:
                wins_a += 1
            elif result == 1:
                wins_b += 1
            elif result == 2:
                draws += 1
            else:
                errors += 1
        else:
            result = play_one_game(agent_fn_b, deck_b, agent_fn_a, deck_a)
            if result == 0:
                wins_b += 1
            elif result == 1:
                wins_a += 1
            elif result == 2:
                draws += 1
            else:
                errors += 1
    elapsed = time.time() - t0

    total = wins_a + wins_b + draws + errors
    win_rate_a = wins_a / max(total, 1)
    # 95% CI Wilson interval (handles small N better than normal approx)
    n = total
    if n > 0:
        p = win_rate_a
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        ci_low, ci_high = max(0, center - margin), min(1, center + margin)
    else:
        ci_low = ci_high = 0
    return {
        "agent_a": agent_a_path,
        "agent_b": agent_b_path,
        "games": num_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "errors": errors,
        "win_rate_a": round(win_rate_a, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "wall_clock_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--shared-deck", default=None,
                   help="Optional: path to a single deck CSV used by BOTH agents. "
                        "When omitted, each agent uses its own read_deck_csv().")
    args = p.parse_args()

    if args.shared_deck:
        with open(args.shared_deck) as f:
            shared = [int(x.strip()) for x in f.read().split() if x.strip()][:60]
        deck_a = deck_b = shared
        deck_src_a = deck_src_b = args.shared_deck
    else:
        # Honor each agent's own deck choice via its read_deck_csv()
        mod_a = importlib.import_module(args.a)
        mod_b = importlib.import_module(args.b)
        deck_a = mod_a.read_deck_csv()
        deck_b = mod_b.read_deck_csv()
        deck_src_a = f"{args.a}.read_deck_csv()"
        deck_src_b = f"{args.b}.read_deck_csv()"

    result = evaluate_pair(args.a, args.b, deck_a, deck_b, args.games)
    result["deck_a_source"] = deck_src_a
    result["deck_b_source"] = deck_src_b
    result["deck_a_size"] = len(deck_a)
    result["deck_b_size"] = len(deck_b)
    import json
    print(json.dumps(result, indent=2))
