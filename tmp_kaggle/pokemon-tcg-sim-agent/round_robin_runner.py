"""Round-robin runner for the live PTCG variant arena.

Tier-1 strict: deterministic engine, deterministic agents, honest counts.
Each agent uses its own declared deck (no --shared-deck) so the matrix
captures the (agent + deck) variant as a single competitive entity. This
also lets us read "best deck" off the v8/v8_water/v8_psychic triangle —
same brain, different decks — and "best agent" off rows that beat the
field across deck profiles.

Usage:
    python round_robin_runner.py --games 10 --out matrix.json
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from local_arena import evaluate_pair  # honest pair evaluator

# Roster: (variant_id, agent_module, label_deck_csv)
ROSTER = [
    ("agent_v8",         "agent_v8",         "deck.csv (darkness)"),
    ("v14_ensemble",     "agent_v14",        "deck.csv (darkness)"),
    ("agent_v8_water",   "agent_v8_water",   "deck_water.csv"),
    ("deck_psychic_v8",  "agent_v8_psychic", "deck_psychic.csv"),
]


def run(num_games: int, out_path: str) -> dict:
    results = []
    t0 = time.time()
    # All unordered pairs i<j
    for i in range(len(ROSTER)):
        for j in range(i + 1, len(ROSTER)):
            a_id, a_mod, a_deck = ROSTER[i]
            b_id, b_mod, b_deck = ROSTER[j]
            mod_a = importlib.import_module(a_mod)
            mod_b = importlib.import_module(b_mod)
            deck_a = mod_a.read_deck_csv()
            deck_b = mod_b.read_deck_csv()
            print(f"[arena] {a_id} vs {b_id}: {num_games} games "
                  f"(decks: A={a_deck} B={b_deck})", flush=True)
            t_pair = time.time()
            res = evaluate_pair(a_mod, b_mod, deck_a, deck_b, num_games)
            res["a_id"] = a_id
            res["b_id"] = b_id
            res["a_deck"] = a_deck
            res["b_deck"] = b_deck
            results.append(res)
            print(f"  -> a_wins={res['wins_a']} b_wins={res['wins_b']} "
                  f"draws={res['draws']} errors={res['errors']} "
                  f"win_rate_a={res['win_rate_a']} "
                  f"[{round(time.time()-t_pair,1)}s]", flush=True)
    total_seconds = round(time.time() - t0, 1)
    summary = {
        "roster": [
            {"variant": v[0], "agent_module": v[1], "deck": v[2]}
            for v in ROSTER
        ],
        "games_per_pair": num_games,
        "pair_count": len(results),
        "total_games": sum(r["games"] for r in results),
        "wall_clock_seconds": total_seconds,
        "pairs": results,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"[arena] wrote {out_path} ({total_seconds}s)", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--out", default="round_robin_matrix.json")
    args = p.parse_args()
    run(args.games, args.out)
