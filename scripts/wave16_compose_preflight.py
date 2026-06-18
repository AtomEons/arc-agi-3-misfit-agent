"""Wave 16 — depth-2 composition over property-bound rules.

Earlier finding: depth-2 over the OLD-contract grammar (43 rules) yielded
ZERO lift across 1849 program combos per task. The question this script
answers: does depth-2 composition over the PROPERTY-BOUND grammar
(waves 7-15, 81 rules) produce new task solves?

For each task, we enumerate all (rule_a, rule_b) pairs where rule_a.fit()
succeeds, then check whether rule_b composed AFTER rule_a yields a
program that solves the task's test input. The midstate is
`rule_a.predict(train_input)`, and rule_b is fit against those midstates.

Receipt at receipts/100day/wave16_compose_preflight.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.arc2_solver import _rule_factories, train_score  # noqa: E402


def _gold(tid, sols):
    return [np.asarray(g, dtype=np.int32) for g in sols[tid]]


def composed_solves(rule_a_factory, rule_b_factory, train_pairs, test_input, gold):
    """Try (rule_a then rule_b) on train_pairs. Return True if both fit and
    the composition predicts gold correctly on the test input."""
    rule_a = rule_a_factory()
    try:
        if not rule_a.fit(train_pairs):
            return False
    except Exception:
        return False
    # Build midstate train pairs
    mid_pairs = []
    try:
        for inp, out in train_pairs:
            mid = rule_a.predict(np.asarray(inp, dtype=np.int32))
            mid_pairs.append((np.asarray(mid, dtype=np.int32),
                              np.asarray(out, dtype=np.int32)))
    except Exception:
        return False
    rule_b = rule_b_factory()
    try:
        if not rule_b.fit(mid_pairs):
            return False
    except Exception:
        return False
    # Apply on test
    try:
        mid_test = rule_a.predict(np.asarray(test_input, dtype=np.int32))
        pred = rule_b.predict(np.asarray(mid_test, dtype=np.int32))
        pred = np.asarray(pred, dtype=np.int32)
    except Exception:
        return False
    return pred.shape == gold.shape and np.array_equal(pred, gold)


def preflight(chal_path, sol_path, label, max_compose_pairs_per_task=2000):
    chal = json.loads(Path(chal_path).read_text(encoding="utf-8"))
    sols = json.loads(Path(sol_path).read_text(encoding="utf-8"))
    factories = _rule_factories()
    print(f"[{label}] {len(chal)} tasks, {len(factories)} factories -> "
          f"{len(factories)**2} compose pairs/task (cap {max_compose_pairs_per_task})",
          flush=True)

    solved = []
    t0 = time.time()
    for idx, (tid, task) in enumerate(chal.items()):
        train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                        np.asarray(p["output"], dtype=np.int32)) for p in task["train"]]
        gold_outs = _gold(tid, sols)
        test_inputs = [np.asarray(t["input"], dtype=np.int32) for t in task["test"]]

        # Only try the first test input (typical case is 1)
        ti = test_inputs[0]
        gold = gold_outs[0]

        # Filter to rule_a that fit
        fit_a = []
        for fa in factories:
            try:
                ra = fa()
                if ra.fit(train_pairs):
                    fit_a.append(fa)
            except Exception:
                continue
        if not fit_a:
            continue

        # Now try compositions
        solved_this = False
        pairs_tried = 0
        for fa in fit_a:
            if solved_this:
                break
            for fb in factories:
                pairs_tried += 1
                if pairs_tried > max_compose_pairs_per_task:
                    break
                try:
                    if composed_solves(fa, fb, train_pairs, ti, gold):
                        solved.append(tid)
                        solved_this = True
                        break
                except Exception:
                    continue
            if pairs_tried > max_compose_pairs_per_task:
                break

        if (idx + 1) % 20 == 0:
            print(f"  [{label}] {idx+1}/{len(chal)} {time.time()-t0:.1f}s "
                  f"solved={len(solved)}", flush=True)

    print(f"[{label}] DONE {time.time()-t0:.1f}s solved={len(solved)}/{len(chal)} "
          f"= {100.0*len(solved)/len(chal):.2f}%", flush=True)
    return {"label": label, "total": len(chal), "solved": len(solved),
            "solved_ids": solved,
            "pct": round(100.0*len(solved)/len(chal), 2),
            "wall_clock_seconds": round(time.time()-t0, 1)}


def main():
    out = {}
    out["eval"] = preflight(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_solutions.json",
        "eval",
    )
    out["training"] = preflight(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_training_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_training_solutions.json",
        "training",
    )
    rcpt = ROOT / "receipts" / "100day" / "wave16_compose_preflight.json"
    rcpt.parent.mkdir(parents=True, exist_ok=True)
    rcpt.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[receipt] {rcpt}", flush=True)


if __name__ == "__main__":
    main()
