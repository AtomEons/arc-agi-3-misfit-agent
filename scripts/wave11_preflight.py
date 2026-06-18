"""Wave 11 pre-flight.

For each task in eval (and training), check whether ANY of the new wave 11
primitives PatternCompleteByPeriodicity / Symmetry / Tile, BgHoleFillByOrbit,
MirrorFillByAxis can:
  1. fit() on the train pairs
  2. .predict() the test input to match gold

Reports per-task IDs that newly lift. Compare against existing wave9 baseline
to find newly-solved tasks attributable to wave 11.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.rules_v3.wave11 import (  # noqa: E402
    PatternCompleteByPeriodicity, PatternCompleteBySymmetry,
    PatternCompleteByTile, BgHoleFillByOrbit, MirrorFillByAxis,
)

WAVE11_FACTORIES = [
    PatternCompleteByPeriodicity,
    PatternCompleteBySymmetry,
    PatternCompleteByTile,
    BgHoleFillByOrbit,
    MirrorFillByAxis,
]


def _gold_outputs(task_id, solutions):
    return [np.asarray(g, dtype=np.int32) for g in solutions[task_id]]


def preflight(challenges_path, solutions_path, label):
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))

    fits = {f.__name__: 0 for f in WAVE11_FACTORIES}
    solves = {f.__name__: [] for f in WAVE11_FACTORIES}
    any_solver_solved = set()

    t0 = time.time()
    for idx, (task_id, task) in enumerate(challenges.items()):
        train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                        np.asarray(p["output"], dtype=np.int32))
                       for p in task["train"]]
        gold_outs = _gold_outputs(task_id, solutions)
        test_inputs = [np.asarray(t["input"], dtype=np.int32) for t in task["test"]]

        for Factory in WAVE11_FACTORIES:
            rule = Factory()
            try:
                if not rule.fit(train_pairs):
                    continue
            except Exception:
                continue
            fits[Factory.__name__] += 1
            try:
                solved_all = True
                for ti, gold in zip(test_inputs, gold_outs):
                    pred = np.asarray(rule.predict(ti), dtype=np.int32)
                    if pred.shape != gold.shape or not np.array_equal(pred, gold):
                        solved_all = False
                        break
                if solved_all:
                    solves[Factory.__name__].append(task_id)
                    any_solver_solved.add(task_id)
            except Exception:
                continue

        if (idx + 1) % 50 == 0:
            print(f"  [{label}] {idx+1}/{len(challenges)} wall={time.time()-t0:.1f}s "
                  f"any_solved={len(any_solver_solved)}")

    print(f"[done {label}] {time.time()-t0:.1f}s")
    print(f"  fits: {fits}")
    print(f"  solves: {solves}")
    print(f"  any_solved: {len(any_solver_solved)} / {len(challenges)}  "
          f"= {100.0*len(any_solver_solved)/len(challenges):.2f}%")
    return {
        "label": label,
        "tasks_total": len(challenges),
        "wave11_fits_per_rule": fits,
        "wave11_solves_per_rule": solves,
        "wave11_any_solver_solved_count": len(any_solver_solved),
        "wave11_any_solver_solved_ids": sorted(any_solver_solved),
        "wave11_any_solver_solved_pct": round(
            100.0 * len(any_solver_solved) / len(challenges), 2),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }


def main():
    out = {}
    eval_report = preflight(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_solutions.json",
        "eval",
    )
    out["eval"] = eval_report

    train_chal = ROOT / "_research" / "arc-agi-2" / "arc-agi_training_challenges.json"
    train_sol = ROOT / "_research" / "arc-agi-2" / "arc-agi_training_solutions.json"
    if train_chal.exists() and train_sol.exists():
        train_report = preflight(train_chal, train_sol, "training")
        out["training"] = train_report

    receipt = ROOT / "receipts" / "100day" / "wave11_preflight.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[receipt] {receipt}")


if __name__ == "__main__":
    main()
