"""Wave 12 + 13 pre-flight on eval and training.

Wave 12: ObjectCorrespondenceProgram (per-task program synth).
Wave 13: NovelColorRecolor + PaletteBijectionWithIdentityExtension + PaintBgWithMissingNonBgColor.

Crucially, run on EVAL since the wave-10 diagnostic showed ZERO global-fit
rules engage with eval. If wave 12 fits ANY eval task, that's first contact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.rules_v3.wave12 import ObjectCorrespondenceProgram  # noqa: E402
from misfit_agent.rules_v3.wave13 import (  # noqa: E402
    NovelColorRecolor, PaletteBijectionWithIdentityExtension,
    PaintBgWithMissingNonBgColor,
)

FACTORIES = [
    ObjectCorrespondenceProgram,
    NovelColorRecolor,
    PaletteBijectionWithIdentityExtension,
    PaintBgWithMissingNonBgColor,
]


def _gold_outputs(task_id, solutions):
    return [np.asarray(g, dtype=np.int32) for g in solutions[task_id]]


def preflight(chal_path, sol_path, label):
    challenges = json.loads(Path(chal_path).read_text(encoding="utf-8"))
    solutions = json.loads(Path(sol_path).read_text(encoding="utf-8"))

    fits = {f.__name__: 0 for f in FACTORIES}
    solves = {f.__name__: [] for f in FACTORIES}
    any_solver_solved = set()
    t0 = time.time()
    for idx, (tid, task) in enumerate(challenges.items()):
        train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                        np.asarray(p["output"], dtype=np.int32))
                       for p in task["train"]]
        gold_outs = _gold_outputs(tid, solutions)
        test_inputs = [np.asarray(t["input"], dtype=np.int32) for t in task["test"]]
        for Factory in FACTORIES:
            rule = Factory()
            try:
                if not rule.fit(train_pairs):
                    continue
            except Exception:
                continue
            fits[Factory.__name__] += 1
            ok = True
            for ti, gold in zip(test_inputs, gold_outs):
                try:
                    pred = np.asarray(rule.predict(ti), dtype=np.int32)
                except Exception:
                    ok = False
                    break
                if pred.shape != gold.shape or not np.array_equal(pred, gold):
                    ok = False
                    break
            if ok:
                solves[Factory.__name__].append(tid)
                any_solver_solved.add(tid)
        if (idx + 1) % 50 == 0:
            print(f"  [{label}] {idx+1}/{len(challenges)} wall={time.time()-t0:.1f}s "
                  f"any_solved={len(any_solver_solved)}", flush=True)
    print(f"[{label}] done in {time.time()-t0:.1f}s", flush=True)
    print(f"  fits: {fits}", flush=True)
    print(f"  solves: {solves}", flush=True)
    print(f"  any_solved: {len(any_solver_solved)} / {len(challenges)} "
          f"= {100.0*len(any_solver_solved)/len(challenges):.2f}%", flush=True)
    return {
        "label": label,
        "tasks_total": len(challenges),
        "fits_per_rule": fits,
        "solves_per_rule": solves,
        "any_solver_solved_count": len(any_solver_solved),
        "any_solver_solved_ids": sorted(any_solver_solved),
        "any_solver_solved_pct": round(
            100.0 * len(any_solver_solved) / len(challenges), 2),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }


def main():
    out = {}
    out["eval"] = preflight(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_solutions.json",
        "eval",
    )
    train_chal = ROOT / "_research" / "arc-agi-2" / "arc-agi_training_challenges.json"
    train_sol = ROOT / "_research" / "arc-agi-2" / "arc-agi_training_solutions.json"
    if train_chal.exists() and train_sol.exists():
        out["training"] = preflight(train_chal, train_sol, "training")
    receipt = ROOT / "receipts" / "100day" / "wave12_13_preflight.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[receipt] {receipt}", flush=True)


if __name__ == "__main__":
    main()
