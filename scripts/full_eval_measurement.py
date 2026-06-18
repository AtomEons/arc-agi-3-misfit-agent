"""Full eval + training measurement with ALL waves (1-9 + 11-15).

Uses solve_task with use_dsl=True. The TRUE Kaggle-equivalent run.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.arc2_solver import solve_task  # noqa: E402


def _gold(tid, sols):
    return [np.asarray(g, dtype=np.int32) for g in sols[tid]]


def measure(chal_path, sol_path, label, time_budget_per_task=15.0):
    chal = json.loads(Path(chal_path).read_text(encoding="utf-8"))
    sols = json.loads(Path(sol_path).read_text(encoding="utf-8"))
    solved = []
    t0 = time.time()
    for idx, (tid, task) in enumerate(chal.items()):
        train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                        np.asarray(p["output"], dtype=np.int32)) for p in task["train"]]
        gold_outs = _gold(tid, sols)
        test_inputs = [np.asarray(t["input"], dtype=np.int32) for t in task["test"]]
        all_ok = True
        for ti, gold in zip(test_inputs, gold_outs):
            try:
                a1, a2 = solve_task(
                    train_pairs, ti,
                    compose_depth=1,
                    use_dsl=True,
                    dsl_time_budget_s=2.0,
                    total_time_budget_s=time_budget_per_task,
                )
                a1 = np.asarray(a1, dtype=np.int32)
                a2 = np.asarray(a2, dtype=np.int32)
            except Exception:
                all_ok = False; break
            ok = (a1.shape == gold.shape and np.array_equal(a1, gold)) or \
                 (a2.shape == gold.shape and np.array_equal(a2, gold))
            if not ok:
                all_ok = False; break
        if all_ok:
            solved.append(tid)
        if (idx + 1) % 25 == 0:
            print(f"  [{label}] {idx+1}/{len(chal)} {time.time()-t0:.1f}s "
                  f"solved={len(solved)}/{idx+1}", flush=True)
    elapsed = time.time() - t0
    print(f"[{label}] DONE {elapsed:.1f}s solved={len(solved)}/{len(chal)} "
          f"= {100.0*len(solved)/len(chal):.2f}%", flush=True)
    return {"label": label, "total": len(chal), "solved_count": len(solved),
            "solved_ids": solved, "pct": round(100.0*len(solved)/len(chal), 2),
            "wall_clock_seconds": round(elapsed, 1)}


def main():
    out = {}
    out["eval"] = measure(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_solutions.json",
        "eval",
        time_budget_per_task=15.0,
    )
    out["training"] = measure(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_training_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_training_solutions.json",
        "training",
        time_budget_per_task=15.0,
    )
    rcpt = ROOT / "receipts" / "100day" / "wave15_full_measurement.json"
    rcpt.parent.mkdir(parents=True, exist_ok=True)
    rcpt.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[receipt] {rcpt}", flush=True)


if __name__ == "__main__":
    main()
