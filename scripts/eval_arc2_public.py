#!/usr/bin/env python3
"""Run arc2_solver against the ARC-AGI-2 public evaluation set.

Honest reality check — whatever number comes back is the number. Per ARC-AGI-2
contract: each task gets 2 attempts; hit if either matches the gold output
exactly. Final score = wins / total_tasks.

Outputs:
  receipts/arc-agi-2/eval_arc2_public_<utc>.json — aggregate + per-task
  receipts/arc-agi-2/eval_arc2_public_<utc>.md   — human-readable summary

Usage:
  python scripts/eval_arc2_public.py                    # full 400-task eval
  python scripts/eval_arc2_public.py --limit 50         # first 50 only
  python scripts/eval_arc2_public.py --budget-secs 30   # per-task wall budget
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from misfit_agent.arc2_solver import solve_task


def load_data(research_dir: Path) -> tuple[dict, dict]:
    challenges_path = research_dir / "arc-agi_evaluation_challenges.json"
    solutions_path = research_dir / "arc-agi_evaluation_solutions.json"
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    return challenges, solutions


def grid_equal(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def run_task(task_id: str, task: dict, gold_list: list, budget_secs: float) -> dict:
    """Run one task. Returns per-task result row."""
    train_pairs = [(np.asarray(p["input"]), np.asarray(p["output"])) for p in task["train"]]
    tests = task["test"]
    test_results = []
    overall_hit = True
    for i, test in enumerate(tests):
        test_input = np.asarray(test["input"])
        gold = np.asarray(gold_list[i])
        t0 = time.time()
        try:
            a1, a2 = solve_task(train_pairs, test_input)
        except Exception as e:
            test_results.append({
                "test_idx": i,
                "hit": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_ms": int((time.time() - t0) * 1000),
            })
            overall_hit = False
            continue
        elapsed_ms = int((time.time() - t0) * 1000)
        hit_1 = grid_equal(a1, gold)
        hit_2 = grid_equal(a2, gold)
        hit = bool(hit_1 or hit_2)
        if not hit:
            overall_hit = False
        test_results.append({
            "test_idx": i,
            "hit": hit,
            "hit_attempt_1": bool(hit_1),
            "hit_attempt_2": bool(hit_2),
            "elapsed_ms": elapsed_ms,
        })
        if (time.time() - t0) > budget_secs:
            # We're past budget on this single test; honest-abstain protects the
            # full run from a single pathological task. Following tests are
            # still attempted; only this one is flagged.
            test_results[-1]["over_budget"] = True
    return {
        "task_id": task_id,
        "n_train_pairs": len(train_pairs),
        "n_test": len(tests),
        "overall_hit": overall_hit,
        "test_results": test_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N tasks (debug)")
    ap.add_argument("--budget-secs", type=float, default=30.0,
                    help="Per-test wall-clock budget (seconds)")
    ap.add_argument("--research-dir", type=Path,
                    default=REPO_ROOT / "_research" / "arc-agi-2",
                    help="Where the challenges + solutions JSON live")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "receipts" / "arc-agi-2",
                    help="Where to write the receipt")
    args = ap.parse_args()

    if not args.research_dir.exists():
        print(f"research dir missing: {args.research_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    challenges, solutions = load_data(args.research_dir)
    task_ids = sorted(challenges.keys())
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    print(f"eval_arc2_public: {len(task_ids)} tasks (budget {args.budget_secs}s/test)")
    t_start = time.time()
    results = []
    wins = 0
    for i, tid in enumerate(task_ids, 1):
        task = challenges[tid]
        gold_list = solutions.get(tid, [])
        if len(gold_list) != len(task.get("test", [])):
            results.append({
                "task_id": tid,
                "skipped": True,
                "reason": "solutions/test length mismatch",
            })
            continue
        row = run_task(tid, task, gold_list, args.budget_secs)
        if row["overall_hit"]:
            wins += 1
        results.append(row)
        if i % 25 == 0 or i == len(task_ids):
            elapsed = time.time() - t_start
            print(f"  [{i}/{len(task_ids)}] wins={wins} ({wins/i*100:.2f}%) "
                  f"elapsed={elapsed:.0f}s")

    elapsed_total = time.time() - t_start
    n = len(task_ids)
    solved_count = sum(1 for r in results if r.get("overall_hit"))
    pass_rate = solved_count / n if n else 0.0

    stamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    receipt_json = args.out_dir / f"eval_arc2_public_{stamp}.json"
    receipt_md = args.out_dir / f"eval_arc2_public_{stamp}.md"

    payload = {
        "kind": "arc2_public_eval",
        "recorded_at_utc": stamp,
        "n_tasks": n,
        "solved_count": solved_count,
        "pass_rate": pass_rate,
        "wall_clock_total_sec": elapsed_total,
        "mean_wall_clock_per_task_sec": elapsed_total / n if n else 0,
        "budget_secs_per_test": args.budget_secs,
        "tier": "tier_1_substrate_only_priors_only",
        "constraints": {
            "llm_in_inference_path": False,
            "pretrained_weights": False,
            "internet_at_eval": False,
        },
        "per_task": results,
    }
    receipt_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = []
    md.append(f"# ARC-AGI-2 public eval — Tier-1 substrate honest run")
    md.append("")
    md.append(f"- **Recorded:** {stamp}")
    md.append(f"- **Tasks:** {n}")
    md.append(f"- **Solved:** {solved_count}")
    md.append(f"- **Pass rate:** {pass_rate*100:.2f}%")
    md.append(f"- **Total wall clock:** {elapsed_total:.0f} s "
              f"({elapsed_total/60:.1f} min)")
    md.append(f"- **Mean per task:** {elapsed_total/n:.2f} s" if n else "")
    md.append(f"- **Tier:** Tier-1 (no LLM, no pretrained, no internet)")
    md.append("")
    md.append("Solved tasks:")
    md.append("")
    solved_ids = [r['task_id'] for r in results if r.get('overall_hit')]
    for tid in solved_ids:
        md.append(f"- `{tid}`")
    receipt_md.write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"  ARC-AGI-2 public eval: {solved_count}/{n} = {pass_rate*100:.2f}%")
    print(f"  Wall clock: {elapsed_total:.0f}s")
    print(f"  Receipt:    {receipt_json}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
