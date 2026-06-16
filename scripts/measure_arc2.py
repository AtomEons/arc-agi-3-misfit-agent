"""Measure arc2_solver on ARC-AGI-2 challenges + solutions.

Usage:
    python scripts/measure_arc2.py --split training --depth 1 --budget 5 --limit 1000
    python scripts/measure_arc2.py --split evaluation --depth 2 --budget 15 --limit 120

Per ARC-AGI-2 contract: for each test input, submit 2 attempts; if either
matches the gold solution exactly, the task counts as solved. We compute
both per-task pass and the rule chosen for the attempt.

Emits a JSONL receipt at receipts/arc/measurement_<split>_d<depth>_<utc>.jsonl
with one row per task plus a final summary row.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.arc2_solver import solve_task  # noqa: E402


def _grid_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def _serializable(grid) -> list[list[int]]:
    return np.asarray(grid, dtype=np.int32).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["training", "evaluation"], required=True)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--budget", type=float, default=5.0, help="Seconds per task (informational; solver is fast)")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    challenges_path = ROOT / "_research" / "arc-agi-2" / f"arc-agi_{args.split}_challenges.json"
    solutions_path = ROOT / "_research" / "arc-agi-2" / f"arc-agi_{args.split}_solutions.json"

    if not challenges_path.exists() or not solutions_path.exists():
        print(f"missing files: {challenges_path} or {solutions_path}", file=sys.stderr)
        return 2

    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))

    utc = int(time.time())
    out_dir = ROOT / "receipts" / "arc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or out_dir / f"measurement_{args.split}_d{args.depth}_{utc}.jsonl"

    task_ids = sorted(challenges.keys())[: args.limit]
    total_pairs = 0
    solved_pairs = 0
    solved_tasks = 0
    total_tasks = 0
    rules_used: dict[str, int] = {}
    start = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        header = {
            "kind": "measurement_header",
            "split": args.split,
            "compose_depth": args.depth,
            "budget_seconds_per_task": args.budget,
            "tasks_planned": len(task_ids),
            "started_unix": utc,
        }
        f.write(json.dumps(header) + "\n")
        f.flush()

        for i, task_id in enumerate(task_ids, 1):
            task = challenges[task_id]
            train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                            np.asarray(p["output"], dtype=np.int32))
                           for p in task["train"]]
            test_pairs = task["test"]
            golds = solutions.get(task_id, [])

            t0 = time.time()
            per_input_results: list[dict] = []
            task_solved_count = 0
            for test_idx, tp in enumerate(test_pairs):
                test_input = np.asarray(tp["input"], dtype=np.int32)
                if test_idx >= len(golds):
                    continue
                gold = np.asarray(golds[test_idx], dtype=np.int32)
                try:
                    a1, a2 = solve_task(train_pairs, test_input, compose_depth=args.depth)
                except Exception as e:
                    per_input_results.append({
                        "test_idx": test_idx,
                        "solved": False,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    total_pairs += 1
                    continue
                solved = _grid_equal(a1, gold) or _grid_equal(a2, gold)
                total_pairs += 1
                if solved:
                    solved_pairs += 1
                    task_solved_count += 1
                per_input_results.append({
                    "test_idx": test_idx,
                    "solved": solved,
                })
            elapsed_ms = int((time.time() - t0) * 1000)
            total_tasks += 1
            task_solved = task_solved_count == len(test_pairs) and len(test_pairs) > 0
            if task_solved:
                solved_tasks += 1
            f.write(json.dumps({
                "kind": "task",
                "task_id": task_id,
                "train_pair_count": len(train_pairs),
                "test_input_count": len(test_pairs),
                "test_inputs_solved": task_solved_count,
                "task_solved_all": task_solved,
                "elapsed_ms": elapsed_ms,
                "per_input_results": per_input_results,
            }) + "\n")
            if i % 50 == 0:
                f.flush()
                print(f"  [{i}/{len(task_ids)}] tasks_solved={solved_tasks} pairs_solved={solved_pairs}/{total_pairs}", flush=True)

        wall = time.time() - start
        footer = {
            "kind": "measurement_footer",
            "split": args.split,
            "compose_depth": args.depth,
            "tasks_total": total_tasks,
            "tasks_solved_all": solved_tasks,
            "task_solve_rate": (solved_tasks / total_tasks) if total_tasks else 0.0,
            "test_inputs_total": total_pairs,
            "test_inputs_solved": solved_pairs,
            "pair_solve_rate": (solved_pairs / total_pairs) if total_pairs else 0.0,
            "wall_clock_seconds": round(wall, 2),
            "ended_unix": int(time.time()),
        }
        f.write(json.dumps(footer) + "\n")
        f.flush()

    print(f"\n=== MEASUREMENT COMPLETE ({args.split}, depth={args.depth}) ===")
    print(f"  tasks_solved_all = {solved_tasks}/{total_tasks} ({solved_tasks/max(total_tasks,1)*100:.2f}%)")
    print(f"  test_inputs_solved = {solved_pairs}/{total_pairs} ({solved_pairs/max(total_pairs,1)*100:.2f}%)")
    print(f"  wall_clock = {wall:.1f}s")
    print(f"  receipt = {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
