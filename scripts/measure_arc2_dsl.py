"""Measure the DSL synthesis engine end-to-end on ARC-AGI-2.

Usage:
    python scripts/measure_arc2_dsl.py --split training --limit 100 --budget-per-task 5.0
    python scripts/measure_arc2_dsl.py --split evaluation --limit 50 --budget-per-task 10.0

Per ARC-AGI-2 contract: for each test input, submit 2 attempts; if either
matches the gold solution exactly, that test input counts as solved. A task is
solved-all when every test input is solved; solved-any when at least one is.

Emits a JSONL receipt at receipts/arc/measurement_dsl_<split>_<unix>.jsonl
with one row per task plus a final summary row.

This script drives the typed DSL synthesizer (src/misfit_agent/dsl). It does
NOT touch arc2_solver.py and is independent of measure_arc2.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.dsl import (  # noqa: E402
    Program,
    evaluate,
    synthesize,
)


def _grid_equal(a: Any, b: Any) -> bool:
    try:
        a_arr = np.asarray(a, dtype=np.int32)
        b_arr = np.asarray(b, dtype=np.int32)
    except Exception:
        return False
    if a_arr.shape != b_arr.shape:
        return False
    return bool(np.array_equal(a_arr, b_arr))


def _safe_evaluate(program: Program, test_input: np.ndarray) -> np.ndarray | None:
    """Evaluate a program on a test input, swallowing execution errors."""
    try:
        out = evaluate(program, test_input)
    except Exception:
        return None
    try:
        return np.asarray(out, dtype=np.int32)
    except Exception:
        return None


def run_one_task(
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    test_inputs: list[np.ndarray],
    golds: list[np.ndarray],
    budget_per_task: float = 5.0,
    max_depth: int = 3,
    beam_width: int = 8,
) -> dict:
    """Synthesize on the train pairs, then predict each test input twice.

    Returns a dict with per-test results and aggregate flags. Honors
    budget_per_task by passing it as the synthesis time budget; if the budget
    elapses inside synthesize, we still take whatever programs are returned.
    """
    t0 = time.time()
    programs: list[Program] = []
    try:
        programs = synthesize(
            train_pairs,
            max_depth=max_depth,
            beam_width=beam_width,
            time_budget_s=budget_per_task,
        )
    except Exception:
        programs = []

    # Top-2 programs become attempt-1 and attempt-2; pad with attempt-1 if only one.
    attempt_progs: list[Program | None] = [None, None]
    if len(programs) >= 1:
        attempt_progs[0] = programs[0]
    if len(programs) >= 2:
        attempt_progs[1] = programs[1]
    elif len(programs) == 1:
        attempt_progs[1] = programs[0]

    per_input: list[dict] = []
    any_solved_all = True
    any_solved_any = False
    n_pairs_solved = 0

    for idx, test_input in enumerate(test_inputs):
        if idx >= len(golds):
            continue
        gold = golds[idx]
        a1 = _safe_evaluate(attempt_progs[0], test_input) if attempt_progs[0] is not None else None
        a2 = _safe_evaluate(attempt_progs[1], test_input) if attempt_progs[1] is not None else None
        a1_correct = a1 is not None and _grid_equal(a1, gold)
        a2_correct = a2 is not None and _grid_equal(a2, gold)
        solved = a1_correct or a2_correct
        if solved:
            n_pairs_solved += 1
            any_solved_any = True
        else:
            any_solved_all = False
        per_input.append({
            "test_idx": idx,
            "attempt_1_correct": bool(a1_correct),
            "attempt_2_correct": bool(a2_correct),
            "any_correct": bool(solved),
        })

    if not test_inputs or len(test_inputs) == 0:
        any_solved_all = False

    return {
        "wall_clock_s": time.time() - t0,
        "programs_evaluated": len(programs),
        "test_inputs": len(test_inputs),
        "test_inputs_solved": n_pairs_solved,
        "task_solved_all": bool(any_solved_all and n_pairs_solved > 0),
        "task_solved_any": bool(any_solved_any),
        "per_input": per_input,
    }


def _load_split(split: str) -> tuple[dict, dict, pathlib.Path]:
    challenges_path = ROOT / "_research" / "arc-agi-2" / f"arc-agi_{split}_challenges.json"
    solutions_path = ROOT / "_research" / "arc-agi-2" / f"arc-agi_{split}_solutions.json"
    if not challenges_path.exists() or not solutions_path.exists():
        raise FileNotFoundError(f"missing {challenges_path} or {solutions_path}")
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    return challenges, solutions, challenges_path


def measure(
    split: str,
    limit: int = 10_000,
    budget_per_task: float = 5.0,
    max_depth: int = 3,
    beam_width: int = 8,
    out_path: pathlib.Path | None = None,
) -> dict:
    """Run the full measurement pass and return the footer summary dict."""
    challenges, solutions, _ = _load_split(split)

    utc = int(time.time())
    out_dir = ROOT / "receipts" / "arc"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = out_dir / f"measurement_dsl_{split}_{utc}.jsonl"

    task_ids = sorted(challenges.keys())[:limit]
    total_tasks = 0
    solved_all = 0
    solved_any_count = 0
    total_pairs = 0
    solved_pairs = 0
    walls: list[float] = []
    start = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        header = {
            "kind": "measurement_header",
            "engine": "dsl",
            "split": split,
            "budget_per_task_s": budget_per_task,
            "max_depth": max_depth,
            "beam_width": beam_width,
            "tasks_planned": len(task_ids),
            "started_unix": utc,
        }
        f.write(json.dumps(header) + "\n")
        f.flush()

        for i, task_id in enumerate(task_ids, 1):
            task = challenges[task_id]
            train_pairs = [
                (np.asarray(p["input"], dtype=np.int32),
                 np.asarray(p["output"], dtype=np.int32))
                for p in task["train"]
            ]
            test_pairs = task["test"]
            test_inputs = [np.asarray(tp["input"], dtype=np.int32) for tp in test_pairs]
            gold_list = solutions.get(task_id, [])
            golds = [np.asarray(g, dtype=np.int32) for g in gold_list]

            result = run_one_task(
                train_pairs,
                test_inputs,
                golds,
                budget_per_task=budget_per_task,
                max_depth=max_depth,
                beam_width=beam_width,
            )
            total_tasks += 1
            total_pairs += result["test_inputs"]
            solved_pairs += result["test_inputs_solved"]
            if result["task_solved_all"]:
                solved_all += 1
            if result["task_solved_any"]:
                solved_any_count += 1
            walls.append(result["wall_clock_s"])

            row = {
                "kind": "task",
                "task_id": task_id,
                "wall_clock_s": round(result["wall_clock_s"], 4),
                "programs_evaluated": result["programs_evaluated"],
                "test_inputs": result["test_inputs"],
                "test_inputs_solved": result["test_inputs_solved"],
                "task_solved_all": result["task_solved_all"],
                "task_solved_any": result["task_solved_any"],
                "per_input": result["per_input"],
            }
            f.write(json.dumps(row) + "\n")
            if i % 25 == 0:
                f.flush()
                print(
                    f"  [{i}/{len(task_ids)}] solved_all={solved_all} "
                    f"solved_any={solved_any_count} pairs={solved_pairs}/{total_pairs}",
                    flush=True,
                )

        wall_total = time.time() - start
        mean_wall = (sum(walls) / len(walls)) if walls else 0.0
        footer = {
            "kind": "measurement_footer",
            "engine": "dsl",
            "split": split,
            "tasks_total": total_tasks,
            "tasks_solved_all": solved_all,
            "tasks_solved_any": solved_any_count,
            "task_solve_rate_all": (solved_all / total_tasks) if total_tasks else 0.0,
            "task_solve_rate_any": (solved_any_count / total_tasks) if total_tasks else 0.0,
            "test_inputs_total": total_pairs,
            "test_inputs_solved": solved_pairs,
            "pair_solve_rate": (solved_pairs / total_pairs) if total_pairs else 0.0,
            "mean_wall_clock_s": round(mean_wall, 4),
            "wall_clock_total_s": round(wall_total, 2),
            "ended_unix": int(time.time()),
            "receipt_path": str(out_path),
        }
        f.write(json.dumps(footer) + "\n")
        f.flush()

    return footer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["training", "evaluation"], required=True)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--budget-per-task", type=float, default=5.0,
                    help="Seconds of synthesis budget per task")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    try:
        footer = measure(
            split=args.split,
            limit=args.limit,
            budget_per_task=args.budget_per_task,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            out_path=args.out,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"\n=== DSL MEASUREMENT COMPLETE ({footer['split']}) ===")
    print(f"  tasks_solved_all = {footer['tasks_solved_all']}/{footer['tasks_total']} "
          f"({footer['task_solve_rate_all']*100:.2f}%)")
    print(f"  tasks_solved_any = {footer['tasks_solved_any']}/{footer['tasks_total']} "
          f"({footer['task_solve_rate_any']*100:.2f}%)")
    print(f"  pairs_solved = {footer['test_inputs_solved']}/{footer['test_inputs_total']} "
          f"({footer['pair_solve_rate']*100:.2f}%)")
    print(f"  mean_wall_clock_s = {footer['mean_wall_clock_s']}")
    print(f"  wall_clock_total_s = {footer['wall_clock_total_s']}")
    print(f"  receipt = {footer['receipt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
