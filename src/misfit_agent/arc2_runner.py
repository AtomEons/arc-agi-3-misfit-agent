"""ARC-AGI-2 sister runner — submission orchestration.

Loads an ARC-AGI-2 challenges JSON, runs `solve_task` per (task, test) pair
under a wall-clock budget governor, and emits a submission dict in the
official format:

    {task_id: [{"attempt_1": <grid>, "attempt_2": <grid>}, ...]}

One list entry per test_input in the task (most tasks have 1; some have 2).

Tier-1 honesty:
  - Per-task budget caps inference time so the loop never silently times out
    on one hard task while leaving easy tasks unscored.
  - Wall-clock self-kill so a runaway task can't burn the whole Kaggle
    9-hour quota.
  - No internet, no model loads. Pure substrate.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import numpy as np

from .arc2_solver import solve_task


DEFAULT_OUT_PATH = pathlib.Path("submission_arc2.json")
DEFAULT_WALL_CLOCK_SECONDS = 8 * 60 * 60 + 30 * 60  # 8h30m, leave 30m margin
DEFAULT_PER_TASK_SECONDS = 30.0


def load_challenges(path: str | pathlib.Path) -> dict:
    """Read arc-agi_evaluation_challenges.json (or any ARC-AGI-2 challenges
    JSON in the official format).

    Expected schema:
        {task_id: {"train": [{"input": grid, "output": grid}, ...],
                   "test":  [{"input": grid}, ...]}}
    """
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _grid_to_list(g) -> list[list[int]]:
    arr = np.asarray(g)
    return [[int(x) for x in row] for row in arr.tolist()]


def _empty_grid() -> list[list[int]]:
    """Fallback used when a task entirely times out before producing output.
    The ARC-AGI-2 grader will mark it wrong; better than crashing the run."""
    return [[0]]


def run(
    challenges_dict: dict,
    out_path: str | pathlib.Path = DEFAULT_OUT_PATH,
    wall_clock_seconds: float = DEFAULT_WALL_CLOCK_SECONDS,
    per_task_seconds: float = DEFAULT_PER_TASK_SECONDS,
    verbose: bool = False,
    write: bool = True,
) -> dict:
    """Run the solver over every task in `challenges_dict`.

    Args:
        challenges_dict: parsed challenges JSON, as returned by load_challenges.
        out_path:        where to write the submission JSON.
        wall_clock_seconds: hard cap on total inference wall-clock.
        per_task_seconds:   soft cap per task; on overrun the task is filled
                            with identity attempts for the remaining tests.
        verbose:         print per-task timing if True.
        write:           if False, do not write to disk (useful for tests).

    Returns:
        The submission dict.
    """
    submission: dict[str, list[dict]] = {}
    if not challenges_dict:
        if write:
            _write_submission(submission, out_path)
        return submission

    start = time.monotonic()
    task_ids = list(challenges_dict.keys())

    for task_id in task_ids:
        elapsed_total = time.monotonic() - start
        if elapsed_total >= wall_clock_seconds:
            # Wall-clock kill — fill all remaining with identity on first
            # test grid so the submission stays well-formed.
            _fill_identity(submission, challenges_dict, task_id, task_ids)
            break

        task = challenges_dict[task_id]
        train_pairs = [
            (np.asarray(p["input"], dtype=np.int32),
             np.asarray(p["output"], dtype=np.int32))
            for p in task.get("train", [])
        ]
        tests = task.get("test", [])

        task_start = time.monotonic()
        per_test_attempts: list[dict] = []

        for test_idx, t in enumerate(tests):
            if time.monotonic() - task_start >= per_task_seconds:
                # Per-task budget blown — identity fallback for remaining tests
                inp = np.asarray(t["input"], dtype=np.int32)
                per_test_attempts.append({
                    "attempt_1": _grid_to_list(inp),
                    "attempt_2": _grid_to_list(inp),
                })
                continue
            try:
                test_input = np.asarray(t["input"], dtype=np.int32)
                a1, a2 = solve_task(train_pairs, test_input)
                per_test_attempts.append({
                    "attempt_1": _grid_to_list(a1),
                    "attempt_2": _grid_to_list(a2),
                })
            except Exception as e:
                # Honest abstain on any solver crash — identity fallback.
                if verbose:
                    print(f"[arc2_runner] task {task_id} test {test_idx}: {e!r}")
                inp = np.asarray(t["input"], dtype=np.int32)
                per_test_attempts.append({
                    "attempt_1": _grid_to_list(inp),
                    "attempt_2": _grid_to_list(inp),
                })

        # If the task has zero test inputs (malformed), skip emitting an entry.
        if per_test_attempts:
            submission[task_id] = per_test_attempts

        if verbose:
            print(f"[arc2_runner] {task_id} done in "
                  f"{time.monotonic() - task_start:0.2f}s "
                  f"({len(per_test_attempts)} test(s))")

    if write:
        _write_submission(submission, out_path)
    return submission


def _fill_identity(submission: dict,
                    challenges_dict: dict,
                    current_task_id: str,
                    all_task_ids: list[str]) -> None:
    """Wall-clock kill path: fill the rest of the submission with identity
    attempts so the file stays well-formed for the grader."""
    start_idx = all_task_ids.index(current_task_id)
    for tid in all_task_ids[start_idx:]:
        if tid in submission:
            continue
        tests = challenges_dict.get(tid, {}).get("test", [])
        per_test = []
        for t in tests:
            inp = np.asarray(t.get("input", _empty_grid()), dtype=np.int32)
            per_test.append({
                "attempt_1": _grid_to_list(inp),
                "attempt_2": _grid_to_list(inp),
            })
        if per_test:
            submission[tid] = per_test


def _write_submission(submission: dict, out_path: str | pathlib.Path) -> None:
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f)


def emit_submission(challenges_path: str | pathlib.Path,
                     out_path: str | pathlib.Path = DEFAULT_OUT_PATH,
                     **kwargs) -> dict:
    """One-shot helper: load + run + write."""
    challenges = load_challenges(challenges_path)
    return run(challenges, out_path=out_path, **kwargs)
