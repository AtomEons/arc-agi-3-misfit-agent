"""AtomEons Misfit Agent — ARC-AGI-2 Wave 9 Submission

Tier-1 strict (no LLM in inference, no pretrained weights, no learned
parameters at eval). Spelke core-knowledge priors only. 65 rule templates,
property-bound at predict-time.

Wave 9 training-split measurement: 31/1000 = 3.10% solved (depth-1 search,
exact-match verification on test inputs). Pre-flight predicted exactly 31;
measurement confirmed.

Source: github.com/AtomEons/arc-agi-3-misfit-agent
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_INPUT_BASE = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PKG_OUT = WORK / "src" / "misfit_agent"


def _find_path(filename_or_dirname: str) -> Path | None:
    """Search /kaggle/input recursively for a file or directory by name."""
    for p in _INPUT_BASE.rglob(filename_or_dirname):
        return p
    return None


def _resolve_input_paths():
    global INPUT_ROOT, SUBSTRATE_ROOT
    challenges = _find_path("arc-agi_test_challenges.json")
    INPUT_ROOT = challenges.parent if challenges else _INPUT_BASE / "arc-prize-2026-arc-agi-2"
    substrate_solver = _find_path("arc2_solver.py")
    SUBSTRATE_ROOT = substrate_solver.parent if substrate_solver else _INPUT_BASE / "atomeons-misfit-substrate"
    print(f"[paths] INPUT_ROOT={INPUT_ROOT}")
    print(f"[paths] SUBSTRATE_ROOT={SUBSTRATE_ROOT}")


INPUT_ROOT = _INPUT_BASE / "arc-prize-2026-arc-agi-2"
SUBSTRATE_ROOT = _INPUT_BASE / "atomeons-misfit-substrate"


def _install_substrate():
    """Mirror substrate files into a /kaggle/working/src/misfit_agent package."""
    PKG_OUT.mkdir(parents=True, exist_ok=True)
    for item in SUBSTRATE_ROOT.iterdir():
        dst = PKG_OUT / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    files = sorted(p.relative_to(PKG_OUT) for p in PKG_OUT.rglob("*.py"))
    print(f"[setup] installed {len(files)} files into {PKG_OUT}")
    print(f"[setup] manifest: {files}")


def _import_solver():
    sys.path.insert(0, str(WORK / "src"))
    from misfit_agent.arc2_solver import solve_task  # noqa: E402
    return solve_task


def _write_identity_submission():
    """Fallback: write submission.json where every attempt is the identity
    grid of the test input. Lets a kernel run land on the leaderboard with
    a non-error baseline even when the substrate dataset failed to mount."""
    challenges_path = INPUT_ROOT / "arc-agi_test_challenges.json"
    if not challenges_path.exists():
        candidates = list(Path("/kaggle/input").rglob("arc-agi_test_challenges.json"))
        if candidates:
            challenges_path = candidates[0]
    if not challenges_path.exists():
        print(f"[fatal] cannot find arc-agi_test_challenges.json")
        return
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    submission = {}
    for tid, task in challenges.items():
        entries = []
        for test in task["test"]:
            arr = np.asarray(test["input"], dtype=np.int32).tolist()
            entries.append({"attempt_1": arr, "attempt_2": arr})
        submission[tid] = entries
    out_path = WORK / "submission.json"
    out_path.write_text(json.dumps(submission), encoding="utf-8")
    print(f"[done-identity] wrote identity submission for {len(submission)} tasks")


def _two_attempts(solve_task, task_dict):
    """Return list of {attempt_1, attempt_2} dicts, one per test input.
    solve_task returns (attempt_1_grid, attempt_2_grid) as numpy arrays.
    """
    train_pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task_dict["train"]]
    out = []
    for test in task_dict["test"]:
        test_inp = np.array(test["input"])
        try:
            attempt_1, attempt_2 = solve_task(
                train_pairs, test_inp,
                compose_depth=1,
                use_dsl=False,
            )
            a1 = np.asarray(attempt_1, dtype=np.int32).tolist()
            a2 = np.asarray(attempt_2, dtype=np.int32).tolist()
        except Exception as exc:
            print(f"[warn] solve_task failed: {exc}")
            a1 = np.asarray(test_inp, dtype=np.int32).tolist()
            a2 = a1
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out


def main():
    print("[boot] AtomEons Misfit Agent — Wave 9 (65 rules, Tier-1 strict)")
    t0 = time.time()

    import os
    print(f"[diag] /kaggle/input contents: {sorted(os.listdir('/kaggle/input'))}")
    for sub in ("competitions", "datasets"):
        p = _INPUT_BASE / sub
        if p.exists():
            print(f"[diag] /kaggle/input/{sub} contents: {sorted(os.listdir(p))[:10]}")
    _resolve_input_paths()
    if not SUBSTRATE_ROOT.exists():
        print(f"[diag] substrate path not found — writing identity baseline")
        _write_identity_submission()
        return

    _install_substrate()
    solve_task = _import_solver()
    print("[boot] solver imported")

    challenges_path = INPUT_ROOT / "arc-agi_test_challenges.json"
    if not challenges_path.exists():
        # Local-dev fallback path
        alt = Path("/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
        if alt.exists():
            challenges_path = alt
        else:
            raise SystemExit(f"missing test challenges at {challenges_path}")
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    print(f"[boot] {len(challenges)} test tasks to solve")

    submission = {}
    started = time.time()
    for i, (task_id, task) in enumerate(challenges.items()):
        try:
            submission[task_id] = _two_attempts(solve_task, task)
        except Exception as exc:
            print(f"[warn] task {task_id} fell back to identity: {exc}")
            entries = []
            for test in task["test"]:
                arr = np.asarray(test["input"], dtype=np.int32).tolist()
                entries.append({"attempt_1": arr, "attempt_2": arr})
            submission[task_id] = entries
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(challenges)}] wall={time.time()-started:.1f}s")

    out_path = WORK / "submission.json"
    out_path.write_text(json.dumps(submission), encoding="utf-8")
    print(f"[done] wrote {out_path} for {len(submission)} tasks in {time.time()-t0:.1f}s")
    non_identity = sum(
        1 for tid, entries in submission.items()
        for e in entries
        if e["attempt_1"] != np.asarray(challenges[tid]["test"][0]["input"]).tolist()
    )
    print(f"[done] entries with non-identity attempt_1: {non_identity}")


if __name__ == "__main__":
    main()
