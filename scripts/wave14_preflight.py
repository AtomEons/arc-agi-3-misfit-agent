"""Wave 14 pre-flight on eval + training (handles shape-changing tasks)."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.rules_v3.wave14 import (  # noqa
    CropToSelectedObject, CropToColorRegion,
    ExtendedObjectCorrespondenceProgram,
)

FACTORIES = [
    CropToSelectedObject,
    CropToColorRegion,
    ExtendedObjectCorrespondenceProgram,
]


def _gold(tid, sols):
    return [np.asarray(g, dtype=np.int32) for g in sols[tid]]


def preflight(chal_path, sol_path, label):
    chal = json.loads(Path(chal_path).read_text(encoding="utf-8"))
    sols = json.loads(Path(sol_path).read_text(encoding="utf-8"))
    fits = {f.__name__: 0 for f in FACTORIES}
    solves = {f.__name__: [] for f in FACTORIES}
    any_solved = set()
    t0 = time.time()
    for idx, (tid, task) in enumerate(chal.items()):
        train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                        np.asarray(p["output"], dtype=np.int32)) for p in task["train"]]
        gold_outs = _gold(tid, sols)
        test_inputs = [np.asarray(t["input"], dtype=np.int32) for t in task["test"]]
        for F in FACTORIES:
            r = F()
            try:
                if not r.fit(train_pairs):
                    continue
            except Exception:
                continue
            fits[F.__name__] += 1
            ok = True
            for ti, gold in zip(test_inputs, gold_outs):
                try:
                    pred = np.asarray(r.predict(ti), dtype=np.int32)
                except Exception:
                    ok = False; break
                if pred.shape != gold.shape or not np.array_equal(pred, gold):
                    ok = False; break
            if ok:
                solves[F.__name__].append(tid)
                any_solved.add(tid)
        if (idx + 1) % 100 == 0:
            print(f"  [{label}] {idx+1}/{len(chal)} {time.time()-t0:.1f}s any={len(any_solved)}", flush=True)
    print(f"[{label}] {time.time()-t0:.1f}s fits={fits} any={len(any_solved)}/{len(chal)}", flush=True)
    return {
        "label": label, "fits": fits, "solves": solves,
        "any_solved_count": len(any_solved),
        "any_solved_ids": sorted(any_solved),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }


def main():
    out = {}
    out["eval"] = preflight(
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_challenges.json",
        ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_solutions.json",
        "eval",
    )
    tc = ROOT / "_research" / "arc-agi-2" / "arc-agi_training_challenges.json"
    ts = ROOT / "_research" / "arc-agi-2" / "arc-agi_training_solutions.json"
    if tc.exists() and ts.exists():
        out["training"] = preflight(tc, ts, "training")
    rcpt = ROOT / "receipts" / "100day" / "wave14_preflight.json"
    rcpt.parent.mkdir(parents=True, exist_ok=True)
    rcpt.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[receipt] {rcpt}", flush=True)


if __name__ == "__main__":
    main()
