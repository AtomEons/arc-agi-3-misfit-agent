"""Wave 10 — eval-gap diagnostic.

The 65-rule grammar scores 31/1000 = 3.10% on training but 0/120 = 0.00%
on eval. WHY?

For each of the 120 eval tasks, count:
  - fit_count    = number of rules that .fit() on the train pairs
  - correct_pred = number of rule predictions that match gold test output
  - any_correct  = does ANY rule predict correctly?

Triage:
  - mean fit_count == 0       -> fit-contract too restrictive (gap = expressiveness)
  - mean fit_count > 0, correct_pred == 0 -> grammar overfits train (gap = generalization)
  - any_correct == 0 across all tasks -> structural mismatch (grammar wrong family)

Outputs receipt at receipts/100day/wave10_eval_diagnostic.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from misfit_agent.arc2_solver import _rule_factories  # noqa: E402

CHAL_PATH = ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_challenges.json"
SOL_PATH = ROOT / "_research" / "arc-agi-2" / "arc-agi_evaluation_solutions.json"


def _gold_outputs(task_id, solutions):
    sol = solutions[task_id]
    return [np.asarray(g, dtype=np.int32) for g in sol]


def main():
    t0 = time.time()
    challenges = json.loads(CHAL_PATH.read_text(encoding="utf-8"))
    solutions = json.loads(SOL_PATH.read_text(encoding="utf-8"))
    factories = _rule_factories()
    print(f"[boot] {len(challenges)} eval tasks, {len(factories)} rule factories")

    rows = []
    rule_fit_counter = {}  # rule_class_name -> fit_count
    rule_correct_counter = {}  # rule_class_name -> correct_count
    fit_distribution = []
    pred_correct_distribution = []
    any_correct_task_ids = []
    fit_but_wrong_task_ids = []

    for idx, (task_id, task) in enumerate(challenges.items()):
        train_pairs = [(np.asarray(p["input"], dtype=np.int32),
                        np.asarray(p["output"], dtype=np.int32))
                       for p in task["train"]]
        gold_outs = _gold_outputs(task_id, solutions)
        test_inputs = [np.asarray(t["input"], dtype=np.int32) for t in task["test"]]

        fit_count = 0
        pred_correct_count = 0
        any_correct = False
        for fa in factories:
            try:
                rule = fa()
                if not rule.fit(train_pairs):
                    continue
                fit_count += 1
                cls_name = rule.__class__.__name__
                rule_fit_counter[cls_name] = rule_fit_counter.get(cls_name, 0) + 1
                # Predict test
                for ti, gold in zip(test_inputs, gold_outs):
                    try:
                        pred = rule.predict(ti)
                    except Exception:
                        continue
                    pred = np.asarray(pred, dtype=np.int32)
                    if pred.shape == gold.shape and np.array_equal(pred, gold):
                        pred_correct_count += 1
                        rule_correct_counter[cls_name] = rule_correct_counter.get(cls_name, 0) + 1
                        any_correct = True
                        break
            except Exception:
                continue

        fit_distribution.append(fit_count)
        pred_correct_distribution.append(pred_correct_count)
        if any_correct:
            any_correct_task_ids.append(task_id)
        elif fit_count > 0:
            fit_but_wrong_task_ids.append(task_id)

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(challenges)}] wall={time.time()-t0:.1f}s "
                  f"fit_mean={np.mean(fit_distribution):.2f} "
                  f"correct_total={sum(1 for x in pred_correct_distribution if x>0)}")

    elapsed = time.time() - t0
    total_correct = sum(1 for x in pred_correct_distribution if x > 0)
    total_fit_zero = sum(1 for x in fit_distribution if x == 0)
    total_fit_positive = sum(1 for x in fit_distribution if x > 0)

    summary = {
        "wave": 10,
        "verb": "eval_gap_diagnostic",
        "tier_1_strict": True,
        "wall_clock_seconds": round(elapsed, 1),
        "eval_set": str(CHAL_PATH),
        "tasks_total": len(challenges),
        "factories_count": len(factories),
        "fit_distribution": {
            "zero_fits": total_fit_zero,
            "any_fit": total_fit_positive,
            "fit_count_mean": float(np.mean(fit_distribution)),
            "fit_count_max": int(np.max(fit_distribution)) if fit_distribution else 0,
            "fit_count_p50": int(np.median(fit_distribution)),
            "fit_count_p90": int(np.percentile(fit_distribution, 90)) if fit_distribution else 0,
        },
        "predict_distribution": {
            "any_correct_tasks": total_correct,
            "any_correct_pct": round(100.0 * total_correct / len(challenges), 2),
            "fit_but_all_wrong_tasks": len(fit_but_wrong_task_ids),
        },
        "rules_that_fit": dict(sorted(rule_fit_counter.items(),
                                      key=lambda kv: -kv[1])),
        "rules_that_predict_correctly": dict(sorted(rule_correct_counter.items(),
                                                    key=lambda kv: -kv[1])),
        "any_correct_task_ids": any_correct_task_ids,
        "fit_but_wrong_sample": fit_but_wrong_task_ids[:30],
    }

    out_path = ROOT / "receipts" / "100day" / "wave10_eval_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] receipt -> {out_path}")
    print(json.dumps({k: summary[k] for k in (
        "fit_distribution", "predict_distribution"
    )}, indent=2))


if __name__ == "__main__":
    main()
