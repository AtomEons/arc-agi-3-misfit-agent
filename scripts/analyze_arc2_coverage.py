#!/usr/bin/env python3
"""Classify ARC-AGI-2 public eval tasks by structural transformation type.

Honest priorities receipt: tells us which rule template additions would have
the biggest coverage gain, evidence-first. Run after eval shows 0% — this is
the diagnostic.

For each task, check the train pairs against a set of cheap "is this rule
sufficient?" tests. If ALL train pairs pass a check, that rule is sufficient.

Checks (cheap geometric, no fitting):
  - identity          : output == input
  - recolor           : same shape, same fg/bg positions, color permutation
  - translate         : same shape, integer shift exists
  - flip_h            : output == np.fliplr(input)
  - flip_v            : output == np.flipud(input)
  - rotate_90/180/270 : output == np.rot90(input, k)
  - transpose         : output == input.T
  - crop_to_bbox      : output == input cropped to fg bounding box
  - tile_*            : output dimensions are multiples; tile-grid match
  - shape_changed     : output.shape != input.shape (so most of the above fail)
  - one_test_only     : single test input; otherwise multi-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from misfit_agent.perceptor import _background_color


def is_identity(inp: np.ndarray, out: np.ndarray) -> bool:
    return inp.shape == out.shape and np.array_equal(inp, out)


def is_recolor(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    if np.array_equal(inp, out):
        return False  # identity, not pure recolor
    bg_in = _background_color(inp)
    bg_out = _background_color(out)
    if not np.array_equal(inp == bg_in, out == bg_out):
        return False  # fg/bg cell positions differ
    perm: dict[int, int] = {}
    for c_in in range(10):
        mask = inp == c_in
        if not mask.any():
            continue
        outs = np.unique(out[mask])
        if outs.size != 1:
            return False
        perm[c_in] = int(outs[0])
    return True


def is_translate(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    bg = _background_color(inp)
    rows, cols = inp.shape
    for dy in range(-min(rows, 6), min(rows, 6) + 1):
        for dx in range(-min(cols, 6), min(cols, 6) + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.full_like(inp, fill_value=bg)
            for r in range(rows):
                for c in range(cols):
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        shifted[nr, nc] = inp[r, c]
            if np.array_equal(shifted, out):
                return True
    return False


def is_flip_h(inp, out): return inp.shape == out.shape and np.array_equal(np.fliplr(inp), out)
def is_flip_v(inp, out): return inp.shape == out.shape and np.array_equal(np.flipud(inp), out)
def is_rotate(inp, out, k): return out.shape == np.rot90(inp, k).shape and np.array_equal(np.rot90(inp, k), out)
def is_rot90(inp, out): return is_rotate(inp, out, 1)
def is_rot180(inp, out): return is_rotate(inp, out, 2)
def is_rot270(inp, out): return is_rotate(inp, out, 3)
def is_transpose(inp, out): return out.shape == inp.T.shape and np.array_equal(inp.T, out)


def is_crop_to_bbox(inp: np.ndarray, out: np.ndarray) -> bool:
    """Output equals input cropped to its foreground bounding box."""
    bg = _background_color(inp)
    fg = inp != bg
    if not fg.any():
        return False
    ys, xs = np.where(fg)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    cropped = inp[r0:r1+1, c0:c1+1]
    return cropped.shape == out.shape and np.array_equal(cropped, out)


def is_tile_2x2(inp, out):
    if out.shape != (inp.shape[0] * 2, inp.shape[1] * 2):
        return False
    return np.array_equal(out, np.tile(inp, (2, 2)))


def is_tile_3x3(inp, out):
    if out.shape != (inp.shape[0] * 3, inp.shape[1] * 3):
        return False
    return np.array_equal(out, np.tile(inp, (3, 3)))


def is_symmetrize_h(inp, out):
    """Output == horizontal mirror concatenation (left half + flipped left half)."""
    if inp.shape != out.shape:
        return False
    rows, cols = inp.shape
    if cols % 2 != 0:
        return False
    half = inp[:, :cols // 2]
    sym = np.concatenate([half, np.fliplr(half)], axis=1)
    return np.array_equal(sym, out)


CHECKS = {
    "identity":     is_identity,
    "recolor":      is_recolor,
    "translate":    is_translate,
    "flip_h":       is_flip_h,
    "flip_v":       is_flip_v,
    "rot90":        is_rot90,
    "rot180":       is_rot180,
    "rot270":       is_rot270,
    "transpose":    is_transpose,
    "crop_to_bbox": is_crop_to_bbox,
    "tile_2x2":     is_tile_2x2,
    "tile_3x3":     is_tile_3x3,
    "symmetrize_h": is_symmetrize_h,
}


def classify_task(task: dict) -> dict:
    """Return per-check counts: how many train pairs each rule covers."""
    train_pairs = [(np.asarray(p["input"]), np.asarray(p["output"])) for p in task["train"]]
    n = len(train_pairs)
    out: dict = {"n_train": n, "checks": {}, "shape_changed_frac": 0.0}

    shape_changes = sum(1 for inp, out_g in train_pairs if inp.shape != out_g.shape)
    out["shape_changed_frac"] = shape_changes / n if n else 0.0

    for name, check in CHECKS.items():
        hits = 0
        for inp, gold in train_pairs:
            try:
                if check(inp, gold):
                    hits += 1
            except Exception:
                pass
        out["checks"][name] = hits

    # "Sufficient" = check passes ALL train pairs
    out["sufficient_rules"] = [
        name for name, hits in out["checks"].items() if hits == n
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", type=Path,
                    default=REPO_ROOT / "_research" / "arc-agi-2")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "receipts" / "arc-agi-2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    challenges = json.loads((args.research_dir / "arc-agi_evaluation_challenges.json").read_text())
    task_ids = sorted(challenges.keys())
    if args.limit:
        task_ids = task_ids[:args.limit]

    per_task = []
    sufficient_counter: Counter = Counter()
    any_sufficient = 0
    pair_check_counter: Counter = Counter()
    pair_total = 0

    for tid in task_ids:
        cls = classify_task(challenges[tid])
        cls["task_id"] = tid
        per_task.append(cls)
        for r in cls["sufficient_rules"]:
            sufficient_counter[r] += 1
        if cls["sufficient_rules"]:
            any_sufficient += 1
        for name, hits in cls["checks"].items():
            pair_check_counter[name] += hits
        pair_total += cls["n_train"]

    n = len(task_ids)
    coverage_currently = sufficient_counter["identity"] + sufficient_counter["recolor"] + sufficient_counter["translate"]
    coverage_with_geom = any_sufficient  # any check sufficient

    import time
    stamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    receipt_json = args.out_dir / f"coverage_analysis_{stamp}.json"
    receipt_md = args.out_dir / f"coverage_analysis_{stamp}.md"

    payload = {
        "kind": "arc2_coverage_analysis",
        "recorded_at_utc": stamp,
        "n_tasks": n,
        "tasks_with_at_least_one_sufficient_rule": any_sufficient,
        "current_3_rule_coverage_upper_bound": coverage_currently,
        "all_geometric_rules_coverage_upper_bound": coverage_with_geom,
        "sufficient_rule_counts": dict(sufficient_counter),
        "per_pair_hit_counts": dict(pair_check_counter),
        "n_train_pairs_total": pair_total,
    }
    receipt_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        f"# ARC-AGI-2 public eval — rule coverage analysis",
        "",
        f"- **Recorded:** {stamp}",
        f"- **Tasks analyzed:** {n}",
        f"- **Tasks with AT LEAST ONE sufficient rule (any geom check):** {any_sufficient} ({any_sufficient/n*100:.1f}%)",
        f"- **Tasks covered by current 3-rule grammar (Identity/Recolor/Translate):** {coverage_currently} ({coverage_currently/n*100:.1f}%)",
        "",
        f"## Sufficient-rule counts (rule covers ALL train pairs in N tasks)",
        "",
        f"| Rule | Tasks covered | % of {n} |",
        f"|---|---|---|",
    ]
    for name, count in sorted(sufficient_counter.items(), key=lambda kv: -kv[1]):
        md.append(f"| `{name}` | {count} | {count/n*100:.1f}% |")
    md.append("")
    md.append(f"## Per-train-pair hit counts (out of {pair_total} total pairs)")
    md.append("")
    md.append(f"| Rule | Pairs covered | % |")
    md.append(f"|---|---|---|")
    for name, count in sorted(pair_check_counter.items(), key=lambda kv: -kv[1]):
        md.append(f"| `{name}` | {count} | {count/pair_total*100:.1f}% |")

    receipt_md.write_text("\n".join(md), encoding="utf-8")
    print(f"coverage_analysis: {receipt_md}")
    print(f"  any_sufficient_rule: {any_sufficient}/{n} ({any_sufficient/n*100:.1f}%)")
    print(f"  current 3-rule coverage upper bound: {coverage_currently}/{n} ({coverage_currently/n*100:.1f}%)")
    print()
    print("Top rules by tasks covered:")
    for name, count in sorted(sufficient_counter.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:<14} {count:>3} tasks ({count/n*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
