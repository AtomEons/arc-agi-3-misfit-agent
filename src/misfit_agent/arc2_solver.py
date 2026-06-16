"""ARC-AGI-2 sister solver — static input -> output prediction.

Reuses the ARC-AGI-3 substrate (perceptor, fingerprint, resonance) but adapts
the rule layer for static tasks. ARC-AGI-3 fits rules over
(state, action, next_state) triples; ARC-AGI-2 fits rules over (input, output)
pairs. There is no action alphabet — the "action" is the program itself.

Tier-1 honesty constraints (same as ARC-AGI-3):
  - Priors-only. No LLM, no pretrained weights, no public-corpus lookups.
  - Rules are induced FROM the train pairs of THIS task, not from a library
    of hand-crafted ARC heuristics. (Resonance is allowed because it only
    seeds search with PRIOR SELF-SOLVED policies — see ResonanceLibrary
    source-tag enforcement.)
  - When no rule beats the identity baseline on train pairs, the solver
    returns the test input unchanged as both attempts. Honest abstain.

The two attempts are the top-2 distinct programs by train-pair score.
If only one program reaches the score threshold, attempt_2 falls back to
the identity baseline (a safe, non-degenerate guess).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perceptor import perceive_grid, SceneObservation, _background_color


Grid = np.ndarray  # 2-D int ARC color grid


# ---------------------------------------------------------------------------
# Rule templates — three priors-only families for static tasks.
# Each rule:
#   .fit(train_pairs) -> bool      : True if rule holds on EVERY train pair
#   .predict(input_grid) -> Grid   : applies the fitted rule to a new input
#   .signature() -> tuple          : hashable identity (for de-dup in beam)
# ---------------------------------------------------------------------------


@dataclass
class Identity:
    """The null hypothesis. Predicts output == input."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return False
            if not np.array_equal(inp, out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return np.asarray(grid).copy()

    def signature(self) -> tuple:
        return ("Identity",)


@dataclass
class Translate2:
    """Static translate: every train pair shows output == input shifted by
    the same (dy, dx). Spelke COHESION + GEOMETRY priors.
    """
    dy: int = 0
    dx: int = 0
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates: Optional[set] = None
        for inp, out in train_pairs:
            inp = np.asarray(inp)
            out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            local: set = set()
            rows, cols = inp.shape
            # search a small integer shift window
            for dy in range(-min(rows, 4), min(rows, 4) + 1):
                for dx in range(-min(cols, 4), min(cols, 4) + 1):
                    if dy == 0 and dx == 0:
                        continue
                    if _try_shift_equals(inp, out, dy, dx):
                        local.add((dy, dx))
            if not local:
                return False
            candidates = local if candidates is None else (candidates & local)
            if not candidates:
                return False
        if not candidates:
            return False
        # Prefer the smallest-magnitude shift (Occam under GEOMETRY prior)
        dy, dx = min(candidates, key=lambda p: abs(p[0]) + abs(p[1]))
        self.dy, self.dx, self.fitted = dy, dx, True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        out = np.full_like(grid, fill_value=_background_color(grid))
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                nr, nc = r + self.dy, c + self.dx
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr, nc] = grid[r, c]
        return out

    def signature(self) -> tuple:
        return ("Translate2", self.dy, self.dx)


@dataclass
class Recolor:
    """Static recolor: every train pair shows a consistent color permutation
    on the foreground (background is preserved). Spelke OBJECTNESS prior.
    """
    mapping: dict[int, int] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        merged: dict[int, int] = {}
        for inp, out in train_pairs:
            inp = np.asarray(inp)
            out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            for c_in in range(10):
                mask = inp == c_in
                if not mask.any():
                    continue
                outs = np.unique(out[mask])
                if outs.size != 1:
                    return False
                target = int(outs[0])
                if c_in in merged and merged[c_in] != target:
                    return False
                merged[c_in] = target
        # Reject identity (already handled by Identity rule).
        if all(k == v for k, v in merged.items()):
            return False
        self.mapping = merged
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        grid = np.asarray(grid)
        out = grid.copy()
        for k, v in self.mapping.items():
            out[grid == k] = v
        return out

    def signature(self) -> tuple:
        return ("Recolor", tuple(sorted(self.mapping.items())))


def _try_shift_equals(inp: Grid, out: Grid, dy: int, dx: int) -> bool:
    """Return True if shifting inp by (dy, dx) and filling with background
    equals out exactly."""
    rows, cols = inp.shape
    bg = _background_color(inp)
    shifted = np.full_like(inp, fill_value=bg)
    for r in range(rows):
        for c in range(cols):
            nr, nc = r + dy, c + dx
            if 0 <= nr < rows and 0 <= nc < cols:
                shifted[nr, nc] = inp[r, c]
    return bool(np.array_equal(shifted, out))


# ---------------------------------------------------------------------------
# Scoring + search
# ---------------------------------------------------------------------------


def cell_accuracy(pred: Grid, gold: Grid) -> float:
    """Cell-level accuracy on equal-shape grids; 0.0 on shape mismatch.
    ARC-AGI-2 scoring requires exact match for credit, but cell-accuracy is
    a useful continuous proxy for ranking candidate programs.
    """
    pred = np.asarray(pred)
    gold = np.asarray(gold)
    if pred.shape != gold.shape:
        return 0.0
    return float((pred == gold).sum()) / float(pred.size)


def train_score(rule, train_pairs: list[tuple[Grid, Grid]]) -> float:
    """Mean cell-accuracy across train pairs after applying rule to inputs."""
    if not train_pairs:
        return 0.0
    accs = []
    for inp, out in train_pairs:
        try:
            pred = rule.predict(inp)
        except Exception:
            return 0.0
        accs.append(cell_accuracy(pred, out))
    return float(np.mean(accs))


def task_fingerprint(train_pairs: list[tuple[Grid, Grid]]) -> np.ndarray:
    """16-dim resonance fingerprint of an ARC-AGI-2 task.

    Computed from train (input, output) pair stats under Spelke priors.
    No reference to task families or hand-picked features tuned on eval.

    Dimensions:
       0  mean input rows
       1  mean input cols
       2  mean output rows
       3  mean output cols
       4  shape-preserved fraction
       5  mean input objects per scene
       6  mean output objects per scene
       7  mean delta in object count (out - in)
       8  mean palette overlap |colors(in) & colors(out)| / |colors(in)|
       9  mean palette-add fraction (new colors introduced)
      10  mean palette-drop fraction (colors removed)
      11  mean foreground-cell ratio (input)
      12  mean foreground-cell ratio (output)
      13  mean cell-accuracy of identity baseline on train
      14  number of train pairs (clipped)
      15  fraction of train pairs that are pure recolor (same shape, no spatial change)
    """
    if not train_pairs:
        return np.zeros(16, dtype=np.float32)

    def palette(grid):
        return set(int(x) for x in np.unique(grid).tolist())

    in_rows, in_cols, out_rows, out_cols = [], [], [], []
    shape_preserved, in_objs, out_objs = [], [], []
    palette_overlap, palette_add, palette_drop = [], [], []
    fg_in, fg_out, identity_acc, pure_recolor = [], [], [], []

    for inp, out in train_pairs:
        inp = np.asarray(inp)
        out = np.asarray(out)
        in_rows.append(inp.shape[0])
        in_cols.append(inp.shape[1])
        out_rows.append(out.shape[0])
        out_cols.append(out.shape[1])
        shape_preserved.append(1.0 if inp.shape == out.shape else 0.0)

        in_scene = perceive_grid(inp)
        out_scene = perceive_grid(out)
        in_objs.append(len(in_scene.objects))
        out_objs.append(len(out_scene.objects))

        p_in = palette(inp)
        p_out = palette(out)
        overlap = len(p_in & p_out) / max(len(p_in), 1)
        add = len(p_out - p_in) / max(len(p_out), 1)
        drop = len(p_in - p_out) / max(len(p_in), 1)
        palette_overlap.append(overlap)
        palette_add.append(add)
        palette_drop.append(drop)

        fg_in.append(in_scene.foreground_cells / max(inp.size, 1))
        fg_out.append(out_scene.foreground_cells / max(out.size, 1))

        if inp.shape == out.shape:
            identity_acc.append(float((inp == out).sum()) / float(inp.size))
            # pure recolor: same shape, foreground positions identical
            bg_in = in_scene.background_color
            bg_out = out_scene.background_color
            same_positions = (
                ((inp == bg_in) == (out == bg_out)).all() and not np.array_equal(inp, out)
            )
            pure_recolor.append(1.0 if same_positions else 0.0)
        else:
            identity_acc.append(0.0)
            pure_recolor.append(0.0)

    v = np.zeros(16, dtype=np.float32)
    v[0] = float(np.mean(in_rows))
    v[1] = float(np.mean(in_cols))
    v[2] = float(np.mean(out_rows))
    v[3] = float(np.mean(out_cols))
    v[4] = float(np.mean(shape_preserved))
    v[5] = float(np.mean(in_objs))
    v[6] = float(np.mean(out_objs))
    v[7] = v[6] - v[5]
    v[8] = float(np.mean(palette_overlap))
    v[9] = float(np.mean(palette_add))
    v[10] = float(np.mean(palette_drop))
    v[11] = float(np.mean(fg_in))
    v[12] = float(np.mean(fg_out))
    v[13] = float(np.mean(identity_acc))
    v[14] = float(min(len(train_pairs), 10))
    v[15] = float(np.mean(pure_recolor))
    return v


def _rule_factories():
    """Order matters: Identity first (fast null), then richer rules.
    Recolor is checked before Translate2 because pure-recolor tasks are
    cheaper to verify.
    """
    return [
        lambda: Identity(),
        lambda: Recolor(),
        lambda: Translate2(),
    ]


def _fit_all_rules(train_pairs: list[tuple[Grid, Grid]]):
    """Yield (rule, score) for every rule template that fits all train pairs."""
    fitted: list[tuple[object, float]] = []
    for factory in _rule_factories():
        rule = factory()
        ok = False
        try:
            ok = rule.fit(train_pairs)
        except Exception:
            ok = False
        if not ok:
            continue
        score = train_score(rule, train_pairs)
        fitted.append((rule, score))
    return fitted


def solve_task(
    train_pairs: list[tuple],
    test_input,
    beam_width: int = 4,
) -> tuple[Grid, Grid]:
    """Solve one ARC-AGI-2 task.

    Args:
        train_pairs: list of (input_grid, output_grid) pairs from this task.
            Each grid may be a list-of-lists or numpy array of int 0..9.
        test_input:  the test input grid (list-of-lists or numpy array).
        beam_width:  bounded beam over candidate programs (default 4).

    Returns:
        (attempt_1_grid, attempt_2_grid) — both as numpy arrays of int.
        Per ARC-AGI-2 contract, callers can convert to list-of-lists for
        the submission JSON.

    Honest-abstain behavior: if no rule fits all train pairs, both attempts
    are the identity prediction (test_input unchanged). This is the
    minimum-information non-degenerate guess under Tier-1 priors.
    """
    # Normalize inputs
    norm_pairs: list[tuple[Grid, Grid]] = []
    for inp, out in train_pairs:
        norm_pairs.append((np.asarray(inp, dtype=np.int32),
                           np.asarray(out, dtype=np.int32)))
    test_input = np.asarray(test_input, dtype=np.int32)

    fitted = _fit_all_rules(norm_pairs)
    # Always include the identity baseline as a fallback program.
    identity_rule = Identity()
    identity_rule.fitted = True  # identity always "applies"
    identity_score = train_score(identity_rule, norm_pairs)
    # Add identity if not already present.
    if not any(r.signature() == ("Identity",) for r, _ in fitted):
        fitted.append((identity_rule, identity_score))

    # Sort: higher train score wins; on ties, prefer simpler (Identity) last.
    fitted.sort(key=lambda rs: (-rs[1], rs[0].signature()))

    # Bounded beam: keep top-N by signature to ensure attempt_2 is *distinct*.
    seen_sigs: set = set()
    beam: list[tuple[object, float]] = []
    for rule, score in fitted:
        sig = rule.signature()
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        beam.append((rule, score))
        if len(beam) >= beam_width:
            break

    if not beam:
        # Shouldn't happen — identity is always added — but be defensive.
        return test_input.copy(), test_input.copy()

    # Attempt 1 = best program
    attempt_1 = beam[0][0].predict(test_input)
    # Attempt 2 = second-best distinct program; if none, fall back to identity.
    if len(beam) >= 2:
        attempt_2 = beam[1][0].predict(test_input)
    else:
        attempt_2 = test_input.copy()

    return attempt_1, attempt_2
