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


@dataclass
class ReflectH:
    """Output = horizontal flip of input. Spelke GEOMETRY prior."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(np.fliplr(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return np.fliplr(np.asarray(grid)).copy()

    def signature(self) -> tuple:
        return ("ReflectH",)


@dataclass
class ReflectV:
    """Output = vertical flip of input. Spelke GEOMETRY prior."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape != out.shape:
                return False
            if not np.array_equal(np.flipud(inp), out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return np.flipud(np.asarray(grid)).copy()

    def signature(self) -> tuple:
        return ("ReflectV",)


@dataclass
class Transpose:
    """Output = transpose of input (diagonal reflection). Spelke GEOMETRY."""
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            if inp.shape[::-1] != out.shape:
                return False
            if not np.array_equal(inp.T, out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return np.asarray(grid).T.copy()

    def signature(self) -> tuple:
        return ("Transpose",)


@dataclass
class Rotate:
    """Output = 90/180/270° rotation of input. k = quarter-turns (1, 2, or 3).
    Spelke GEOMETRY prior (orientation symmetry group).
    """
    k: int = 1
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        candidates = {1, 2, 3}
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            local = set()
            for k in (1, 2, 3):
                rot = np.rot90(inp, k=k)
                if rot.shape == out.shape and np.array_equal(rot, out):
                    local.add(k)
            candidates &= local
            if not candidates:
                return False
        # Prefer smallest k (Occam).
        self.k = min(candidates)
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return np.rot90(np.asarray(grid), k=self.k).copy()

    def signature(self) -> tuple:
        return ("Rotate", self.k)


@dataclass
class CropToBbox:
    """Output = crop of input to the bounding box of non-background cells.
    Spelke OBJECTNESS prior (focus on the figure, not the ground).
    """
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            cropped = self._crop(inp)
            if cropped.shape != out.shape:
                return False
            if not np.array_equal(cropped, out):
                return False
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return self._crop(np.asarray(grid)).copy()

    @staticmethod
    def _crop(grid: Grid) -> Grid:
        bg = _background_color(grid)
        mask = grid != bg
        if not mask.any():
            return grid.copy()
        ys, xs = np.where(mask)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        return grid[r0:r1+1, c0:c1+1]

    def signature(self) -> tuple:
        return ("CropToBbox",)


@dataclass
class Tile:
    """Output = input tiled to a consistent (rows_factor, cols_factor) across
    every train pair. Spelke GEOMETRY (translation symmetry of tiling).
    """
    rf: int = 1
    cf: int = 1
    fitted: bool = False

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        if not train_pairs:
            return False
        rf_cf = None
        for inp, out in train_pairs:
            inp = np.asarray(inp); out = np.asarray(out)
            ih, iw = inp.shape; oh, ow = out.shape
            if oh % ih != 0 or ow % iw != 0:
                return False
            rf, cf = oh // ih, ow // iw
            if rf < 1 or cf < 1 or (rf == 1 and cf == 1):
                return False
            tiled = np.tile(inp, (rf, cf))
            if tiled.shape != out.shape or not np.array_equal(tiled, out):
                return False
            if rf_cf is None:
                rf_cf = (rf, cf)
            elif rf_cf != (rf, cf):
                return False
        if rf_cf is None:
            return False
        self.rf, self.cf = rf_cf
        self.fitted = True
        return True

    def predict(self, grid: Grid) -> Grid:
        return np.tile(np.asarray(grid), (self.rf, self.cf)).copy()

    def signature(self) -> tuple:
        return ("Tile", self.rf, self.cf)


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
    Each rule encodes a single Spelke prior (cohesion / geometry / topology / numerosity).
    """
    from .arc2_rules_v2 import all_v2_factories
    base = [
        lambda: Identity(),
        lambda: Recolor(),
        lambda: Translate2(),
        lambda: ReflectH(),
        lambda: ReflectV(),
        lambda: Transpose(),
        lambda: Rotate(k=1),
        lambda: Rotate(k=2),
        lambda: Rotate(k=3),
        lambda: CropToBbox(),
        lambda: Tile(),
    ]
    return base + all_v2_factories()


@dataclass
class Composed:
    """Depth-2 composed program: rule_a, then rule_b. Same fit/predict/
    signature interface as the base rules so the existing beam works.

    TIER-1 HONESTY: composition is a SEARCH STRATEGY over the existing
    base templates. No new priors introduced. The composed program is
    fitted by chaining two already-fitted base rules where the midstate
    after rule_a is the legitimate "intermediate train output" that
    rule_b must map to the final train output. If both base fits succeed,
    the composed program is the deterministic chain.
    """
    rule_a: object
    rule_b: object
    fitted: bool = False
    consistency_score: float = 0.0

    def fit(self, train_pairs: list[tuple[Grid, Grid]]) -> bool:
        # rule_a fits ONLY on its own — we do not require it to map all the
        # way to the output. It maps to a midstate. Then rule_b is fitted
        # against (midstate, output) pairs. If both succeed, the chain holds.
        if not train_pairs:
            return False
        try:
            # rule_a fits against the train pairs treating (input, midstate)
            # where midstate is unknown — we cannot fit rule_a "to nothing".
            # Instead, the search is: for every plausible rule_a parameterization,
            # produce midstates and check whether rule_b can fit (midstates, outputs).
            # For v1 we use the dumb-but-correct approach: try fitting rule_a as
            # if it were a one-shot rule (so its parameters are pinned by the
            # train pairs), then ALSO try fitting rule_b on (rule_a(inputs), outputs).
            # This catches the common cases:
            #   (Identity ∘ Recolor) == Recolor   — already covered by base
            #   (Recolor ∘ Translate2) — first recolor, then translate
            #   (Translate2 ∘ Recolor) — first translate, then recolor
            # which are the two genuinely-new depth-2 cases.
            # We also keep Identity as a sentinel so (Identity ∘ X) is allowed
            # (degenerates to X but the beam de-dupes by signature).
            #
            # The honest fit is:
            #   1. Try rule_a.fit(train_pairs) under the assumption that its
            #      output is the FINAL output (degenerate case — caller will
            #      dedupe by signature).
            #   2. If rule_a is Identity, midstates = inputs unchanged.
            #   3. Else if rule_a fits, produce midstates = rule_a.predict(inputs).
            #   4. Fit rule_b on (midstates, outputs) as a fresh train pair set.
            #   5. Accept only if rule_b fits ALL midstate→output pairs.
            if isinstance(self.rule_a, Identity):
                midstates = [np.asarray(inp).copy() for inp, _ in train_pairs]
            else:
                if not self.rule_a.fit(train_pairs):
                    return False
                midstates = []
                for inp, _ in train_pairs:
                    try:
                        midstates.append(self.rule_a.predict(np.asarray(inp)))
                    except Exception:
                        return False
            mid_pairs = list(zip(midstates, [out for _, out in train_pairs]))
            if not self.rule_b.fit(mid_pairs):
                return False
            # Final score = cell accuracy on (input, output) after full chain.
            self.fitted = True
            self.consistency_score = train_score(self, train_pairs)
            # Require an EXACT-fit composed program (every train pair matches
            # cell-perfectly) — Tier-1 honest, no partial-credit dressing.
            if self.consistency_score < 1.0 - 1e-9:
                self.fitted = False
                return False
            return True
        except Exception:
            return False

    def predict(self, grid: Grid) -> Grid:
        return self.rule_b.predict(self.rule_a.predict(np.asarray(grid)))

    def signature(self) -> tuple:
        return ("Composed", self.rule_a.signature(), self.rule_b.signature())


def _fit_all_rules(
    train_pairs: list[tuple[Grid, Grid]],
    compose_depth: int = 1,
) -> list[tuple[object, float]]:
    """Yield (rule, score) for every program that fits all train pairs.

    compose_depth=1 -> base rules only (legacy behavior).
    compose_depth=2 -> base rules + every depth-2 chain over base.
    """
    fitted: list[tuple[object, float]] = []
    base_rules: list[tuple[object, float]] = []
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
        base_rules.append((rule, score))
    fitted.extend(base_rules)

    if compose_depth >= 2:
        # Try every (factory_a, factory_b) ordered pair. We deliberately
        # try chains where rule_a does NOT solve alone — that's where
        # composition adds reach. Identity-as-rule_a is also allowed; the
        # beam dedupes by signature so it costs only the fit attempt.
        factories = _rule_factories()
        for fa in factories:
            for fb in factories:
                # Skip the trivial (Identity, Identity) chain.
                a_inst = fa(); b_inst = fb()
                if isinstance(a_inst, Identity) and isinstance(b_inst, Identity):
                    continue
                composed = Composed(rule_a=fa(), rule_b=fb())
                try:
                    if composed.fit(train_pairs):
                        fitted.append((composed, composed.consistency_score))
                except Exception:
                    continue
    return fitted


def solve_task(
    train_pairs: list[tuple],
    test_input,
    beam_width: int = 4,
    compose_depth: int = 1,
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

    fitted = _fit_all_rules(norm_pairs, compose_depth=compose_depth)
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
