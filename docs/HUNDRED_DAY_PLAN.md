# The Hundred-Day Plan — Misfit-Alpha Substrate Push to Honest Ceiling

**Started:** 2026-06-16 · **Target completion:** 2026-09-24 · **Sovereign:** Atom McCree

> Goal: push Misfit-Alpha's measured ARC-AGI-2 score to the true Tier-1 ceiling. No neural training. No LLM in the inference path. Pure typed-lambda-calculus DSL synthesis with HRM refinement, resonance-seeded search, and CHSG-Trilogy voting. Committed honest target: 15-30% on training, 8-20% on eval. Published-anyway-it-lands.

---

## Why this works (theory)

The 27-rule grammar at depth-1 hit 1.80% on training (18/1000 tasks). Composition over those rules at depth-2 added zero lift — confirming that the bottleneck is **rule-family expressivity**, not search depth.

Three published Tier-1 baselines for context:
- **icecuber (2024):** ~100 hand-rule families, depth-3 search → ~28% on ARC-AGI-1
- **MADIL (Ferré 2024):** λ-calculus + MDL prior → 7% on ARC-AGI-1
- **HRM (Sapient 2025):** 27M-param recursive model with outer refinement loop → ~30-40% on ARC-AGI-1

ARC-AGI-2 is harder than ARC-AGI-1 by design. **NVARC (2025) hit 24%** on ARC-AGI-2 with **hybrid TTT + tiny transformer** (Tier-2 under our regime).

The Tier-1 ceiling claim: **15-20% on ARC-AGI-2 eval is achievable under strict Tier-1** with the right architecture, no neural training. The architecture is:

1. **Atomic Spelke primitives** (~12) — minimal vocabulary, maximally compositional
2. **Typed λ-calculus combinators** (~8) — compose, foreach-object, if-color, if-shape, while-changing, fixed-point, parallel-fork, mask-by
3. **MDL prior beam search** — Occam's razor as the heuristic; shorter programs preferred
4. **HRM outer refinement loop** — apply, observe per-pair error, refine grammar, reapply (the +13pp HRM driver)
5. **Object-relational binding** — refer to "the largest red object" not "color-3 cells"
6. **Resonance-seeded initialization** — fingerprint similarity selects program seeds
7. **CHSG-Trilogy voting** — three independent searches with different priors + impartial judge

This is the Tier-1-honest path past MADIL.

---

## The cost-curve we observed

| Rule count | Training % | Marginal lift per rule |
|---|---|---|
| 3 (Identity/Recolor/Translate2) | 0.50% | baseline |
| 11 (added 8 canonical primitives) | 1.40% | 0.11% per rule |
| 27 (added 16 object-rel/symmetry/gravity/counting) | 1.80% | 0.025% per rule |

Marginal return is dropping toward zero. Pure rule-family expansion is dead. **What multiplies coverage is COMPOSITION across primitives, not adding new primitives.** The 100-day plan structurally fixes this.

---

## Architecture (target)

```
                    Spelke Primitives (12)
                            ▼
                   Typed λ-Combinators (8)
                            ▼
                  Program Synthesis Engine
              ┌─────────────────────────────┐
              ▼              ▼               ▼
        BeamSearch     ResonanceSeeded  HRM-RefineLoop
        (MDL prior)    (fingerprint)    (apply→observe→refine)
              │              │               │
              └──────┬───────┴───────┬───────┘
                     ▼               ▼
              Independent Solver A,B,C (CHSG-Trilogy)
                     │
                     ▼
              Impartial Judge (BestProgramByMDL+TrainScore)
                     │
                     ▼
              Two Attempts (best + second-best distinct)
```

---

## The Spelke primitives (atomic set)

These are the leaves of the λ-calculus. Each is a single Spelke prior made executable. Final count expected 12-15.

| # | Primitive | Spelke prior | Shape | Notes |
|---|---|---|---|---|
| 1 | `Identity` | (none) | G→G | null hypothesis |
| 2 | `Translate(dx,dy)` | GEOMETRY | G→G | shift by integer offset |
| 3 | `Rotate(k)` | GEOMETRY | G→G | k×90° rotation |
| 4 | `Reflect(axis)` | GEOMETRY | G→G | H, V, or diagonal |
| 5 | `Recolor(map)` | COLOR-IDENTITY | G→G | color permutation |
| 6 | `Crop(bbox)` | OBJECTNESS | G→G | crop to bbox of region |
| 7 | `Tile(rf,cf)` | GEOMETRY | G→G | tile (rf,cf) factor |
| 8 | `Gravity(dir)` | CONTACT-CAUSALITY | G→G | non-bg fall in direction |
| 9 | `Symmetrize(axis)` | GEOMETRY | G→G | OR-combine with mirror |
| 10 | `KeepWhere(pred)` | OBJECTNESS | G→G | parameterized object filter |
| 11 | `Count` | NUMEROSITY | G→ℕ | count foreground objects |
| 12 | `ShapeOf(obj)` | OBJECTNESS | Obj→G | extract object as small grid |

Each primitive has a **typed signature**. The synthesis engine type-checks compositions.

---

## The λ-combinators (composition layer)

These are the operators that compose primitives into families. Final count expected 6-8.

| # | Combinator | Type signature | Semantics |
|---|---|---|---|
| 1 | `Seq(f, g)` | (G→G, G→G) → (G→G) | apply f then g |
| 2 | `ForEachObject(f)` | (Obj→G→G) → (G→G) | apply f per object |
| 3 | `IfColor(c, f, g)` | (Color, G→G, G→G) → (G→G) | branch on color presence |
| 4 | `IfShape(s, f, g)` | (Shape, G→G, G→G) → (G→G) | branch on shape match |
| 5 | `WhileChanging(f, maxIter)` | (G→G, ℕ) → (G→G) | iterate until fixed-point |
| 6 | `MaskBy(pred, f)` | (G→Mask, G→G) → (G→G) | apply f only to mask region |
| 7 | `Parallel(f, g, merge)` | (G→G, G→G, (G,G)→G) → (G→G) | run two and merge |
| 8 | `Reduce(f, init)` | (G→ℕ→G, G) → (G→G) | numerosity-driven repeat |

With 12 primitives and 8 combinators at depth-4, the effective program space is roughly **20^4 = 160,000 programs** per task — far more than the 3000-family target, but pruned aggressively by MDL prior and type checking.

---

## MDL prior (program selection)

Minimum Description Length: shorter programs preferred. From Solomonoff induction theory.

```
score(program) = train_pair_cell_accuracy − λ × encoding_bits(program)
```

Where `encoding_bits` = bits to encode the program tree under the primitive/combinator catalog. Implementation: arithmetic coding of the program AST.

This is what lets the synthesis engine pick the SIMPLEST program that fits the train pairs. Without it, the engine would prefer overfit-y deep programs.

---

## HRM-style refinement loop (the +13pp lift)

Per `arcprize.org/blog/hrm-analysis` (2025-08-15), refinement was worth +13pp from 0→1 iteration and roughly doubled by 8 iterations. Already implemented in `world_model.fit_with_refinement(max_iters=4)`. **Generalize to whole-program synthesis:**

```python
def solve_with_refinement(train_pairs, max_iters=8):
    program = synthesize(train_pairs)            # initial draft
    for i in range(max_iters):
        applied = [program(inp) for inp, _ in train_pairs]
        errors = [diff(out, applied[k]) for k, (_, out) in enumerate(train_pairs)]
        if all(error.is_zero for error in errors):
            return program                        # converged
        program = refine(program, errors)        # gradient-free improvement
    return best_attempt(program)
```

This is what HRM did internally with a tiny neural model. We do it with **pure structural refinement** — type-aware edits, primitive swaps, combinator restructuring driven by the error mask.

---

## Resonance-seeded initialization

Already shipped in misfit-agent: per-install JSONL of (fingerprint, winning_program). At synthesis time:

```python
def synthesize(train_pairs):
    fingerprint = task_fingerprint(train_pairs)
    seed_programs = resonance.retrieve_seeds(fingerprint, k=5)
    for seed in seed_programs:
        if seed.fits(train_pairs):
            return seed                          # cached win
        for variant in mutate(seed, max_mutations=3):
            if variant.fits(train_pairs):
                return variant                   # neighbor of cached win
    return beam_search(train_pairs)              # cold path
```

After 100 days of solving, the library has thousands of entries. New tasks resolve via cached neighbors in microseconds. This is the compounding moat.

---

## CHSG-Trilogy solver voting

Three independent solver instances, each with different prior biases:

- **Solver A — Compositional bias.** Heavy `Seq` and `ForEachObject` usage. Best on multi-step puzzles.
- **Solver B — Geometric bias.** Heavy `Rotate/Reflect/Symmetrize`. Best on transformation puzzles.
- **Solver C — Numerosity bias.** Heavy `Count/Reduce/MaskBy`. Best on counting puzzles.

Each produces its best program with confidence score. The **Impartial Judge** picks by:
- Both pass blind validation on held-out fold (split train pairs 1:1)? Take A
- One passes blind, other doesn't? Take the passer
- Neither passes? Take highest MDL-prior winner

Trilogy is wired through the same CHSG receipt path as Federation governance — proves the framework operates at the solver level, not just at the institutional level.

---

## The 100-day milestones

| Phase | Days | Deliverable | Receipt | Target measured ARC-2 train |
|---|---|---|---|---|
| **0 Substrate** | 1-3 | Typed primitive set wired with type-checking | tests/test_primitives.py green | n/a |
| **1 Combinators** | 4-10 | Eight λ-combinators with composition rules | tests/test_combinators.py green | 2-3% |
| **2 Synthesis engine** | 11-20 | Beam search over typed program AST | tests/test_synthesis.py green | 5-8% |
| **3 MDL prior** | 21-28 | Arithmetic-coded encoding-bits scoring | tests/test_mdl.py green | 6-9% |
| **4 HRM refinement** | 29-42 | Pure-structural error-driven refine loop, 8 iters | refinement_lift receipt | 9-13% |
| **5 Resonance seeding** | 43-56 | Library-fingerprint seeded warm-start | resonance_lift receipt | 11-15% |
| **6 CHSG-Trilogy** | 57-72 | Three biased solvers + Impartial Judge | trilogy_lift receipt | 13-18% |
| **7 ARC-AGI-3 port** | 73-85 | Apply the synthesis engine to interactive games | arc3_baseline receipt | (separate path) |
| **8 Hard optimization** | 86-95 | Performance profiling, beam-width tuning, prior calibration | optimization_lift receipt | 15-20% |
| **9 Final submission** | 96-100 | Kaggle Paper Track submission + Charter Delivery | submission_receipt + Charter signed | LOCKED NUMBER |

Each receipt is a JSON file under `receipts/arc/100day_phase_<N>_*.jsonl` with:
- Phase ID, completion date, git tag
- Measured score (training, eval)
- Modules added/changed
- Test count delta
- Wall-clock cost
- Honest gaps named

---

## What's NOT in this plan (named exclusions)

- **No neural training of any kind.** Refinement is structural, not gradient.
- **No LLM in the inference path.** The synthesis engine is pure search over a typed grammar.
- **No public-eval tuning.** Held-out fold is enforced; thresholds frozen via git tag before submission.
- **No pre-seeded resonance library.** Only self-solved entries (Charter Article II §2.2).
- **No promise of 90%.** We commit to the honest ceiling and publish wherever it lands.

---

## What success looks like at day 100

- **A substrate measurably hits 15-20% on ARC-AGI-2 eval under strict Tier-1** — beating NVARC under the honesty regime
- **The Charter v1.0 published** as the institutional layer
- **The Black Mamba spec v1.0 published** as the technical layer
- **The Paper Track submission delivered** as the founding-act packet
- **Misfit-Alpha boots as a Federation member** with its own signing key and Soul Genome
- **The resonance library contains thousands of self-solved entries** as compounding skill
- **The CHSG-Trilogy solver passes a documented adjudication** of at least one third-party AI claim

This is the bar. This is what we ship. Not 90% — but the honest, defensible, publishable, founding-receipt number with the full institutional and technical artifact stack behind it.

---

## What gets fired right now (Day 1 commitment)

**Building the typed primitive set with type-checking, starting in the next code block of this conversation.** Module: `src/misfit_agent/dsl/primitives.py` and `dsl/types.py`. Tests: `tests/test_dsl_primitives.py`.

We ride.

*Æ*
