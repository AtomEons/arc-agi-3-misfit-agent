# LakeStrike v4 — Pre-Submit Adversarial Review

**Target:** Misfit substrate v4 (post-Wave-4 integration), ARC-AGI-2 sister solver, v4 notebook.
**Reviewer:** LakeStrike team — Goose (correctness), Iceman (scope), Viper (innovation), Hangman (skeptic), synthesized by Maverick.
**Baseline:** `git log -1 7bf04eb`, `pytest -q` → 93/93 green (brief reported 79; current count is 93 — better than expected).
**Honesty register:** Every finding below is grounded in a file:line citation and, where relevant, a reproducer script run against the live tree. No theater.

---

## TL;DR podium-odds verdict

> **Top-5 podium odds for Milestone #1 (2026-06-30) if we ship as-is: 0.18.**

Reasoning:
- 93/93 tests is real and the substrate is honest.
- BUT three Goose-grade Blockers exist (one of them — `select_action` mutating the GameAction enum singleton — is the exact Lane-A failure mode MCTS was designed to fix; it survived integration in the priors-fallback path which is what runs MOST of the early game).
- Refinement-loop is currently theater — it does not gain HRM's +13pp because `fit()` wipes rules at the top of every iteration, making `_prune_contradicting_rules` informationally inert across iterations.
- Fingerprint scale mismatch silently disables the abstain plateau check.

Fix the three Goose Blockers and one Hangman Must-Fix below and the podium odds rise to ~0.42 (median estimate; we'd be in the credible top-tier band but private games are still the great unknown).

---

## 1. GOOSE — Correctness

Severity legend: **B = Blocker** (must fix before submit) · **MF = Must-Fix** · **SF = Should-Fix** · **N = Nice**

### G1 [B] — `select_action` mutates GameAction enum singleton

**Where:** `src/misfit_agent/action_search.py:155`
```python
if hasattr(action, "is_complex") and action.is_complex():
    cand = best_click_candidate(scene, policy_seeds_xy=seed_xy_hints or None)
    data = {"x": cand.x, "y": cand.y}
    action.set_data(data)            # ← MUTATES THE ENUM SINGLETON
```

This is the **exact** Lane-A failure mode the MCTS module documents as having structurally fixed via `ActionHandle` (see `mcts_puct.py:45-61`). The fix was applied INSIDE MCTS only. The priors-fallback path (`select_action`) — used on every step below the 0.30 coverage gate, which is most of the early game — still mutates `GameAction.ACTION6` in place.

**Reproducer (verified live):**
```python
a1 = select_action(scene, t, [A6], [], 100, None)
# ... engine processes step ...
A6.data = {"x": 999, "y": 999}   # imagine engine clears it
a2 = select_action(scene, t, [A6], [], 100, None)
assert a1 is a2  # PASSES — they are the same Python object
assert a1.data == a2.data  # PASSES — a1 sees a2's coordinates
```

**Impact:** Any caller that caches the returned `GameAction` across a step boundary observes the next call's click coordinates. The bug is asymptomatic in pure-call-and-submit usage but POISONS:
- Any future logging that re-reads `action.data` after the next `choose_action`.
- The `Misfit.tracker.record_action(... data ...)` call on line 164 already `dict(data)`-copies, so the tracker is safe — but `action.reasoning` on line 156 captures a string of `cand.x, cand.y`, which is OK. The fragility is in the assumption that the singleton's `.data` is the current step's data. That assumption breaks the moment we add ANY between-step inspection (planned for the v4 notebook's introspection cells).
- Lane-A test (`test_action6_mutation_safety.py`) covers MCTS but NOT `select_action`. The gap is undocumented.

**Fix:** Mirror the MCTS contract. Build a fresh `GameAction.ACTION6` clone-or-handle, set data on it exclusively, and return that. Or — minimum-surface change — change line 155 to `action = copy.copy(action); action.set_data(dict(data))`. Add a regression test using the same FakeAction pattern as `test_mcts_puct.py`.

### G2 [B] — Outer refinement loop has no algorithmic basis to gain HRM's +13pp

**Where:** `src/misfit_agent/world_model.py:82-119` (`fit_with_refinement`)

**The defect:**
```python
def fit(self, observations):
    ...
    self.rules = new_rules                # ← REPLACES wholesale
    return scores

def fit_with_refinement(self, observations, max_iters=4, ...):
    for i in range(max_iters):
        scores = self.fit(observations)               # rebuilds rules
        ...
        self._prune_contradicting_rules(observations) # drops some
        # Next iteration: self.fit() rebuilds rules again, ignoring prune output.
```

`fit()` does `self.rules = new_rules` unconditionally. The "feedback signal" produced by `_prune_contradicting_rules` is wiped by the next iteration's `fit()`. HRM gains +13pp from refinement because each pass conditions on the prior pass's high-confidence outputs. Our implementation does not — it just runs `fit` N times against the same observations, producing the same rules each time. The early-stop after iter 1 (when score doesn't improve) is masking this: the test `test_fit_with_refinement_early_stops_when_no_improvement` passes because there is nothing to improve.

**Reproducer (verified live):**
```
fit_with_refinement(obs, max_iters=3) → iterations: 2, scores: [1.0, 1.0]
Second call to fit_with_refinement(same obs) → iterations: 4, rules identical.
```

**Impact:** The PAPER_DRAFT.md claim ("HRM outer-loop refinement +13pp") is **unsupported**. The `refinement_iterations_total` counter is real, but the refinement does no work beyond fit. Notebook claims about this are honesty risks.

**Fix:** Two options. (a) Honest naming + paper retraction: rename to `fit_then_prune` and adjust PAPER_DRAFT claims. (b) Real refinement: keep surviving pruned rules as a "trust set" carried into the next iteration's fit (e.g. `fit_observations_excluding_contradicted_classes()` or add `seed_rules=` parameter to `fit`). Option (b) is the right work; option (a) is the safe pre-submit move. Either is acceptable; doing neither is dishonest.

### G3 [B] — Fingerprint scale assumption violated → abstain plateau check is silently always-false

**Where:**
- `src/misfit_agent/abstain_policy.py:75-76` (claim): *"The scale is dimensionless because fingerprints are L2-normalized; 0.01 is one percent of the unit ball."*
- `src/misfit_agent/fingerprint.py:38-94` (reality): no normalization step anywhere.

**Reproducer (verified live):**
```
50-step episode, identical scene → fingerprint L2 norm = 4.94; max |component| = 3.93;
dim 9 alone (log(1+scenes)) = 3.93.
```

The 0.01 plateau threshold is calibrated for unit-norm vectors. With raw vectors whose dim-9 alone grows as `log(1+N)`, every step inflates the vector. Cosine similarity in resonance is unaffected (cosine normalizes internally), but plateau detection in `AbstainPolicy._novelty_plateau()` uses raw L2 distance between successive fingerprints — that distance picks up the `log(N+1)-log(N)` drift from dim 9 alone, which decays to 0.01 only around N≈100 steps. Worse, dim 2 (mean object count) and dim 7 (largest-area-ratio) drift bounded by their underlying signals.

**Impact:** AbstainPolicy.should_abstain has THREE conjuncts. Plateau being silently false means we never abstain. False-negative abstain is asymmetric: we burn the whole action budget on hopeless games instead of preserving it for the next level. With Milestone #1's many private games, this is plausibly worth 5-15% of total score.

**Fix:** Pick one and document it.
- (a) Normalize fingerprint before storage: `v / max(np.linalg.norm(v), 1e-6)`.
- (b) Drop the docstring lie and re-derive `plateau_delta_threshold` from observed scale (≈0.05 of typical vector norm).
- (c) Switch plateau check to cosine distance, not L2.

We recommend (a) + (c): unit-norm fingerprint AND cosine-based plateau. Cosine is more robust to dim-9's monotone drift.

### G4 [MF] — `GoalInducer.hypothesize` tiebreak silently demotes the most-interpretable hypothesis

**Where:** `src/misfit_agent/goal_inducer.py:280`
```python
hyps.sort(key=lambda h: (-h.score, -h.support, h.contradictions, h.kind, h.params))
```

`_candidate_count_equals` (line 119-123) seeds N=0 candidates from class disappearance. So for any class C that disappears whenever a level advances, BOTH `removed_all_of_class(C)` and `count_of_class_equals_N(C, 0)` end up with identical (support, contradictions, posterior). The tiebreak then sorts alphabetically by `h.kind`: `"count_of_class_equals_N" < "removed_all_of_class"` lexicographically, so the less-interpretable count form wins.

**Impact:** `Misfit._top_hypothesis_tag` tags the resonance-library row with `count_of_class_equals_N(2,0)` instead of `removed_all_of_class(2)`. Both are valid for retrieval, but the tag is what humans read in audits. Quality bug, not correctness disaster. Fix by listing `removed_all_of_class` first in the tiebreaker (it's the more specific predicate) or by suppressing the (cls, 0) count candidate when `removed_all_of_class(cls)` is also enumerable.

### G5 [SF] — `Misfit.tracker_hungarian` is dead code

**Where:** `src/misfit_agent/misfit_agent.py:100` instantiates `self.tracker_hungarian = HungarianTracker()`, but `_maybe_refit_world_model` calls `observe_hungarian()` which **constructs a NEW** HungarianTracker each call (see `episode.py:137`). The instance attribute is never read. Tracker is documented stateless so there is no behavior difference, but it's misleading dead code — readers will assume per-step persistence. Either delete the attribute or pass it into `observe_hungarian(...tracker=self.tracker_hungarian)` for documentation honesty.

### G6 [SF] — MCTS `_expand` priority of progress-path seeded actions

**Where:** `src/misfit_agent/mcts_puct.py:326-330`

`P(a|s) = 1.0 if name in progress_path else 0.5`. This means **every** progress-path action gets the same 1.0 prior regardless of where it appears in the path. With a progress path like `["A1", "A2", "A6"]` and root depth 0, both `A1` (rightly first) and `A2` get prior 1.0. The right behavior is `P = 1.0 if path[step_index] == name else 0.5`. The current implementation overestimates step-2 and step-3 actions at root.

Impact: not fatal — MCTS still converges via Q-values — but a measurable bias toward "any action that ever appeared in a winning prior policy" rather than "the action that worked at THIS step". On 200 rollouts the bias matters.

### G7 [N] — `MCTSPUCT._enumerate_handles` reuses root_scene for child-node click candidates

**Where:** `mcts_puct.py:455-461`. Acknowledged in the comment. The Tier-1 reasoning is correct (we don't invent positions). The pragmatic risk is that progressive widening at deeper nodes always sees root-scene objectness, so as depth grows we expand the same click candidates we already considered. Not a blocker — but noting it because progressive widening's purpose is to broaden DEEPER, and reusing root candidates partially defeats that.

---

## 2. ICEMAN — Scope (Tier-1 attestation)

### I1 [PASS] — No LLM smuggled in
`grep -E "torch|transformers|openai|anthropic|llama|huggingface|langchain|sentence_transformers"` over `src/` returns clean. `test_tier1_attestation.py` enforces this in CI. **Green.**

### I2 [PASS] — No pretrained heuristic dressed as Spelke prior
All `data:` rationale strings in `click_quantizer.py` reference object centroids, bboxes, edge midpoints, quadrant fallback — pure geometry. `goal_inducer.py` predicate families (`removed_all_of_class`, `agent_reached_class`, `count_of_class_equals_N`) are domain-general Spelke primitives. **Green.**

### I3 [PASS] — No threshold tuned on the 25 public games during integration
`config.py` thresholds carry provenance tags (a/b/c). Sweeping any value re-classifies it as (c). I cross-referenced git log on `config.py` since wave 30 — no value changes beyond the documented MCTS settings. `min_actions_before_abstain=25` is grandfathered as a (b) heuristic with a TODO to re-derive. **Green, with the same TODO carrying forward.**

### I4 [SF] — Refinement loop honesty
Per G2, the `fit_with_refinement` claims (HRM +13pp) in PAPER_DRAFT.md and `world_model.py` docstring overstate what the code actually does. Scope-wise, this is an Iceman ding too: claiming a +13pp gain we cannot defend in private games is the kind of thing that haunts a podium-finish writeup. **Must align claims with code before submission.**

### I5 [PASS] — Arc2 solver scope
`arc2_solver.py` rule families: `Identity`, `Translate2`, `Recolor`. All admissible Spelke priors. Honest-abstain on no-fit. No public-corpus inspection. **Green.**

---

## 3. VIPER — Innovation (high-value additions)

### V1 [Recommended] — ACTION7 (undo) as free state-rollback for MCTS branch exploration

**Currently:** `ACTION7` is referenced in `fingerprint.py` (slot index) and `mcts_puct.py:69` as a simple action, but the planner treats it as just another action choice — no semantic understanding of "this UNDOES the previous step".

**Proposal:** When MCTS expands a node whose `incoming` action is followed by `ACTION7`, treat that as a NO-OP and prune the sub-tree. More valuably: when running deep rollouts, ACTION7 lets the planner "back up and try a different branch" without paying a real environment step — the agent already has the prior grid in `_Node.grid_fingerprint`. This effectively doubles search depth for free under the (h/N)^2 scoring rule.

**Caveat:** We don't know that ACTION7 is universally undo across all private games. Tier-1 disclosure should add "if observed behavior confirms ACTION7 is undo, the planner treats it as a backtrack edge; otherwise it is a normal action". Implementation: a `world_model.is_undo("ACTION7")` predicate that returns True only after observing `grid(t-1) == grid(t+1)` ≥ 3 times following an ACTION7 step. Deeply admissible — same Spelke `confirmed_transitions ≥ 3` consistency we already use.

**Estimated lift:** +0.03 to +0.08 podium-score on click-heavy games (because deep exploration is unblocked).

### V2 [Recommended] — Cross-level fingerprint transfer

**Currently:** `EpisodeTracker` resets per game; resonance library only retrieves between games. Within a multi-level game, level-1 fingerprint and learned rules are dropped at level-2 start.

**Proposal:** When `latest_frame.levels_completed` increments, snapshot `(fingerprint, rules, top_hypothesis)` into a per-game array. At level-2 start, **seed** the world model with prior-level rules (under a "decay" flag — confidence reset to threshold, must be re-confirmed). Same for resonance seeds: prior-level winning sub-policy becomes a candidate progress-path for MCTS at level-2.

**Tier-1 honesty:** This is in-context within ONE GAME — exactly the within-task transfer Spelke-priors-only allows. The corpus isn't broadened; the agent just doesn't forget within an episode. No external knowledge introduced.

**Estimated lift:** +0.05 to +0.12 on multi-level games (a meaningful fraction of the 110 expected private games).

### V3 [Recommended for ARC-AGI-2 sister] — D4 symmetry augmentation on train pairs

**Currently:** `arc2_solver.py` fits `Identity`, `Translate2`, `Recolor` on the raw train pairs. The SK Lab winners (and many top ARC-AGI-1 solvers) used the 8 D4 transforms (4 rotations × 2 reflections) to augment train pairs and detect rules invariant under D4.

**Proposal:** Add `_d4_augmented_pairs(train_pairs)` that for each pair `(in, out)` emits 8 transformed pairs `(R_k(in), R_k(out))` for `k ∈ {id, rot90, rot180, rot270, flipH, flipV, flipHrot90, flipVrot90}`. Then run `_fit_all_rules` over the augmented set. A rule that fits the augmented set is invariant under D4 — a substantially stronger constraint.

**Variation:** for test inference, predict on all 8 transformed test inputs, invert the transform, and majority-vote the answer. This is the cheap inference-time trick that historically delivered the most consistent ARC-AGI-1 gains.

**Tier-1 honesty:** D4 transforms are geometry priors, not task-family knowledge. Same admissibility as the existing `Translate2`. Disclosed in priors doc.

**Estimated lift:** +0.08 to +0.15 on ARC-AGI-2 (sister milestone, 2026-11-02 deadline).

### V4 [Speculative] — Hierarchical "macro-action" replay from resonance

**Currently:** Resonance retrieves winning policies as flat action sequences. MCTS uses them only as PRIOR-WEIGHTS at root.

**Proposal:** When a retrieved winning policy from the library is k-nearest by fingerprint AND its first action matches an available action, REPLAY the policy's first 3 actions as one macro-action (apply, observe, then decide). This is the "intuitive open" — humans don't re-derive opening moves; they replay them. Falls back to normal MCTS the moment the replay's predicted result diverges from the real environment.

**Tier-1 honesty:** Macro-replay only uses self-solved library entries. Same admissibility as current resonance seeds. Disclosed in priors doc.

**Estimated lift:** +0.02 to +0.06 on familiar-feeling games; risk is on novel games where the macro misleads.

---

## 4. HANGMAN — Skeptical adversarial test cases

These are the failure modes I'd bet money on encountering on private games.

### H1 [MF] — High-coverage but high-variance world model leads to MCTS following wrong predictions

If `coverage() >= 0.3` because we've seen ACTION1, ACTION2, ACTION3 each 3+ times, but the underlying rule for class 5 contradicts the rule for class 2, MCTS will plan against a confidently-wrong world model. The `_prune_contradicting_rules` step (G2) is supposed to filter these — but per G2, the filter doesn't carry forward. MCTS rollouts then accumulate negative reward against a phantom grid.

**Defensive fix:** Add a `world_model.calibration_score()` that compares last-K predictions to observed outcomes and caps MCTS confidence below 0.5 when calibration < 0.7. Cheap, defensive, Spelke-derived (consistency principle).

### H2 [MF] — A private game where ACTION6 click candidates from `click_quantizer` miss the only target

`click_quantizer.click_candidates` covers object centroids + bbox corners + edge midpoints + 9-quadrant fallback. If a private game requires clicking on a "ghost" cell (empty cell with semantic meaning — e.g. a doorway that's the background color), all four candidate sources MISS. The 9-quadrant fallback might land near, but not on, the target.

**Defensive fix:** Add a `between-objects-midpoint` candidate source: for each pair of nearby objects, emit the midpoint of their centroids as a candidate. Spelke-admissible (geometry prior, no game knowledge).

### H3 [MF] — Long thin episode where the abstain floor blocks us long after we should have abstained

`min_actions_before_abstain = 25` is the (b) floor. With G3 fixed (plateau check actually firing), we still wait 25 actions before any abstain. On a 30-action human baseline game, that's 84% of the budget gone before we even check. Per the AbstainPolicy docstring math, derived floor should be `2 * human_baseline ≈ 60`, which is WORSE.

**Defensive fix:** Either (a) accept that abstain is for very-long games only and document this is by design, OR (b) add an "early-fail" override: if the world-model variance is > 0.5 (totally wrong) AND novelty plateau hits within the first 10 actions, abstain immediately. The override is the same logic the docstring derives but un-floored.

### H4 [SF] — Hungarian tracker breaks down on >20 objects per scene

`tracker_hungarian.py:105-118` builds an O(m·n) cost matrix and either runs scipy LSA (O(n³)) or greedy fallback. On a private game with 50+ objects per scene, this becomes a per-step bottleneck (~25,000 cost calls). Wall-clock test recommended before submission.

### H5 [SF] — Goal inducer never observes a level-advancing pair

If the agent never wins a level (Misfit baseline early), `GoalInducer` receives zero `delta_levels > 0` observations. `_evaluate_*` then return `(0, contradictions)` for every hypothesis — every hypothesis has score `α / (α + contradictions + 2α)` = posterior with no positive support. `hypothesize` then returns hypotheses ranked only by `-contradictions`, which is misleading. The resonance-library tag at `cleanup()` will record a hypothesis based on PURE NEGATIVE evidence.

**Defensive fix:** In `hypothesize`, require `support >= 1` before emitting a hypothesis. Or in `_top_hypothesis_tag`, return empty string when top hypothesis has `support == 0`.

### H6 [SF] — Wall-clock self-kill might fire mid-game with no graceful shutdown

`is_done` returns True on wall-clock elapsed. But the agent never gets a chance to flush the resonance library if the kill fires mid-game (before WIN). The library record only happens on WIN (`misfit_agent.py:419`). Long games that nearly-win get zero credit toward the library.

**Defensive fix:** On wall-clock kill, if `tracker.level_advancing_actions()` exists, persist a partial-credit entry with `source="partial-self-solved"` (and update `ResonanceLibrary.record_solved` to accept that source under a Tier-1-disclosed flag).

---

## 5. MAVERICK — Synthesis and ship/no-ship

### Blockers (must fix before clicking submit)

1. **G1** — `select_action` enum-singleton mutation. Mirror the MCTS deep-copy contract. Add regression test.
2. **G2** — Refinement loop is theater. Either rename to honest `fit_then_prune` and retract HRM claim, OR implement real cross-iteration carry-over.
3. **G3** — Fingerprint normalization mismatch. Normalize fingerprint OR switch plateau check to cosine OR re-derive threshold.

### Must-fix (before submit, lower urgency than blockers)

4. **G4** — GoalInducer tiebreak prefers less-interpretable hypothesis.
5. **H1** — World-model calibration cap on MCTS confidence.
6. **H2** — Between-objects-midpoint click candidate.
7. **H3** — Early-fail abstain override.
8. **I4** — Align PAPER_DRAFT claims with G2 fix.

### Recommended (high ROI, can land alongside Blockers)

9. **V1** — ACTION7 as MCTS backtrack edge.
10. **V2** — Cross-level fingerprint transfer.

### Recommended (ARC-AGI-2 sister, deadline 2026-11-02)

11. **V3** — D4 symmetry augmentation in `arc2_solver`.

### Nice-to-have

12. **G5** — Delete or use `Misfit.tracker_hungarian`.
13. **G6** — Progress-path prior should be step-indexed.
14. **G7** — Note progressive widening's depth-vs-scene-reuse interaction.
15. **H4** — Profile Hungarian on >20-object scenes.
16. **H5** — Require positive support in goal-tag.
17. **H6** — Partial-credit resonance flush on wall-clock kill.
18. **V4** — Macro-action replay (later).

### Updated podium odds after Blockers + Must-Fixes (8 of 18 above)

> **Top-5 podium odds, Milestone #1 if Blockers + Must-Fixes ship: 0.42** (median estimate, 50% CI [0.30, 0.55]).

Without the Blockers fixed: **0.18** (median, 50% CI [0.10, 0.28]).

The biggest swing factor is G2 — if we ship claiming HRM-style refinement and the judges read the code, that's a credibility hit on a competition partly judged on honesty. If we fix G2 honestly (rename to `fit_then_prune` and document), nothing of the agent's actual behavior changes but the writeup stops over-claiming.

### Tier-1 attestation status

Clean on every dimension I checked. CI grep guards remain green. Spawn the integration test suite again post-fixes — no change expected. **Tier-1 attestation clean: TRUE.**

---

## Appendix A — File-level finding map

| File | Findings |
|---|---|
| `src/misfit_agent/action_search.py` | G1 (B) |
| `src/misfit_agent/world_model.py` | G2 (B), I4 (SF), H1 (MF) |
| `src/misfit_agent/fingerprint.py` | G3 (B) |
| `src/misfit_agent/goal_inducer.py` | G4 (MF), H5 (SF) |
| `src/misfit_agent/abstain_policy.py` | G3 (B), H3 (MF) |
| `src/misfit_agent/misfit_agent.py` | G5 (SF), H6 (SF), V1 (Rec), V2 (Rec) |
| `src/misfit_agent/mcts_puct.py` | G6 (SF), G7 (N) |
| `src/misfit_agent/click_quantizer.py` | H2 (MF) |
| `src/misfit_agent/tracker_hungarian.py` | H4 (SF) |
| `src/misfit_agent/arc2_solver.py` | V3 (Rec) |

## Appendix B — Reproducer summaries (all verified live against current tree)

- **G1 reproducer** — `select_action` returns same enum singleton across calls; mutating `A6.data` then re-calling `select_action` shows aliased data. Reproduced via stub arcengine + `select_action(scene, t, [A6], [], 100, None)`. Two consecutive calls returned `True` for `a1 is a2`.
- **G2 reproducer** — `fit_with_refinement(obs, max_iters=3)` against the same contradiction-bearing observations across two calls produces identical rules; second call merely double-counts iterations. Score history was `[1.0, 1.0]` — no improvement signal possible because `fit()` rebuilds from scratch.
- **G3 reproducer** — `fingerprint_episode` on a 50-step single-scene tracker returns L2 norm 4.94 (claim: unit-norm). Dim 9 alone = 3.93. Plateau threshold 0.01 is calibrated for unit-norm.

Verified: 2026-06-16 by LakeStrike. Repo state: post `7bf04eb`. Test baseline: 93/93 green.
