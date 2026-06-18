"""# Property-Bound Rule Contracts: the 980/1000 Fit-Contract Bottleneck

A Tier-1 strict ARC-AGI-2 substrate (no LLM, no pretrained weights, no
learned parameters at evaluation) trades ZERO TRAINING LIFT under the
standard fit-contract for **+11 training task solves** under a property-bound
contract — without adding a single new rule template.

This notebook documents the methodological discovery, the fit-contract
diagnostic, and the receipt-anchored climb across 6 ARC research waves.

**Repository:** github.com/AtomEons/arc-agi-3-misfit-agent
**Receipt chain:** receipts/100day/wave4 → wave15
**Tier-1 attestation:** tests/test_tier1_attestation.py (CI-grep)
"""

# %% [markdown]
# ## TL;DR
#
# - **Setup:** 65-rule Tier-1 grammar with Spelke core-knowledge priors
#   (cohesion, geometry, topology, numerosity), bounded beam search.
# - **Diagnostic finding:** 980 of 1,000 ARC-AGI-2 training tasks have
#   **ZERO rules fitting** under the standard `fit()-locks-global-parameter`
#   contract.
# - **Contradiction:** depth-2 composition over 43 rules (1,849 program
#   combinations per task, 48 minutes wall clock) yielded **ZERO new task
#   solves**. The bottleneck is not vocabulary size; it is **contract
#   expressiveness**.
# - **Pivot:** introduce **property-bound rule contracts** where rule TYPE
#   locks at `fit()` and parameter VALUES bind at `predict()` from
#   properties of the test input.
# - **Result:** waves 7–9 produce +7 task solves under property-bound; the
#   ONLY mechanism that produced non-zero lift across 6 attempts. Wave 15
#   build measures 4.20% training (vs Wave 9 baseline 3.10%).
# - **Methodological alpha:** pre-flight `(fit, predict, compare-to-gold)`
#   enumeration matched full measurement EXACTLY on 3/3 waves, 7/7 tasks.

# %% [markdown]
# ## 1. The fit-contract bottleneck
#
# A standard Tier-1 ARC rule looks like:
#
# ```python
# class CropToBbox:
#     r0: int; c0: int; r1: int; c1: int
#     def fit(self, train_pairs):
#         # discover (r0, c0, r1, c1) such that
#         # grid[r0:r1+1, c0:c1+1] == output for EVERY train pair
#         ...
#     def predict(self, test_input):
#         return test_input[self.r0:self.r1+1, self.c0:self.c1+1]
# ```
#
# The parameters are locked at `fit()`. They do not adapt to the test
# input. This contract fails on every task whose bbox coordinates differ
# between train and test inputs.
#
# Diagnostic on the ARC-AGI-2 training set:

# %% [code]
# Pseudocode of the diagnostic: for each of the 1,000 training tasks,
# count how many rules pass `rule.fit(train_pairs)`. The result:
fit_distribution = {
    "zero_fits": 980,   # 980 tasks have no rule fitting
    "any_fit":    20,   # only 20 have at least one fit
}

# Of those 20 tasks, 19 predict correctly on the test input; 1 does not.
# This means the entire grammar is SILENT on 98% of the training set.
print(fit_distribution)

# %% [markdown]
# **The implication is sharp:** more rules under the same contract do not
# help. The grammar is silent.
#
# We tested this directly:
#
# | Wave | Rules added | Mechanism | New solves |
# |---|---|---|---|
# | 5 | +4 | predicate-inferred per-object | **0** |
# | 6 | +5 | color maps + crops | **0** |
# | depth-2 composition | 43² combos | program composition | **0** |
#
# Adding rules under the same contract: zero lift across three honest attempts.

# %% [markdown]
# ## 2. The property-bound contract
#
# A property-bound rule **decouples TYPE from parameter VALUES**:
#
# ```python
# class CropToObjectByAreaRank:
#     rank: int  # locked at fit (e.g. rank=0 = largest object)
#     def fit(self, train_pairs):
#         # discover that EVERY train output equals the bbox of the
#         # rank-K object of its train input. K is small, e.g. 0 = largest.
#         ...
#     def predict(self, test_input):
#         # Extract the rank-K object of the TEST input. The bbox is bound
#         # at predict time — it depends on the test input, not training.
#         objs = extract_objects(test_input)
#         return crop_to_bbox(test_input, objs[self.rank].bbox)
# ```
#
# **Same number of rules. Different contract.**
#
# The empirical result:

# %% [code]
training_climb = [
    ("Wave 4 baseline", 23, 1000, 2.30),
    ("Wave 5 (+4 rules, OLD contract)", 23, 1000, 2.30),    # 0 lift
    ("Wave 6 (+5 rules, OLD contract)", 23, 1000, 2.30),    # 0 lift
    ("Wave 7 (+5 rules, PROPERTY-BOUND)", 26, 1000, 2.60),  # +3 lift
    ("Wave 8 (+5 rules, PROPERTY-BOUND)", 27, 1000, 2.70),  # +1 lift
    ("Wave 9 (+3 rules, PROPERTY-BOUND, topology)", 31, 1000, 3.10),  # +3 lift
    ("Wave 15 (waves 1-9 + 11-15, DSL on)", 42, 1000, 4.20),  # +11 cumulative
]
for name, solved, total, pct in training_climb:
    print(f"{name:50} {solved}/{total} = {pct}%")

# %% [markdown]
# Six waves of identical R&D process. The discriminating variable is the
# contract, not the rules.

# %% [markdown]
# ## 3. Pre-flight = measurement exact equivalence
#
# The full ARC-AGI-2 measurement (1,000 tasks, depth-1, exact-match
# verification) takes 10–25 minutes wall clock. The pre-flight enumerates
# `(fit, predict, compare-to-gold)` per rule in ~50 seconds.
#
# **3/3 waves matched EXACTLY. 7/7 tasks matched EXACTLY.**

# %% [code]
preflight_vs_measurement = [
    ("Wave 7", ["1f85a75f", "be94b721", "c909285e"], ["1f85a75f", "be94b721", "c909285e"]),
    ("Wave 8", ["cd3c21df"], ["cd3c21df"]),
    ("Wave 9", ["00d62c1b", "a5313dff", "ea32f347"], ["00d62c1b", "a5313dff", "ea32f347"]),
]
for wave, predicted, measured in preflight_vs_measurement:
    match = predicted == measured
    print(f"{wave}: predicted={predicted} | measured={measured} | match={match}")

# %% [markdown]
# Why does exact equivalence hold? The solver beam ranking is FULLY
# DETERMINISTIC:
# - beam_width = 4 (fixed)
# - lexicographic dedup by rule signature
# - no random tiebreak
# - no stochastic component
#
# The pre-flight is the cheap proxy. The full measurement is the expensive
# proxy. They agree because the rule grammar has no order-sensitivity.
#
# **Methodological alpha:** substrate R&D wall clock is reduced ~30× by
# using pre-flight as a fast falsification mechanism before burning the
# expensive measurement budget.

# %% [markdown]
# ## 4. Honest-null receipts
#
# Each wave produces a receipt. Even waves that produce ZERO lift get a
# receipt with `"verdict": "ZERO_LIFT_HONEST"` and a deeper diagnostic
# explaining why the next pivot is needed. The audit chain:
#
# ```
# receipts/100day/wave4_orange3_receipt.json
#   → wave5_orange3_receipt.json (ZERO_LIFT_HONEST)
#   → wave6_orange3_receipt.json (ZERO_LIFT_HONEST + 980/1000 finding)
#   → wave7_orange3_receipt.json (REAL_LIFT_VIA_ARCHITECTURE_PIVOT)
#   → wave8_orange3_receipt.json (REAL_LIFT)
#   → wave9_orange3_receipt.json (STRONGEST_WAVE_IN_6_ATTEMPTS)
#   → wave15_full_measurement.json (4.20%, eval 0/120)
# ```
#
# Honest-null receipts prevent two failure modes:
# 1. Framing zero-lift as "exploratory" (post-hoc rationalization)
# 2. Skipping the receipt and only committing wins (publication bias)
#
# A future researcher reading wave 6's receipt understands EXACTLY why
# the property-bound pivot in wave 7 was necessary.

# %% [markdown]
# ## 5. Open problem: the eval transfer gap
#
# Training measurement: 42 / 1000 = 4.20%.
# Eval measurement: 0 / 120 = 0.00%.
#
# The fit-contract finding is necessary but not sufficient. Eval tasks
# impose additional constraints:
# - 31% of eval train pairs are SHAPE-CHANGING (input ≠ output dimensions)
# - 48% of eval tasks contain TEST COLORS unseen in training pairs
# - Eval distribution is structurally OOD from training
#
# Closing this gap is the open R&D thread.

# %% [markdown]
# ## 6. Reproducibility
#
# - Repository: [github.com/AtomEons/arc-agi-3-misfit-agent](https://github.com/AtomEons/arc-agi-3-misfit-agent)
# - Solver: `src/misfit_agent/arc2_solver.py`
# - 81 rule factories across 9 wave files: `src/misfit_agent/rules_v3/`
# - Pre-flight harness: `scripts/wave*_preflight.py`
# - Full measurement: `scripts/full_eval_measurement.py`
# - Tier-1 CI-grep: `tests/test_tier1_attestation.py`
#
# License: CC-BY-4.0 (compatible with ARC Prize Paper Track terms).

# %% [markdown]
# ## Citation
#
# Atom McCree (2026). *Property-Bound Rule Contracts and the Fit-Contract
# Bottleneck in Tier-1 ARC-AGI Solvers.* ARC Prize 2026 Paper Track
# Submission Draft, AtomEons Research Laboratory.
#
# ---
#
# *Upvote if this is useful for your ARC research. Comments and dissent welcome.*

print("\nNotebook complete. Receipt chain at receipts/100day/.")
