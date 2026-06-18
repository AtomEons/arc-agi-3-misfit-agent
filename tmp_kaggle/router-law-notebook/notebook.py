"""# The Router Law: Four Axioms for the Corpus Callosum

## The constraint that defines the architecture

The Double-Brain partition gives us a fast Reflex (RAM-locked, sparse,
sub-100 ms) and a slow Cortex (SSD-locked, dense, 100 ms to 4 s). What it
does not give us is the answer to: who decides which one fires?

The naive answer -- "a classifier model" -- is structurally wrong:

- Every reflex pays the classifier's forward-pass cost first
  → defeats the latency budget.
- The classifier has its own Dunning-Kruger failure modes (small classifiers
  are confidently wrong).
- The classifier becomes another component to train, audit, ground in PEM,
  and verify under Tier-1 attestation.

Biology solved this hundreds of millions of years ago. Vertebrate brains
do not use a third neural network to decide between reflex and deliberation.
They use METABOLIC GATING: the cortex is metabolically expensive, the cortex
stays asleep unless thermodynamically forced awake by ascending physical
signals.

The Router Law is the substrate's metabolic gating. Four brutal,
deterministic, zero-compute physical tripwires. No neural network. No
training. Pure silicon-level filters.
"""

# %% [markdown]
# ## Axiom 1 - The Vagus Filter (zero-compute pre-triage)
#
# Reflexes do not read novels.

# %% [code]
import re

VAGUS_WORD_LIMIT = 40
VAGUS_TRIGGER_REGEX = re.compile(
    r"\b(why|explain|plan|code|compare|yesterday|history|summarize|analyze)\b|```",
    re.IGNORECASE,
)

def vagus_filter(user_input: str) -> bool:
    """Return True if input must skip the reflex and go straight to cortex."""
    if len(user_input.split()) > VAGUS_WORD_LIMIT:
        return True
    if VAGUS_TRIGGER_REGEX.search(user_input):
        return True
    return False

# Examples
inputs = [
    "turn on the porch light",
    "why did the deploy fail yesterday",
    "summarize the last 200 commits",
    "ack",
    "explain how the Vagus Filter works",
]
for inp in inputs:
    print(f"  {'COR' if vagus_filter(inp) else 'REF'}  {inp}")

# %% [markdown]
# ## Axiom 2 - The Logprob Flinch (mathematical hesitation)
#
# Small models suffer from AI Dunning-Kruger. They cannot self-report their own
# incompetence. So we do not ask them. We measure their hesitation directly.

# %% [code]
LOGPROB_FLINCH_THRESHOLD = 0.85

def reflex_passed_logprob_flinch(first_token_prob: float) -> bool:
    return first_token_prob >= LOGPROB_FLINCH_THRESHOLD

# Reflex emits "TOKEN  prob=0.93" → confident → emit
# Reflex emits "TOKEN  prob=0.41" → distribution shattered → SIGKILL + wake cortex
for p in [0.99, 0.92, 0.85, 0.71, 0.32]:
    verdict = "REFLEX" if reflex_passed_logprob_flinch(p) else "FLINCH -> CORTEX"
    print(f"  p={p:.2f}  →  {verdict}")

# %% [markdown]
# **The key insight:** the model cannot lie about its own probability
# distribution. The distribution IS the lie detector. A confident wrong answer
# still has a confident-looking distribution -- but a guessing model's
# distribution shatters across many possible tokens.

# %% [markdown]
# ## Axiom 3 - Token Asphyxiation (breath limit)
#
# Muscle twitches are short. Reflexes do not write essays.

# %% [code]
TOKEN_ASPHYXIATION_LIMIT = 32

def reflex_passed_token_asphyxiation(token_count: int, emitted_terminus: bool) -> bool:
    return emitted_terminus and token_count < TOKEN_ASPHYXIATION_LIMIT

# A reflex that needs 33+ tokens is not a reflex. It is deliberation that
# hasn't admitted it's deliberation. We delete the rambling attempt from
# the SQLite spine so the hallucination is NOT remembered as canon.
print("Reflex hits token 32 without <|im_end|>?")
print("  → SIGKILL the process")
print("  → DELETE FROM events WHERE id = <provisional_id>")
print("  → wake cortex")

# %% [markdown]
# ## Axiom 4 - The Lethality Matrix (basal-ganglia veto)
#
# Reflexes can drop a hot plate to save your hand. They cannot sign a mortgage.

# %% [code]
LETHALITY_THRESHOLD = 2

# Every tool in the spine carries a hardcoded risk_level (0-3).
TOOL_LADDER = [
    ("read_file",      0, "trivial"),
    ("query_state",    0, "trivial"),
    ("write_log",      1, "reversible"),
    ("cache_set",      1, "reversible"),
    ("delete_file",    2, "destructive"),
    ("git_commit",     2, "destructive"),
    ("rm_rf",          3, "Sovereign-only"),
    ("payment_send",   3, "Sovereign-only"),
]
print(f"{'Tool':<16} {'Risk':<6} {'Class':<16} Reflex-allowed?")
print("-" * 60)
for name, risk, klass in TOOL_LADDER:
    allowed = "YES" if risk < LETHALITY_THRESHOLD else "NO (Mirror Audit)"
    print(f"{name:<16} {risk:<6} {klass:<16} {allowed}")

# %% [markdown]
# When Reflex proposes a tool with `risk >= 2`, the Corpus Callosum freezes the
# execution and routes the proposal to the Cortex for a Mirror Audit:
#
# > "System 1 proposes executing rm -rf path X based on User Input Y. Review
# > context, audit for logical contradictions, and Approve/Deny."
#
# The Cortex either signs the action with a PEM receipt or rejects it with a
# logged dissent.

# %% [markdown]
# ## The Cybernetic Router (bare-metal Python)

# %% [code]
import sqlite3
from typing import Optional

def corpus_callosum(user_input: str, db: sqlite3.Connection) -> str:
    # AXIOM 1 - Vagus Filter
    if vagus_filter(user_input):
        return wake_cortex("Vagus Filter", user_input)

    # Fire reflex
    action, confidence, token_count = query_blackmamba_reflex(user_input)

    # AXIOM 2 - Logprob Flinch
    if confidence < LOGPROB_FLINCH_THRESHOLD:
        return wake_cortex(f"Logprob Flinch p={confidence:.2f}", user_input)

    # AXIOM 3 - Token Asphyxiation
    if token_count >= TOKEN_ASPHYXIATION_LIMIT:
        rollback_reflex_event_in_spine(db)
        return wake_cortex("Token Asphyxiation", user_input)

    # AXIOM 4 - Lethality Matrix
    row = db.execute("SELECT risk_level FROM tools WHERE name = ?",
                     (action['tool'],)).fetchone()
    risk = row[0] if row else 3
    if risk >= LETHALITY_THRESHOLD:
        return wake_cortex("Mirror Audit Required", user_input,
                           proposed_reflex=action)

    # All four tripwires cleared
    receipt = execute_tool(action)
    log_to_sqlite_spine(db, "System 1 (Reflex)", user_input, action, receipt)
    return receipt

# Stubs for the notebook demonstration
def query_blackmamba_reflex(s): return ({"tool": "read_file"}, 0.93, 12)
def execute_tool(a): return "ok"
def log_to_sqlite_spine(*a, **k): pass
def rollback_reflex_event_in_spine(db): pass
def wake_cortex(reason, *a, **k): return f"CORTEX woke: {reason}"

print("Router is plumbing. No floating-point inside the Router.")

# %% [markdown]
# ## The Metabolic Envelope
#
# On a Beelink N150 (15W TDP, 8-core Twin Lake, Intel iGPU via Vulkan, NVMe SSD):

# %% [code]
regimes = [
    ("Reflex",  "~90%", "CPU spikes ~50ms. SSD sleeps. iGPU sleeps. Near-zero thermal load."),
    ("Cortex",  "~10%", "NVMe spins up. 4.7GB GGUF streams to iGPU via Vulkan. N150 hits 15W for ~3s. Receipt to SQLite. Drops to rest."),
]
print(f"{'Regime':<8} {'Frequency':<10} Hardware state")
print("-" * 100)
for r, f, s in regimes:
    print(f"{r:<8} {f:<10} {s}")
print()
print("The substrate SURFS its own thermodynamic limits.")
print("Edge-native sovereign AGI cannot afford to wake the Cortex on every turn.")
print("The Router Law makes 90% / 10% the steady state.")

# %% [markdown]
# ## Cross-references
#
# - Black Mamba v1 doctrine: [Black Mamba 13-Layer Cognitive Substrate](https://www.kaggle.com/code/atommccree/black-mamba-13-layer-cognitive-substrate-atomeons)
# - Black Mamba v2 (Double-Brain): [Double Brain Cognitive Architecture TSU Mamba](https://www.kaggle.com/code/atommccree/double-brain-cognitive-architecture-tsu-mamba)
# - Repository: [github.com/AtomEons/arc-agi-3-misfit-agent](https://github.com/AtomEons/arc-agi-3-misfit-agent)
#
# ## Citation
#
# Atom McCree (2026). *The Router Law: Four Axioms for the Corpus Callosum.*
# AtomEons Research Laboratory. CC-BY-4.0.
#
# *Disclosure ID: ATOM-BM-v2-RouterLaw-2026-0617*
#
# ---
#
# *Upvote if this is useful for your edge AGI / sovereign-node / agent
# orchestration research.*

print("\nRouter Law complete. The metabolic gating is the architecture.")
