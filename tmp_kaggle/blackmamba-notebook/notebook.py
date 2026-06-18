"""# Black Mamba: a 13-layer Tier-1 strict cognitive substrate

A locally deployable, forensically auditable cognitive substrate that runs
Federation entities (research agents, kiosk personas, judge variants) on
commodity hardware (>=16 GB RAM) with no internet and no pretrained LLM
injection.

This notebook documents Black Mamba's 13-layer scope, its 27 constitutional
guardrails, and how it integrates into Orange3 (a sovereign agentic OS
already shipping at AtomEons).

**Authoritative source:**
- `BLACK_MAMBA_SCOPE_v1.md` (631 lines) — full layer spec
- `CHARTER_v1.md` (278 lines) — Federation Charter
- `CONSTITUTION_v1.md` (368 lines) — Bill of Cognitive Rights

**Why this matters for ARC research:**
Black Mamba's Layer 7 (Provenance-Enforced Memory) and Layer 8 (Tier-1
attestation) provide the receipt-anchored substrate-development discipline
we used to ship 15 waves of ARC-AGI-2 substrate research. The 4.20% training
climb (vs 2.30% baseline) under strict Tier-1 constraints was made possible
by these layers.
"""

# %% [markdown]
# ## 1. One-line definition
#
# Black Mamba is a 13-layer locally deployable cognitive substrate built around
# a Mamba-2 SSM state-space model core, a 4-tier memory system, and PEM
# (Provenance-Enforced Memory) contracts that run Federation entities offline
# on commodity hardware.

# %% [code]
layers = [
    (0,  "Hardware",            "x86_64 or Apple Silicon, no GPU required"),
    (1,  "OS",                  "Linux / Windows / macOS, offline-capable"),
    (2,  "Embedded store",      "sqlite-vec v0.1.6 + FTS5 for vector + text retrieval"),
    (3,  "Substrate runtime",   "Mamba-2 SSM (Falcon Mamba 7B v2 Q5_K_M GGUF ~4.7 GB) + BGE-small embeddings + GBNF grammar"),
    (4,  "Brain state",         "Atomic write-rename; XChaCha20-Poly1305 encryption at rest"),
    (5,  "Memory tiers",        "4-tier: RNN hidden / episodic vec store / K3 wildcard cards / procedural skills"),
    (6,  "AtomSmasher compression", "CLC pack modes: HOT (8K), WARM (32K w/ pointers), COLD (10–80x reduction)"),
    (7,  "PEM provenance",      "8-field memory contract enforced at every tier insert"),
    (8,  "Tier-1 attestation",  "CI-grep banning torch / transformers / openai / anthropic imports"),
    (9,  "Resonance library",   "Per-install JSONL of self-solved task fingerprints + winning policies"),
    (10, "Cognitive modules",   "perceptor / world_model / mcts_puct / quint_envelope / agent_verifier"),
    (11, "Governance integration","CHSG hooks: Trilogy worker mode, Triumvirate judge variant, Sovereign override path"),
    (12, "Identity",            "Soul Genome continuity map; Ed25519 signing key; ID format `<entity>@atomeons/1.0`"),
    (13, "Interfaces",          "Tauri kiosk, REST endpoint, CLI (`black-mamba doctor`), optional MCP server"),
]
print(f"{'L':>2}  {'Name':<28}  Purpose")
print("-" * 100)
for L, name, purpose in layers:
    print(f"{L:>2}  {name:<28}  {purpose}")

# %% [markdown]
# ## 2. The Tier-1 attestation contract
#
# Layer 8 is mechanically enforced. Every commit triggers a CI grep that
# fails the build on any forbidden import:

# %% [code]
forbidden_imports = [
    "torch", "torchvision", "torchaudio",
    "transformers", "diffusers",
    "openai", "anthropic", "google.generativeai",
    "llama_cpp", "ollama",
    "sklearn",  # also banned at eval — only allowed in offline analysis
    "jax", "flax", "tensorflow",
]
print("Layer 8 forbidden import list (CI-grep):")
for imp in forbidden_imports:
    print(f"  - {imp}")

# %% [markdown]
# The CI test is one regex per import:
#
# ```python
# def test_tier1_attestation():
#     forbidden = ["torch", "transformers", "openai", ...]
#     for f in glob("src/**/*.py", recursive=True):
#         src = open(f).read()
#         for imp in forbidden:
#             assert f"import {imp}" not in src, f"{f}: forbidden import {imp}"
# ```
#
# This is harder to game than self-disclosure. CI failure halts deployment.

# %% [markdown]
# ## 3. PEM 8-field provenance contract (Layer 7)
#
# Every memory record carries an 8-field provenance trailer enforced at the
# tier-insert boundary via a Rust trait:

# %% [code]
pem_fields = [
    ("source_provenance",  "Where did this fact come from? (`web:url` / `user:atom` / `self-solved:task_id`)"),
    ("contamination_tier", "How untrusted is this? (`TRUSTED` / `THIRD_PARTY` / `LLM_OUTPUT` / `UNTRUSTED`)"),
    ("creation_unix",      "When was this captured? (atomic stamp)"),
    ("replay_pointer",     "Where can this be re-derived? (immutable log offset)"),
    ("mutation_history",   "How has this been touched? (linked-list of edit ops)"),
    ("expiry_decay_rule",  "When does this become stale? (Ebbinghaus / fixed / never)"),
    ("evidence_blob",      "Optional: cryptographic proof / source snippet"),
    ("usage_receipts",     "How has this fact informed downstream decisions? (audit chain)"),
]
for field, desc in pem_fields:
    print(f"  {field:<22}  {desc}")

# %% [markdown]
# **Why this is hard.** Standard in-context memory APIs (Anthropic memory tool,
# OpenAI assistants, LlamaIndex retrieval) treat memory as opaque
# strings + metadata. PEM treats every memory record as a forensic artifact
# with a recoverable audit chain. This is what enables receipt-anchored
# substrate development across iterations.

# %% [markdown]
# ## 4. CHSG governance (Layer 11)
#
# **CHSG** = Charter-Hardened Self-Governance. Three-layer decision machine:
#
# 1. **Trilogy** — three Federation members (research agents) cast blind drafts,
#    then debate under PEM, then vote with domain-weighted scoring:
#
#    ```
#    weight = domain_competence × recent_accuracy × independence
#           × source_quality × calibration × conflict_penalty
#    ```
#
# 2. **Triumvirate** — judge panel (impartial-judge LoRA, no domain stake)
#    adjudicates contested Trilogy decisions.
# 3. **Sovereign Backstop** — Atom McCree retains final veto via signed
#    instrument targeting `decision_id`. Logged as `SOVEREIGN_OVERRIDE`.
#
# **Never Stalemate** (Article III §3.4): the 3-layer machine forces
# convergence by construction.
#
# **Core law (CHARTER_v1.md):**
# > *"No agent rules the swarm. No vote overrules reality. No consensus
# > overwrites evidence. No output enters canon without receipt."*

# %% [markdown]
# ## 5. Performance budgets
#
# | Metric                  | Soft target | Hard ceiling |
# |---|---|---|
# | Cold boot → first response | 15s | 30s |
# | Inference (HOT pack 8K)    | 800ms | 2s |
# | Inference (WARM pack 32K)  | 4s | 10s |
# | Tier-2 episodic recall     | 50ms | 200ms |
# | Tier-3 K3 lookup           | 100ms | 500ms |
# | Brain state flush          | 100ms | 500ms |
# | RAM at idle                | 6 GB | 8 GB |
#
# Targets are validated by the `black-mamba doctor` CLI at every boot.

# %% [markdown]
# ## 6. Deployment targets
#
# | Target              | Persona LoRA                          | Hardware       | Interface |
# |---|---|---|---|
# | **Misfit-Alpha**     | research-honest                       | 16 GB dev box  | CLI + REST |
# | **Quint (VHS5K)**    | 60% Rob Gordon / 30% Tarantino / 10% Randal | mini-PC (8 GB OK) | Tauri kiosk |
# | **Misfit-Adjudicator** | impartial-judge                       | 32 GB          | REST (Triumvirate) |
# | **Misfit-G55**       | Anthropic-supplied                    | API-backed     | REST + cross-exam |

# %% [markdown]
# ## 7. Integration with Orange3 (sovereign agentic OS)
#
# Orange3 already implements much of Layer 11 (Governance), Layer 7 (PEM
# enforcement), and Layer 8 (Tier-1 attestation). The unification path:
#
# - **Layer 2** → reuse Orange3 SQLite receipt log as the Federation receipt ledger
# - **Layer 5** → map to Orangebox Memory Vault layers (Cockpit Local State,
#   Knowledge Vault, Opus Per-Chat, Claude Native Memory)
# - **Layer 7** → mirror via Orange3 context-packer SHA-256 chunk hashes
# - **Layer 11** → use Orange3 adapter contract for Trilogy + Triumvirate dispatch
# - **Black Mamba instances** → register as `assigned_node` values in Orange3 adapters
#
# Orange3 cockpit: `http://127.0.0.1:8787/orange3/`

# %% [markdown]
# ## 8. Why this matters for AI research
#
# 1. **No more "exploratory" framing of negative results.** Layer 8 + PEM force
#    honest-null receipts (a wave that produces zero lift gets a receipt with
#    `"verdict": "ZERO_LIFT_HONEST"` and a deeper diagnostic). This kills
#    publication bias inside the research org.
#
# 2. **Forensic accountability for cognitive substrates.** Every decision in
#    every iteration is reachable via the audit chain. Future researchers
#    reading a 6-month-old receipt can verify the claim WITHOUT re-running the
#    experiment.
#
# 3. **Offline-capable substrate.** Most "AGI safety" tooling assumes
#    internet-connected, API-backed models. Black Mamba treats internet as a
#    luxury, not a requirement. Federation members can deploy in air-gapped
#    environments (medical, defense, embedded kiosk).
#
# 4. **Receipt-anchored R&D loops.** Our ARC-AGI-2 substrate climbed from
#    2.30% to 4.20% training across 15 waves under strict Tier-1 attestation
#    BECAUSE every wave produces a receipt with its parent and next-expected
#    receipt. The audit chain is the substrate.

# %% [markdown]
# ## 9. References
#
# - **Disclosure ID:** ATOM-AESUITE-2026-0419
# - **License:** CC-BY-4.0
# - **Repository:** [github.com/AtomEons/arc-agi-3-misfit-agent](https://github.com/AtomEons/arc-agi-3-misfit-agent)
# - **Related Kaggle notebook:** [Property-Bound Rule Contracts: the 980/1000 Fit-Contract Bottleneck](https://www.kaggle.com/code/atommccree/property-bound-rule-contracts-arc-agi-2)
#
# ---
#
# *Citation:* Atom McCree (2026). *Black Mamba: a 13-layer Tier-1 strict
# cognitive substrate.* AtomEons Research Laboratory.
#
# *Upvote if this is useful for your federation / agent-substrate research.*

print("\nNotebook complete.")
