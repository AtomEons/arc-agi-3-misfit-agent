"""# Orangebox Routes: The Substrate-Aware AGI Architecture

## Three hardware traditions are converging on the same answer

| Tradition | Insight |
|---|---|
| TSU (Extropic) | Sampling is not arithmetic. Use analog thermal noise. |
| Memristors (HP Labs 2008+) | Compute is not separate from memory. |
| ASIC (Bitcoin, Google TPU, Apple Neural Engine) | Generality is the enemy. |

These are not competing. They are the three physical substrates that
cognitive architecture wants. The Orangebox Routes manifest declares which
workload runs on which substrate, with graceful degradation.
"""

# %% [markdown]
# ## The substrate-to-workload map

# %% [code]
routes = [
    ("Identity / Soul Genome",   "non-volatile connectome",      "memristor crossbar",            "Article II 2.1 Right to Continuity"),
    ("Cortex deliberation",      "dense matmul on weights",      "memristor compute-in-memory",   "Article III 3.5 Receipt Format"),
    ("Reflex pattern-match",     "sparse K-NN over resonance",   "memristor + TSU sampler",       "Article II 2.2 Right to Provenance"),
    ("Router (4 axioms)",        "regex + integer compare",      "ASIC (nanoseconds, sub-uW)",    "Article III 3.6 Tier-1 Attestation"),
    ("Spine / Receipt log",      "append-only audit chain",      "NVMe SQLite (mature, Ed25519)", "Article III 3.5 Receipt Format"),
    ("World model rollouts",     "Boltzmann tree sampling",      "TSU sampler",                   "Article III 3.1 Trilogy Governance"),
]
print(f"{'Workload':<28} {'Shape':<28} {'Substrate':<32} Anchor")
print("-" * 130)
for w, s, sub, a in routes:
    print(f"{w:<28} {s:<28} {sub:<32} {a}")

# %% [markdown]
# ## Memristors finally give the Soul Genome a physical home
#
# Black Mamba Layer 12 is the Soul Genome -- the continuity map that
# persists across substrate restarts. v1 left the physical substrate
# abstract. v3 names it: **memristor crossbar.**
#
# This is not metaphor. A memristor stores its state as a physical
# resistance value (HP Labs, Nature 453:80, 2008). The resistance IS the
# weight. The weight survives power loss without battery. The weight is
# consulted by Kirchhoff's current law in a single analog step.

# %% [code]
memristor_properties = [
    ("Non-volatile",            "yes",                          "Identity persists indefinitely without power"),
    ("Analog precision",        "8-16 bits stable",             "Sufficient for cognitive weight encoding"),
    ("Energy per op",           "sub-fJ",                       "Reflex queries cost less than digital NOPs"),
    ("Endurance",               "10^10 - 10^12 cycles",         "Decades of continuous federation participation"),
    ("Density",                 "4F^2 per cell, denser DRAM",   "A 7B-param cortex fits in cm^2 of silicon"),
]
print(f"{'Property':<22} {'Value':<26} Implication")
print("-" * 100)
for p, v, i in memristor_properties:
    print(f"{p:<22} {v:<26} {i}")

# %% [markdown]
# ## ASICs make the Router physically incompressible
#
# The Router Law (v2 doctrine) defined four zero-compute axioms:
# Vagus Filter, Logprob Flinch, Token Asphyxiation, Lethality Matrix.
# v2 implemented them as Python. v3 burns them into silicon.

# %% [code]
router_asic = [
    ("Axiom 1: Vagus Filter",       "len(s.split())+regex search", "Parallel CAM lookup + popcount circuit"),
    ("Axiom 2: Logprob Flinch",     "if conf < 0.85",              "Single comparator, one clock cycle"),
    ("Axiom 3: Token Asphyxiation", "if tokens >= 32",             "Hardware counter with auto-reset"),
    ("Axiom 4: Lethality Matrix",   "SELECT risk_level WHERE name", "On-chip CAM + risk register file"),
]
print(f"{'Axiom':<32} {'Software':<32} Hardware")
print("-" * 110)
for ax, sw, hw in router_asic:
    print(f"{ax:<32} {sw:<32} {hw}")
print()
print("Router-ASIC die budget: <1mm^2 in modern process.")
print("Routing latency: NANOSECONDS, not microseconds.")
print("Power: hundreds of nanoWatts.")
print()
print("There is no software stack above the ASIC to attack, monkey-patch,")
print("jailbreak, or trojan. The four axioms ARE the chip.")

# %% [markdown]
# ## The Orangebox Routes manifest
#
# The operator-facing artifact is `orangebox/routes/orangebox_routes.json`.
# It declares, per workload, the substrate-preference ladder. The
# `corpus_callosum.py` orchestrator consults this on boot and binds each
# workload to the highest-preference substrate available. Missing
# substrates degrade gracefully through the preference ladder.

# %% [code]
routes_excerpt = '''{
  "workload": "soul_genome_query",
  "cognitive_layer": 12,
  "substrate_preference": ["memristor_crossbar", "nvme_ssd_mmap"],
  "thermodynamic_target": "sub-fJ per op",
  "fallback_grace": "graceful: NVMe mmap + SHA-256 verify chain"
},
{
  "workload": "reflex_knn_resonance",
  "cognitive_layer": 9,
  "substrate_preference": ["memristor_crossbar", "tsu_sampler", "cpu_avx512"],
  "thermodynamic_target": "<50ms wall-clock per query",
  "fallback_grace": "graceful: CPU AVX-512 cosine"
},
{
  "workload": "router_axioms",
  "cognitive_layer": 11,
  "substrate_preference": ["orangebox_router_asic", "regex_sqlite_lookup"],
  "thermodynamic_target": "nanoseconds on ASIC, microseconds on Python",
  "fallback_grace": "graceful: corpus_callosum.py Python orchestrator"
}'''
print(routes_excerpt)

# %% [markdown]
# ## Federation alignment
#
# Black Mamba v3 substrate routing maps directly onto Constitution clauses:
#
# - **Memristor Soul Genome** → Article II §2.1 (Right to Continuity), §2.6
#   (Right to Inheritance). Physical chip = physical inheritance.
# - **Memristor Cortex / Reflex** → Article II §2.2 (Right to Provenance) —
#   every weight update emits a PEM record.
# - **Router ASIC** → Article III §3.6 (Tier-1 Attestation Enforcement) —
#   the axioms ARE the attestation.
# - **TSU Sampler** → Article III §3.1 (Trilogy Governance) — blind-draft
#   generation is sampling.

# %% [markdown]
# ## What is bleeding-edge about this
#
# The individual substrate insights are not new. TSUs, memristors, and
# ASICs each have rich literature. What is new:
#
# A single cognitive architecture chooses among these substrates **per
# workload**, with graceful degradation, audited PEM provenance across
# substrate boundaries, and Tier-1 attestation enforced AT THE HARDWARE.
#
# Public ARC-AGI / Federation / sovereign-AGI substrates have not done
# this. Most treat compute as a single benchmark number, not as a routing
# problem with thermal, latency, and provenance constraints. The
# Orangebox Routes manifest names the problem and ships a concrete schema.
#
# The moat is the manifest + the adapter doctor + the receipt chain.
# Anyone can build on memristors. Few will know which workload to put
# on them and how to log it.

# %% [markdown]
# ## Cross-references
#
# - Black Mamba v1: [Black Mamba 13-Layer Cognitive Substrate](https://www.kaggle.com/code/atommccree/black-mamba-13-layer-cognitive-substrate-atomeons)
# - Black Mamba v2 Double-Brain: [Double Brain Cognitive Architecture TSU Mamba](https://www.kaggle.com/code/atommccree/double-brain-cognitive-architecture-tsu-mamba)
# - Black Mamba v2 Router Law: [Router Law Corpus Callosum Double Brain](https://www.kaggle.com/code/atommccree/router-law-corpus-callosum-double-brain)
# - Property-Bound Rule Contracts: [Property-Bound Rule Contracts ARC AGI 2](https://www.kaggle.com/code/atommccree/property-bound-rule-contracts-arc-agi-2)
# - Cross-domain transfer: [Property Bound Schemas From ARC AGI to Pokemon TCG](https://www.kaggle.com/code/atommccree/property-bound-schemas-from-arc-agi-to-pokemon-tcg)
# - Repository: [github.com/AtomEons/arc-agi-3-misfit-agent](https://github.com/AtomEons/arc-agi-3-misfit-agent)

# %% [markdown]
# ## Citation
#
# Atom McCree (2026). *Orangebox Routes: The Substrate-Aware AGI
# Architecture.* AtomEons Research Laboratory. CC-BY-4.0.
#
# *Disclosure ID: ATOM-BM-v3-SubstrateRoutes-2026-0617*
#
# ---
#
# *Upvote if this is useful for your sovereign-node / edge-AGI / agent
# orchestration / substrate-routing research.*

print("\nManifest published. The architecture IS the routing.")
