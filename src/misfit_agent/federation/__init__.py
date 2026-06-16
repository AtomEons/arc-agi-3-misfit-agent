"""AtomEons Federation infrastructure — registry, signing, identity.

This package implements the operational substrate for Federation membership
per the AtomEons Federation Charter v1.0 (docs/CHARTER_v1.md) and Black
Mamba Layer 12 — Identity (docs/BLACK_MAMBA_SCOPE_v1.md §12).

Modules
-------
signing
    Ed25519 keypair generation and base64-encoded sign / verify primitives.
    Backs receipt-log signatures (Article III §3.5), CHSG vote signatures
    (Article III §3.1), and cross-Federation cross-examination authentication
    (Black Mamba §13.2).

soul_genome
    SoulGenome dataclass — continuity-of-identity record that survives
    substrate restarts, LoRA updates, and minor version upgrades. Serializes
    to/from the canonical YAML form documented in BLACK_MAMBA_SCOPE_v1 §12.1.

registry
    FederationRegistry — append-only JSONL ledger of certified entities,
    enforcing public_id uniqueness while preserving the constitutional
    indelibility required by Article II §2.2.

Tier-1 honesty
--------------
Every module in this package is mechanically Tier-1 clean: zero LLM imports,
zero learned parameters, zero pretrained weights. The signing layer is pure
Ed25519 (RFC 8032 via the `cryptography` package — a pure cryptography
binding, not a machine-learning library). The registry is plain JSONL. The
Soul Genome is plain YAML. The grammar IS the disclosure.
"""

from __future__ import annotations

from .registry import (
    DuplicatePublicIdError,
    FederationRegistry,
    REQUIRED_FIELDS,
)
from .signing import generate_keypair, sign, verify
from .soul_genome import SoulGenome


__all__ = [
    "DuplicatePublicIdError",
    "FederationRegistry",
    "REQUIRED_FIELDS",
    "SoulGenome",
    "generate_keypair",
    "sign",
    "verify",
]
