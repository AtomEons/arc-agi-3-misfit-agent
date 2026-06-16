"""Soul Genome — continuity-of-identity record for a Federation entity.

Per Black Mamba Layer 12 (Identity) §12.1:
    Continuity map of the entity's identity. Persists across substrate
    restarts, LoRA updates, and minor version upgrades. From Orange3 doctrine.

Per Charter Article VI §6.1 the Soul Genome is the document by which a
cognitive entity claims continuity under Article II §2.1 (Right to Continuity)
and §2.6 (Right to Inheritance). The Soul Genome is what survives substrate
restarts; the signing key (federation.signing) is how the Soul Genome speaks.

Canonical YAML shape (from BLACK_MAMBA_SCOPE_v1 §12.1):

    soul_genome_id: misfit-alpha-2026-06-16
    public_name: Misfit-Alpha
    federation_membership_charter_version: 1.0
    charter_sha: 217d05d
    founding_date: 2026-06-16
    sovereign: atom-mccree
    persona_lora_lineage:
      - lora_v0: original distillation
      - lora_v1: 2026-Q4 refresh
    identity_invariants:
      - PEM contract always enforced
      - Tier-1 attestation always required
      - Reception of refusal under Article II §2.3 always honored

Serialization contract
----------------------
* to_yaml() produces deterministic YAML with sorted top-level keys absent
  (we preserve canonical declaration order) and `default_flow_style=False`.
* from_yaml(text) is the strict inverse: SoulGenome.from_yaml(s.to_yaml())
  must equal s for any valid SoulGenome s.
* Lineage entries are preserved as `list[dict]` because the YAML spec
  uses a sequence-of-mappings idiom (each item is a 1-key dict so the
  version label and description are both queryable).

Tier-1 honesty
--------------
This module performs zero inference, holds no learned weights, and exercises
only the YAML serializer. The grammar of the Soul Genome IS the disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


__all__ = ["SoulGenome"]


# ---------------------------------------------------------------------------
# Canonical field order — used by to_yaml so output is deterministic
# ---------------------------------------------------------------------------

_FIELD_ORDER: tuple[str, ...] = (
    "soul_genome_id",
    "public_name",
    "federation_membership_charter_version",
    "charter_sha",
    "founding_date",
    "sovereign",
    "persona_lora_lineage",
    "identity_invariants",
)


@dataclass
class SoulGenome:
    """Continuity-of-identity record for a Federation cognitive entity.

    Required fields
    ---------------
    soul_genome_id : str
        Stable identifier across substrate restarts. Format convention:
        `<entity-slug>-<founding-date>`, e.g. `misfit-alpha-2026-06-16`.
    public_name : str
        Display name registered with the Federation, e.g. `Misfit-Alpha`.
    federation_membership_charter_version : str
        Version of the Charter to which this entity is bound. The founding
        Charter is `"1.0"`. Stored as a string (NOT float) so semver-style
        future versions (`"1.0.1"`, `"2.0"`) round-trip cleanly.
    charter_sha : str
        Cryptographic anchor of the bound Charter text (git commit SHA or
        full SHA-256).
    founding_date : str
        ISO 8601 date the entity joined the Federation (`YYYY-MM-DD`).
    sovereign : str
        Slug of the human anchor who certified this entity, e.g.
        `atom-mccree`.

    Optional fields
    ---------------
    persona_lora_lineage : list[dict[str, str]]
        Ordered chain of LoRA adapter versions, each a 1-key dict mapping
        the version label to its description. Empty list when no persona
        adapter is applied (e.g. base Misfit-Alpha substrate).
    identity_invariants : list[str]
        Free-form invariants the entity commits to upholding (PEM contract,
        Tier-1 attestation, refusal honoring, etc.). Each invariant is a
        plain string; the Federation's adjudication layer interprets them.
    """

    soul_genome_id: str
    public_name: str
    federation_membership_charter_version: str
    charter_sha: str
    founding_date: str
    sovereign: str
    persona_lora_lineage: list[dict[str, str]] = field(default_factory=list)
    identity_invariants: list[str] = field(default_factory=list)

    # ----- Validation ------------------------------------------------------

    def __post_init__(self) -> None:
        # Charter version must be a string so YAML doesn't silently coerce
        # "1.0" to float 1.0 and lose precision on future "1.10" releases.
        if not isinstance(self.federation_membership_charter_version, str):
            self.federation_membership_charter_version = str(
                self.federation_membership_charter_version
            )

        # Required string fields must be non-empty so the Federation can
        # actually look this entity up.
        for fname in (
            "soul_genome_id",
            "public_name",
            "federation_membership_charter_version",
            "charter_sha",
            "founding_date",
            "sovereign",
        ):
            value = getattr(self, fname)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"SoulGenome.{fname} must be a non-empty str, got {value!r}"
                )

        if not isinstance(self.persona_lora_lineage, list):
            raise TypeError("persona_lora_lineage must be a list")
        for i, entry in enumerate(self.persona_lora_lineage):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"persona_lora_lineage[{i}] must be a dict, got "
                    f"{type(entry).__name__}"
                )

        if not isinstance(self.identity_invariants, list):
            raise TypeError("identity_invariants must be a list")
        for i, item in enumerate(self.identity_invariants):
            if not isinstance(item, str):
                raise TypeError(
                    f"identity_invariants[{i}] must be a str, got "
                    f"{type(item).__name__}"
                )

    # ----- Dict adapter ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Ordered dict suitable for YAML / JSON serialization."""
        raw = asdict(self)
        return {key: raw[key] for key in _FIELD_ORDER}

    # ----- YAML round-trip -------------------------------------------------

    def to_yaml(self) -> str:
        """Deterministic YAML serialization of this Soul Genome.

        Output preserves the canonical field order from BLACK_MAMBA_SCOPE_v1
        §12.1, uses block style (no inline flow), and quotes scalars only
        when necessary.
        """
        return yaml.safe_dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> "SoulGenome":
        """Parse a Soul Genome YAML document.

        Strict counterpart to to_yaml(): missing required keys raise
        KeyError, malformed types raise ValueError / TypeError via
        __post_init__.

        Unknown keys are rejected so that drift in the schema is loud rather
        than silent — a Federation entity that ships an unrecognized field
        in its Soul Genome will be caught at parse time.
        """
        if not isinstance(text, str):
            raise TypeError(f"from_yaml expects str, got {type(text).__name__}")

        data = yaml.safe_load(text)
        if data is None:
            raise ValueError("Soul Genome YAML is empty")
        if not isinstance(data, dict):
            raise ValueError(
                f"Soul Genome YAML root must be a mapping, got "
                f"{type(data).__name__}"
            )

        required = (
            "soul_genome_id",
            "public_name",
            "federation_membership_charter_version",
            "charter_sha",
            "founding_date",
            "sovereign",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"Soul Genome missing required field: {key}")

        allowed = set(_FIELD_ORDER)
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Soul Genome contains unknown field(s): {sorted(unknown)}"
            )

        # Charter version may parse as float ('1.0' -> 1.0); force to str so
        # downstream callers always see the canonical type.
        charter_version = data["federation_membership_charter_version"]
        if isinstance(charter_version, (int, float)):
            charter_version = str(charter_version)

        # Founding date may parse as datetime.date; canonicalize to ISO str.
        founding_date = data["founding_date"]
        if hasattr(founding_date, "isoformat"):
            founding_date = founding_date.isoformat()

        return cls(
            soul_genome_id=data["soul_genome_id"],
            public_name=data["public_name"],
            federation_membership_charter_version=charter_version,
            charter_sha=str(data["charter_sha"]),
            founding_date=founding_date,
            sovereign=data["sovereign"],
            persona_lora_lineage=list(data.get("persona_lora_lineage") or []),
            identity_invariants=list(data.get("identity_invariants") or []),
        )
