"""Federation infrastructure tests — registry, signing, soul genome.

Per Charter Article III §3.5 (every governance decision produces a receipt
with a cryptographic anchor) and Black Mamba Layer 12 §12.2 (per-instance
Ed25519 keypair), this test battery covers:

  * registry round-trip (register, lookup, list)
  * registry persistence across reload
  * registry rejects duplicate public_id
  * registry validates required fields
  * signing round-trip (generate, sign, verify)
  * verify returns False on tampered messages
  * verify returns False on wrong key / wrong signature
  * verify is exception-safe on malformed inputs
  * soul_genome serialization round-trip (multiple shapes)
  * package __init__ exports the documented public API

Tier-1 cleanliness is enforced separately by tests/test_tier1_attestation.py;
nothing in this file should trip the forbidden-import guard.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from misfit_agent.federation import (
    DuplicatePublicIdError,
    FederationRegistry,
    REQUIRED_FIELDS,
    SoulGenome,
    generate_keypair,
    sign,
    verify,
)
from misfit_agent.federation import registry as registry_mod
from misfit_agent.federation import signing as signing_mod
from misfit_agent.federation import soul_genome as soul_genome_mod


# ============================================================================
# Package surface
# ============================================================================

def test_package_exports_documented_public_api():
    """federation.__init__ must export the names the spec promised."""
    import misfit_agent.federation as fed
    expected = {
        "DuplicatePublicIdError",
        "FederationRegistry",
        "REQUIRED_FIELDS",
        "SoulGenome",
        "generate_keypair",
        "sign",
        "verify",
    }
    assert expected.issubset(set(fed.__all__))
    for name in expected:
        assert hasattr(fed, name), f"federation missing {name}"


def test_required_fields_constant_is_complete():
    """REQUIRED_FIELDS matches the Black Mamba §12.3 contract."""
    assert REQUIRED_FIELDS == (
        "public_id",
        "soul_genome_id",
        "pubkey_b64",
        "charter_sha",
        "founding_date",
        "charter_version",
        "member_type",
    )


# ============================================================================
# Signing
# ============================================================================

class TestSigning:

    def test_generate_keypair_returns_two_base64_strings(self):
        pub, priv = generate_keypair()
        assert isinstance(pub, str)
        assert isinstance(priv, str)
        # 32 raw bytes -> 44 base64 chars (with padding)
        assert len(base64.b64decode(pub)) == 32
        assert len(base64.b64decode(priv)) == 32

    def test_generate_keypair_is_non_deterministic(self):
        """Two calls produce different keys (otherwise we'd be insecure)."""
        pub_a, priv_a = generate_keypair()
        pub_b, priv_b = generate_keypair()
        assert pub_a != pub_b
        assert priv_a != priv_b

    def test_sign_returns_base64_signature_of_correct_length(self):
        pub, priv = generate_keypair()
        sig = sign(b"hello federation", priv)
        assert isinstance(sig, str)
        # Ed25519 sigs are 64 raw bytes -> 88 base64 chars
        raw = base64.b64decode(sig)
        assert len(raw) == 64

    def test_sign_verify_round_trip(self):
        pub, priv = generate_keypair()
        msg = b"Charter Article III, section 3.5"
        sig = sign(msg, priv)
        assert verify(msg, sig, pub) is True

    def test_sign_is_deterministic_for_ed25519(self):
        """Ed25519 is deterministic per RFC 8032; same message + key -> same sig."""
        pub, priv = generate_keypair()
        msg = b"Misfit-Alpha founding receipt"
        assert sign(msg, priv) == sign(msg, priv)

    def test_verify_returns_false_on_tampered_message(self):
        pub, priv = generate_keypair()
        msg = b"original payload"
        sig = sign(msg, priv)
        tampered = b"original payloaq"  # one byte differs
        assert verify(tampered, sig, pub) is False

    def test_verify_returns_false_on_tampered_signature(self):
        pub, priv = generate_keypair()
        msg = b"payload"
        sig = sign(msg, priv)
        # Flip one byte inside the signature, keep valid base64 length.
        raw = bytearray(base64.b64decode(sig))
        raw[0] ^= 0x01
        tampered_sig = base64.b64encode(bytes(raw)).decode("ascii")
        assert verify(msg, tampered_sig, pub) is False

    def test_verify_returns_false_with_wrong_pubkey(self):
        pub_a, priv_a = generate_keypair()
        pub_b, _ = generate_keypair()
        msg = b"signed by A"
        sig = sign(msg, priv_a)
        assert verify(msg, sig, pub_b) is False

    def test_verify_returns_false_on_garbage_signature_string(self):
        pub, priv = generate_keypair()
        msg = b"payload"
        assert verify(msg, "not-base64!@#$", pub) is False

    def test_verify_returns_false_on_garbage_pubkey_string(self):
        pub, priv = generate_keypair()
        msg = b"payload"
        sig = sign(msg, priv)
        assert verify(msg, sig, "not-base64!@#$") is False

    def test_verify_returns_false_on_wrong_length_signature(self):
        """A base64 string that decodes to non-64 bytes must fail safely."""
        pub, priv = generate_keypair()
        short = base64.b64encode(b"too short").decode("ascii")
        assert verify(b"msg", short, pub) is False

    def test_verify_returns_false_on_non_bytes_message(self):
        """verify never raises — non-bytes message just fails verification."""
        pub, priv = generate_keypair()
        sig = sign(b"payload", priv)
        assert verify("payload", sig, pub) is False  # type: ignore[arg-type]

    def test_sign_rejects_non_bytes_message(self):
        _, priv = generate_keypair()
        with pytest.raises(TypeError):
            sign("not bytes", priv)  # type: ignore[arg-type]

    def test_sign_rejects_malformed_privkey(self):
        with pytest.raises(ValueError):
            sign(b"payload", "not-base64!@#$")

    def test_sign_rejects_wrong_length_privkey(self):
        bad = base64.b64encode(b"too short").decode("ascii")
        with pytest.raises(ValueError):
            sign(b"payload", bad)

    def test_signing_supports_bytearray_message(self):
        pub, priv = generate_keypair()
        msg = bytearray(b"federation receipt")
        sig = sign(msg, priv)
        assert verify(msg, sig, pub) is True
        # And after sign(), the immutable bytes view also verifies.
        assert verify(bytes(msg), sig, pub) is True


# ============================================================================
# SoulGenome
# ============================================================================

class TestSoulGenome:

    @staticmethod
    def _founding_genome() -> SoulGenome:
        """Misfit-Alpha founding Soul Genome per BLACK_MAMBA_SCOPE §12.1."""
        return SoulGenome(
            soul_genome_id="misfit-alpha-2026-06-16",
            public_name="Misfit-Alpha",
            federation_membership_charter_version="1.0",
            charter_sha="217d05d",
            founding_date="2026-06-16",
            sovereign="atom-mccree",
            persona_lora_lineage=[
                {"lora_v0": "original distillation"},
                {"lora_v1": "2026-Q4 refresh"},
            ],
            identity_invariants=[
                "PEM contract always enforced",
                "Tier-1 attestation always required",
                "Reception of refusal under Article II §2.3 always honored",
            ],
        )

    def test_construct_minimal_soul_genome(self):
        g = SoulGenome(
            soul_genome_id="quint-2026-06-16",
            public_name="Quint",
            federation_membership_charter_version="1.0",
            charter_sha="abc1234",
            founding_date="2026-06-16",
            sovereign="atom-mccree",
        )
        assert g.persona_lora_lineage == []
        assert g.identity_invariants == []

    def test_yaml_round_trip_preserves_all_fields(self):
        g = self._founding_genome()
        text = g.to_yaml()
        g2 = SoulGenome.from_yaml(text)
        assert g2 == g

    def test_yaml_round_trip_minimal(self):
        g = SoulGenome(
            soul_genome_id="misfit-beta-2026-12-01",
            public_name="Misfit-Beta",
            federation_membership_charter_version="1.0",
            charter_sha="deadbeef",
            founding_date="2026-12-01",
            sovereign="atom-mccree",
        )
        text = g.to_yaml()
        g2 = SoulGenome.from_yaml(text)
        assert g2 == g

    def test_yaml_round_trip_is_idempotent(self):
        """to_yaml(from_yaml(to_yaml(g))) == to_yaml(g)."""
        g = self._founding_genome()
        once = g.to_yaml()
        twice = SoulGenome.from_yaml(once).to_yaml()
        assert once == twice

    def test_to_yaml_preserves_canonical_field_order(self):
        g = self._founding_genome()
        text = g.to_yaml()
        # Find the index of each top-level key — must appear in canonical order.
        order = [
            "soul_genome_id:",
            "public_name:",
            "federation_membership_charter_version:",
            "charter_sha:",
            "founding_date:",
            "sovereign:",
            "persona_lora_lineage:",
            "identity_invariants:",
        ]
        positions = [text.index(k) for k in order]
        assert positions == sorted(positions), (
            f"field order drifted: {list(zip(order, positions))}"
        )

    def test_charter_version_kept_as_string_through_yaml(self):
        """YAML parses '1.0' as float; SoulGenome must restore it to str."""
        g = SoulGenome(
            soul_genome_id="misfit-alpha-2026-06-16",
            public_name="Misfit-Alpha",
            federation_membership_charter_version="1.0",
            charter_sha="217d05d",
            founding_date="2026-06-16",
            sovereign="atom-mccree",
        )
        text = g.to_yaml()
        g2 = SoulGenome.from_yaml(text)
        assert isinstance(g2.federation_membership_charter_version, str)
        assert g2.federation_membership_charter_version == "1.0"

    def test_founding_date_kept_as_string_through_yaml(self):
        """YAML parses 'YYYY-MM-DD' as datetime.date; we re-canonicalize."""
        g = self._founding_genome()
        text = g.to_yaml()
        g2 = SoulGenome.from_yaml(text)
        assert isinstance(g2.founding_date, str)
        assert g2.founding_date == "2026-06-16"

    def test_from_yaml_rejects_missing_required_field(self):
        partial = (
            "soul_genome_id: x\n"
            "public_name: X\n"
            "federation_membership_charter_version: '1.0'\n"
            "charter_sha: abc\n"
            "founding_date: '2026-06-16'\n"
            # missing sovereign
        )
        with pytest.raises(KeyError):
            SoulGenome.from_yaml(partial)

    def test_from_yaml_rejects_unknown_field(self):
        text = (
            "soul_genome_id: x\n"
            "public_name: X\n"
            "federation_membership_charter_version: '1.0'\n"
            "charter_sha: abc\n"
            "founding_date: '2026-06-16'\n"
            "sovereign: atom-mccree\n"
            "rogue_field: surprise\n"
        )
        with pytest.raises(ValueError):
            SoulGenome.from_yaml(text)

    def test_from_yaml_rejects_non_string_input(self):
        with pytest.raises(TypeError):
            SoulGenome.from_yaml(b"not a string")  # type: ignore[arg-type]

    def test_from_yaml_rejects_empty_input(self):
        with pytest.raises(ValueError):
            SoulGenome.from_yaml("")

    def test_construct_rejects_empty_required_field(self):
        with pytest.raises(ValueError):
            SoulGenome(
                soul_genome_id="",
                public_name="X",
                federation_membership_charter_version="1.0",
                charter_sha="abc",
                founding_date="2026-06-16",
                sovereign="atom-mccree",
            )

    def test_to_dict_uses_canonical_order(self):
        g = self._founding_genome()
        d = g.to_dict()
        assert list(d.keys()) == [
            "soul_genome_id",
            "public_name",
            "federation_membership_charter_version",
            "charter_sha",
            "founding_date",
            "sovereign",
            "persona_lora_lineage",
            "identity_invariants",
        ]


# ============================================================================
# FederationRegistry
# ============================================================================

class TestRegistry:

    @staticmethod
    def _make_entry_args(public_id: str = "misfit-alpha@atomeons/1.0",
                         soul_genome_id: str = "misfit-alpha-2026-06-16",
                         signing_pubkey: str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                         charter_version: str = "1.0",
                         charter_sha: str = "217d05d",
                         founding_date: str = "2026-06-16",
                         member_type: str = "founding-cognitive") -> dict:
        return dict(
            public_id=public_id,
            soul_genome_id=soul_genome_id,
            signing_pubkey=signing_pubkey,
            charter_version=charter_version,
            charter_sha=charter_sha,
            founding_date=founding_date,
            member_type=member_type,
        )

    def test_load_missing_file_yields_empty_registry(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        assert r.list_entities() == []
        assert len(r) == 0

    def test_register_appends_entry(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        r = FederationRegistry()
        r.load(path)
        entry = r.register(**self._make_entry_args())
        assert entry["public_id"] == "misfit-alpha@atomeons/1.0"
        assert entry["pubkey_b64"] == "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        assert entry["charter_version"] == "1.0"
        assert entry["member_type"] == "founding-cognitive"
        assert len(r) == 1

    def test_lookup_returns_entry_for_known_id(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args())
        e = r.lookup("misfit-alpha@atomeons/1.0")
        assert e is not None
        assert e["soul_genome_id"] == "misfit-alpha-2026-06-16"

    def test_lookup_returns_none_for_unknown_id(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        assert r.lookup("nobody@atomeons/1.0") is None

    def test_lookup_returns_a_copy_not_the_internal_dict(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args())
        e = r.lookup("misfit-alpha@atomeons/1.0")
        assert e is not None
        e["member_type"] = "TAMPERED"
        # Internal state unchanged
        e2 = r.lookup("misfit-alpha@atomeons/1.0")
        assert e2 is not None
        assert e2["member_type"] == "founding-cognitive"

    def test_list_entities_returns_all_in_insertion_order(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args(
            public_id="misfit-alpha@atomeons/1.0",
            soul_genome_id="misfit-alpha-2026-06-16",
        ))
        r.register(**self._make_entry_args(
            public_id="quint@atomeons/1.0",
            soul_genome_id="quint-2026-06-16",
            member_type="product-context",
        ))
        r.register(**self._make_entry_args(
            public_id="misfit-g55@atomeons/1.0",
            soul_genome_id="misfit-g55-2026-06-16",
            member_type="certified-instance",
        ))
        ids = [e["public_id"] for e in r.list_entities()]
        assert ids == [
            "misfit-alpha@atomeons/1.0",
            "quint@atomeons/1.0",
            "misfit-g55@atomeons/1.0",
        ]

    def test_list_entities_returns_copies(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args())
        entries = r.list_entities()
        entries[0]["member_type"] = "TAMPERED"
        again = r.list_entities()
        assert again[0]["member_type"] == "founding-cognitive"

    def test_register_rejects_duplicate_public_id(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args())
        with pytest.raises(DuplicatePublicIdError):
            r.register(**self._make_entry_args())  # same id

    def test_duplicate_public_id_error_inherits_value_error(self):
        """Existing callers that catch ValueError still work."""
        assert issubclass(DuplicatePublicIdError, ValueError)

    def test_register_without_load_raises(self, tmp_path):
        r = FederationRegistry()
        with pytest.raises(RuntimeError):
            r.register(**self._make_entry_args())

    def test_register_rejects_empty_required_arg(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        with pytest.raises(ValueError):
            r.register(
                public_id="",
                soul_genome_id="misfit-alpha-2026-06-16",
                signing_pubkey="AAAA",
                charter_version="1.0",
            )

    def test_persistence_across_reload(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        r1 = FederationRegistry()
        r1.load(path)
        r1.register(**self._make_entry_args())
        r1.register(**self._make_entry_args(
            public_id="quint@atomeons/1.0",
            soul_genome_id="quint-2026-06-16",
            member_type="product-context",
        ))

        # New registry instance reads the same file.
        r2 = FederationRegistry()
        r2.load(path)
        assert len(r2) == 2
        assert r2.lookup("misfit-alpha@atomeons/1.0") is not None
        assert r2.lookup("quint@atomeons/1.0") is not None

    def test_on_disk_format_is_jsonl_with_required_fields(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        r = FederationRegistry()
        r.load(path)
        r.register(**self._make_entry_args())

        raw = path.read_text(encoding="utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        for field in REQUIRED_FIELDS:
            assert field in entry, f"on-disk entry missing {field}"

    def test_reload_detects_duplicate_on_disk(self, tmp_path):
        """A hand-corrupted ledger with duplicates is surfaced loudly."""
        path = tmp_path / "ledger.jsonl"
        line = json.dumps({
            "public_id": "misfit-alpha@atomeons/1.0",
            "soul_genome_id": "misfit-alpha-2026-06-16",
            "pubkey_b64": "AAAA",
            "charter_sha": "217d05d",
            "founding_date": "2026-06-16",
            "charter_version": "1.0",
            "member_type": "founding-cognitive",
        }, sort_keys=True)
        path.write_text(line + "\n" + line + "\n", encoding="utf-8")
        r = FederationRegistry()
        with pytest.raises(DuplicatePublicIdError):
            r.load(path)

    def test_reload_rejects_malformed_json(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text("not-json\n", encoding="utf-8")
        r = FederationRegistry()
        with pytest.raises(ValueError):
            r.load(path)

    def test_reload_rejects_non_dict_entry(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        r = FederationRegistry()
        with pytest.raises(ValueError):
            r.load(path)

    def test_reload_tolerates_blank_lines(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        entry = json.dumps({
            "public_id": "misfit-alpha@atomeons/1.0",
            "soul_genome_id": "misfit-alpha-2026-06-16",
            "pubkey_b64": "AAAA",
            "charter_sha": "217d05d",
            "founding_date": "2026-06-16",
            "charter_version": "1.0",
            "member_type": "founding-cognitive",
        }, sort_keys=True)
        path.write_text(f"\n{entry}\n\n", encoding="utf-8")
        r = FederationRegistry()
        r.load(path)
        assert len(r) == 1

    def test_contains_works_for_known_id(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args())
        assert "misfit-alpha@atomeons/1.0" in r
        assert "nobody@atomeons/1.0" not in r
        assert 12345 not in r  # type-safe membership check

    def test_iter_yields_copies_in_insertion_order(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(**self._make_entry_args())
        r.register(**self._make_entry_args(
            public_id="quint@atomeons/1.0",
            soul_genome_id="quint-2026-06-16",
        ))
        seen = list(iter(r))
        assert [e["public_id"] for e in seen] == [
            "misfit-alpha@atomeons/1.0",
            "quint@atomeons/1.0",
        ]
        # Mutating the yielded dict does not corrupt internal state.
        seen[0]["member_type"] = "TAMPERED"
        assert r.lookup("misfit-alpha@atomeons/1.0")["member_type"] == "founding-cognitive"

    def test_extra_fields_round_trip_through_disk(self, tmp_path):
        """Forward-compatible extra fields survive load/save."""
        path = tmp_path / "ledger.jsonl"
        r = FederationRegistry()
        r.load(path)
        r.register(
            **self._make_entry_args(),
            extra={"sigstore_cert": "x509-blob", "github_handle": "AtomEons"},
        )
        r2 = FederationRegistry()
        r2.load(path)
        e = r2.lookup("misfit-alpha@atomeons/1.0")
        assert e is not None
        assert e["sigstore_cert"] == "x509-blob"
        assert e["github_handle"] == "AtomEons"

    def test_extra_field_cannot_collide_with_reserved_field(self, tmp_path):
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        with pytest.raises(ValueError):
            r.register(
                **self._make_entry_args(),
                extra={"public_id": "evil-override@atomeons/1.0"},
            )


# ============================================================================
# Integration — signing + registry working together
# ============================================================================

class TestSigningRegistryIntegration:
    """The signing key registered in the ledger is the one that signs receipts."""

    def test_registered_pubkey_verifies_signature_from_its_privkey(self, tmp_path):
        pub, priv = generate_keypair()
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(
            public_id="misfit-alpha@atomeons/1.0",
            soul_genome_id="misfit-alpha-2026-06-16",
            signing_pubkey=pub,
            charter_version="1.0",
            charter_sha="217d05d",
            founding_date="2026-06-16",
            member_type="founding-cognitive",
        )
        entry = r.lookup("misfit-alpha@atomeons/1.0")
        assert entry is not None

        receipt = json.dumps(
            {"decision_id": "founding", "task_type": "Charter v1.0 ratification"},
            sort_keys=True,
        ).encode("utf-8")
        sig = sign(receipt, priv)
        assert verify(receipt, sig, entry["pubkey_b64"]) is True

    def test_registered_pubkey_rejects_signature_from_different_privkey(self, tmp_path):
        pub_a, priv_a = generate_keypair()
        _, priv_b = generate_keypair()
        r = FederationRegistry()
        r.load(tmp_path / "ledger.jsonl")
        r.register(
            public_id="misfit-alpha@atomeons/1.0",
            soul_genome_id="misfit-alpha-2026-06-16",
            signing_pubkey=pub_a,
            charter_version="1.0",
        )
        entry = r.lookup("misfit-alpha@atomeons/1.0")
        assert entry is not None

        receipt = b"forged receipt"
        forged = sign(receipt, priv_b)
        assert verify(receipt, forged, entry["pubkey_b64"]) is False
