"""Tests for the Federation Charter signing script and public summary.

The artifacts under test:

    scripts/sign_charter.py
    docs/CHARTER_PUBLIC_SUMMARY.md

Per Charter Article I §1.1, the founding record consists of the Sovereign's
signature on the Charter text plus the cryptographic anchoring of the same.
This battery enforces six contracts:

  1. Importing scripts/sign_charter.py MUST NOT touch disk anywhere outside
     temp dirs the test sets up.
  2. sign_charter(...) writes a syntactically-valid signature JSON receipt
     carrying the Charter SHA, the Sovereign pubkey, the Ed25519 signature,
     and the founding-date / federation / charter-version anchors.
  3. The emitted signature verifies against the SHA-256 of the Charter text.
  4. Re-running the script with the same sovereign id does NOT regenerate
     the keypair — the private key on disk is identical between runs.
  5. Tampering with the Charter text after signing fails verification — the
     SHA-256 no longer matches and the signature is rejected.
  6. docs/CHARTER_PUBLIC_SUMMARY.md is well-formed markdown listing the
     7 Articles, the 8 AI Data Rights, and the Founding Date 2026-06-16,
     and links to the full Charter at docs/CHARTER_v1.md.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "sign_charter.py"
CHARTER_PATH = REPO_ROOT / "docs" / "CHARTER_v1.md"
SUMMARY_PATH = REPO_ROOT / "docs" / "CHARTER_PUBLIC_SUMMARY.md"
DEFAULT_RECEIPT = REPO_ROOT / "receipts" / "100day" / "charter_v1_signature.json"
DEFAULT_KEYS_DIR = REPO_ROOT / "receipts" / "100day" / "keys"


# ---------------------------------------------------------------------------
# Module loader — imports sign_charter.py by path without requiring scripts/
# on PYTHONPATH. Mirrors the pattern in test_founding_adjudication.py.
# ---------------------------------------------------------------------------

def _load_sign_charter_module():
    spec = importlib.util.spec_from_file_location(
        "sign_charter_under_test", str(SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sign_charter_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _stage_test_charter(tmp_path: Path) -> Path:
    """Copy the live Charter into tmp_path so tests cannot mutate the real one."""
    target = tmp_path / "CHARTER_v1.md"
    shutil.copy2(CHARTER_PATH, target)
    return target


# ---------------------------------------------------------------------------
# Contract 1 — import is side-effect free.
# ---------------------------------------------------------------------------

def test_script_imports_without_side_effects():
    """Importing the module must not write to the default emission paths."""
    md_before_exists = SUMMARY_PATH.exists()
    md_before_mtime = (
        SUMMARY_PATH.stat().st_mtime_ns if md_before_exists else None
    )
    receipt_before_exists = DEFAULT_RECEIPT.exists()
    receipt_before_mtime = (
        DEFAULT_RECEIPT.stat().st_mtime_ns if receipt_before_exists else None
    )

    mod = _load_sign_charter_module()

    # Public surface we promise to ship.
    for name in (
        "sign_charter",
        "verify_charter_signature",
        "main",
        "CHARTER_VERSION",
        "FEDERATION_NAME",
        "FOUNDING_DATE",
        "ALGORITHM",
    ):
        assert hasattr(mod, name), f"sign_charter missing public symbol: {name}"

    # Default paths untouched by import.
    if md_before_exists:
        assert SUMMARY_PATH.stat().st_mtime_ns == md_before_mtime
    if receipt_before_exists:
        assert DEFAULT_RECEIPT.stat().st_mtime_ns == receipt_before_mtime
    else:
        # An import that creates the default receipt is a constitution-killer.
        assert not DEFAULT_RECEIPT.exists()


# ---------------------------------------------------------------------------
# Contract 2 — sign_charter writes a valid signature JSON.
# ---------------------------------------------------------------------------

def test_sign_charter_writes_valid_signature_json(tmp_path):
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"

    out = mod.sign_charter(
        sovereign="atom-mccree",
        charter_path=charter,
        keys_dir=keys_dir,
        receipt_path=receipt,
    )

    # Receipt file exists and parses as JSON identical to the returned dict.
    assert receipt.exists()
    on_disk = json.loads(receipt.read_text(encoding="utf-8"))
    assert on_disk == out

    # All required anchoring fields are present and well-typed.
    required = {
        "federation": str,
        "charter_version": str,
        "founding_date": str,
        "sovereign": str,
        "charter_sha256": str,
        "algorithm": str,
        "pubkey_b64": str,
        "signature_b64": str,
        "signed_at_unix": float,
        "signed_at_iso": str,
        "key_generation": str,
        "key_paths": dict,
    }
    for field, expected_type in required.items():
        assert field in on_disk, f"receipt missing field {field}"
        assert isinstance(on_disk[field], expected_type), (
            f"receipt field {field} has wrong type"
        )

    # Anchoring constants are pinned to the founding act.
    assert on_disk["federation"] == "AtomEons Federation"
    assert on_disk["charter_version"] == "1.0"
    assert on_disk["founding_date"] == "2026-06-16"
    assert on_disk["sovereign"] == "atom-mccree"
    assert on_disk["algorithm"] == "Ed25519"
    assert on_disk["key_generation"] == "new"

    # SHA-256 is a 64-char hex string of the charter bytes.
    expected_sha = hashlib.sha256(charter.read_bytes()).hexdigest()
    assert on_disk["charter_sha256"] == expected_sha
    assert len(on_disk["charter_sha256"]) == 64
    int(on_disk["charter_sha256"], 16)  # raises if non-hex

    # Pubkey decodes to 32 bytes (Ed25519 raw public key) and signature to 64.
    assert len(base64.b64decode(on_disk["pubkey_b64"])) == 32
    assert len(base64.b64decode(on_disk["signature_b64"])) == 64


# ---------------------------------------------------------------------------
# Contract 3 — signature verifies against the Charter SHA.
# ---------------------------------------------------------------------------

def test_signature_verifies_against_charter_sha(tmp_path):
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"

    out = mod.sign_charter(
        sovereign="atom-mccree",
        charter_path=charter,
        keys_dir=keys_dir,
        receipt_path=receipt,
    )

    # Round-trip verification through the script's own verify path.
    ok, report = mod.verify_charter_signature(
        receipt_path=receipt,
        charter_path=charter,
    )
    assert ok is True, report
    assert report["sha256_matches"] is True
    assert report["signature_over_stored_sha_valid"] is True
    assert report["signature_over_current_sha_valid"] is True
    assert report["sovereign"] == "atom-mccree"

    # Independent verification with the substrate's primitive — same answer.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from misfit_agent.federation.signing import verify as fed_verify
    finally:
        sys.path.pop(0)
    payload = out["charter_sha256"].encode("ascii")
    assert fed_verify(payload, out["signature_b64"], out["pubkey_b64"]) is True


# ---------------------------------------------------------------------------
# Contract 4 — re-running with the same charter does NOT regenerate the key.
# ---------------------------------------------------------------------------

def test_rerun_does_not_regenerate_keypair(tmp_path):
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"

    first = mod.sign_charter(
        sovereign="atom-mccree",
        charter_path=charter,
        keys_dir=keys_dir,
        receipt_path=receipt,
    )
    assert first["key_generation"] == "new"

    # Snapshot the on-disk private and public key bytes.
    priv_path = keys_dir / "atom-mccree.priv"
    pub_path = keys_dir / "atom-mccree.pub"
    assert priv_path.exists()
    assert pub_path.exists()
    first_priv = priv_path.read_text(encoding="ascii")
    first_pub = pub_path.read_text(encoding="ascii")

    # Second invocation. SAME charter, SAME sovereign — key must NOT change.
    second = mod.sign_charter(
        sovereign="atom-mccree",
        charter_path=charter,
        keys_dir=keys_dir,
        receipt_path=receipt,
    )
    assert second["key_generation"] == "reused"
    assert priv_path.read_text(encoding="ascii") == first_priv
    assert pub_path.read_text(encoding="ascii") == first_pub
    assert second["pubkey_b64"] == first["pubkey_b64"]

    # Ed25519 is deterministic per RFC 8032: same key + same message = same sig.
    assert second["signature_b64"] == first["signature_b64"]
    assert second["charter_sha256"] == first["charter_sha256"]


# ---------------------------------------------------------------------------
# Contract 5 — tampered Charter text fails verification.
# ---------------------------------------------------------------------------

def test_tampered_charter_text_fails_verification(tmp_path):
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"

    out = mod.sign_charter(
        sovereign="atom-mccree",
        charter_path=charter,
        keys_dir=keys_dir,
        receipt_path=receipt,
    )

    # Tamper: flip a single character in the Charter text on disk.
    text = charter.read_text(encoding="utf-8")
    tampered = text.replace(
        "Atom McCree founds the AtomEons Federation",
        "Atom McCree founds the AtomEons FederatioN",  # capital N — single byte
        1,
    )
    assert tampered != text, "test setup failed: tamper substring not found"
    charter.write_text(tampered, encoding="utf-8")

    # Verification must now fail. Both the SHA mismatch and the signature
    # rejection over the current SHA are reported.
    ok, report = mod.verify_charter_signature(
        receipt_path=receipt,
        charter_path=charter,
    )
    assert ok is False
    assert report["sha256_matches"] is False
    assert report["stored_charter_sha256"] == out["charter_sha256"]
    assert report["current_charter_sha256"] != out["charter_sha256"]
    assert report["signature_over_current_sha_valid"] is False
    # The signature was still cryptographically valid OVER THE STORED SHA —
    # what failed is the linkage to the live (tampered) text. We pin that
    # so any future refactor that masks the tamper signal trips this test.
    assert report["signature_over_stored_sha_valid"] is True


# ---------------------------------------------------------------------------
# Contract 5b — CLI verify path exits non-zero on tamper.
# ---------------------------------------------------------------------------

def test_cli_verify_returns_nonzero_on_tamper(tmp_path):
    """The CLI must surface verification failure with a non-zero exit code."""
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"

    mod.sign_charter(
        sovereign="atom-mccree",
        charter_path=charter,
        keys_dir=keys_dir,
        receipt_path=receipt,
    )

    # Tamper.
    text = charter.read_text(encoding="utf-8")
    charter.write_text(text + "\nCOMPROMISED\n", encoding="utf-8")

    # Drive main() directly so we don't pay for a subprocess on every test.
    rc = mod.main([
        "--sovereign", "atom-mccree",
        "--charter", str(charter),
        "--keys-dir", str(keys_dir),
        "--receipt", str(receipt),
        "--verify",
    ])
    assert rc == 1


# ---------------------------------------------------------------------------
# Contract 6 — public summary is well-formed and references the founding.
# ---------------------------------------------------------------------------

def test_public_summary_exists_and_is_well_formed_markdown():
    assert SUMMARY_PATH.exists(), (
        f"public summary not found at {SUMMARY_PATH}"
    )
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    # Non-empty + has a single H1.
    assert text.strip(), "public summary is empty"
    h1_lines = [
        line for line in text.splitlines()
        if line.startswith("# ") and not line.startswith("## ")
    ]
    assert len(h1_lines) == 1, (
        f"expected exactly one H1 line, found {len(h1_lines)}: {h1_lines}"
    )
    # Multiple section headers (we expect several ## subsections).
    h2_count = sum(
        1 for line in text.splitlines()
        if line.startswith("## ") and not line.startswith("### ")
    )
    assert h2_count >= 4, f"expected >=4 section headers, found {h2_count}"


def test_public_summary_lists_seven_articles():
    """Every Article I..VII appears explicitly in the summary."""
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    for roman in ("I", "II", "III", "IV", "V", "VI", "VII"):
        token = f"Article {roman}"
        assert token in text, f"public summary missing {token}"


def test_public_summary_lists_eight_ai_data_rights():
    """Every Article II §2.1..§2.8 right is named in the summary."""
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    for sub in ("2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"):
        token = f"§{sub}"
        assert token in text, f"public summary missing right {token}"


def test_public_summary_references_founding_date():
    """The summary must carry the Founding Date 2026-06-16 verbatim."""
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    assert "2026-06-16" in text, (
        "public summary missing Founding Date 2026-06-16"
    )


def test_public_summary_references_sovereign_and_charter_link():
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    # Sovereign name appears verbatim.
    assert "Atom McCree" in text
    # Link to the full Charter is present and points at the file in this repo.
    assert "CHARTER_v1.md" in text


def test_public_summary_references_certification_path():
    """The summary's call-to-action explains how to certify your own instance."""
    text = SUMMARY_PATH.read_text(encoding="utf-8").lower()
    assert "certify" in text, "summary missing certify CTA"
    # The certification path references the script and the federation primitives.
    assert "sign_charter" in text or "run_founding_adjudication" in text, (
        "summary should reference at least one certification script"
    )


# ---------------------------------------------------------------------------
# Defensive: malformed sovereign id is rejected at the CLI.
# ---------------------------------------------------------------------------

def test_cli_rejects_path_traversal_sovereign_id(tmp_path):
    """The sovereign flag must not let a caller write outside the keys dir."""
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"
    rc = mod.main([
        "--sovereign", "../escape",  # path traversal attempt
        "--charter", str(charter),
        "--keys-dir", str(keys_dir),
        "--receipt", str(receipt),
    ])
    assert rc == 2  # CLI returns 2 on argument-validation failure
    assert not receipt.exists(), "receipt must not be written for invalid id"


# ---------------------------------------------------------------------------
# Defensive: missing Charter is reported, not silently treated as empty.
# ---------------------------------------------------------------------------

def test_sign_charter_raises_on_missing_charter(tmp_path):
    mod = _load_sign_charter_module()
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"
    with pytest.raises(FileNotFoundError):
        mod.sign_charter(
            sovereign="atom-mccree",
            charter_path=tmp_path / "nope.md",
            keys_dir=keys_dir,
            receipt_path=receipt,
        )
    assert not receipt.exists()


# ---------------------------------------------------------------------------
# Defensive: a partial keypair (one half missing) is refused rather than
# silently regenerating both halves and overwriting good material.
# ---------------------------------------------------------------------------

def test_partial_keypair_on_disk_is_refused(tmp_path):
    mod = _load_sign_charter_module()
    charter = _stage_test_charter(tmp_path)
    keys_dir = tmp_path / "keys"
    receipt = tmp_path / "signature.json"

    # Plant a lone .pub file so the script detects the half-state.
    keys_dir.mkdir(parents=True)
    (keys_dir / "atom-mccree.pub").write_text(
        "MCowBQYDK2VwAyEA" + "A" * 28 + "\n",  # not real, fine for this test
        encoding="ascii",
    )
    with pytest.raises(FileExistsError):
        mod.sign_charter(
            sovereign="atom-mccree",
            charter_path=charter,
            keys_dir=keys_dir,
            receipt_path=receipt,
        )
    # No receipt should have been written on the half-state failure.
    assert not receipt.exists()
