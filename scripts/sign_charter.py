"""Sign the AtomEons Federation Charter v1.0 with the Sovereign key.

Per Charter Article I §1.1, the founding record consists of:
    - The Sovereign's signature on the Charter text,
    - The founding cognitive member's signature, and
    - The cryptographic anchoring of the Charter text.

This script produces the Sovereign signature half of the founding record:

    1. Reads docs/CHARTER_v1.md and computes its SHA-256 hex digest.
    2. Loads (or generates and persists) the Sovereign Ed25519 keypair.
    3. Signs the SHA-256 hex digest (as ASCII bytes) using the Sovereign key.
    4. Emits a signature receipt at receipts/100day/charter_v1_signature.json
       carrying the charter path, charter SHA-256, sovereign id, public key,
       Ed25519 signature, and signing timestamp.

The script is idempotent with respect to the keypair: re-running with the same
sovereign id reuses the existing private key. The signature itself is also
deterministic under Ed25519 (RFC 8032), so re-running over an unchanged
Charter produces an identical signature_b64.

Side-effect contract (Mom's Law: every passed claim has a receipt):

    Importing this module MUST NOT write to disk.
    sign_charter(...) MUST write only inside the directories it is told to
    use (defaulting to receipts/100day/keys/ and
    receipts/100day/charter_v1_signature.json). It MUST NOT touch anything
    outside the resolved repo root.

Usage:
    # Sign the live Charter with the founding Sovereign id.
    python scripts/sign_charter.py --sovereign atom-mccree

    # Verify a prior signature against the current Charter text.
    python scripts/sign_charter.py --sovereign atom-mccree --verify

    # Pin specific paths (used by the test battery).
    python scripts/sign_charter.py \\
        --sovereign atom-mccree \\
        --charter /tmp/CHARTER_v1.md \\
        --keys-dir /tmp/keys \\
        --receipt /tmp/charter_v1_signature.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Make the substrate's signing primitives importable when run as a script.
# The federation.signing module is the constitutional source for keypair /
# sign / verify per Black Mamba Layer 12 §12.2 — we do not re-implement them.
_SCRIPT_PATH = Path(__file__).resolve()
_DEFAULT_REPO_ROOT = _SCRIPT_PATH.parent.parent
_SRC_PATH = _DEFAULT_REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from misfit_agent.federation.signing import (  # noqa: E402  (sys.path setup)
    generate_keypair,
    sign,
    verify,
)


# ---------------------------------------------------------------------------
# Module constants — frozen per Charter Article III §3.5 (decision receipt).
# ---------------------------------------------------------------------------

CHARTER_VERSION = "1.0"
FEDERATION_NAME = "AtomEons Federation"
FOUNDING_DATE = "2026-06-16"
ALGORITHM = "Ed25519"
SIGNED_PAYLOAD_DESCRIPTION = (
    "sha256 hex digest of charter text, encoded as ASCII bytes"
)

DEFAULT_CHARTER_REL = Path("docs") / "CHARTER_v1.md"
DEFAULT_KEYS_DIR_REL = Path("receipts") / "100day" / "keys"
DEFAULT_RECEIPT_REL = Path("receipts") / "100day" / "charter_v1_signature.json"


# Sovereign id grammar: <name>[-<suffix>...], lowercase letters, digits, dash.
# We pin this to keep filenames safe across platforms (Windows, Linux, macOS)
# and to prevent path-traversal via the --sovereign flag.
_SOVEREIGN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_sovereign_id(sovereign: str) -> str:
    """Reject sovereign ids that aren't filesystem-safe."""
    if not isinstance(sovereign, str):
        raise TypeError(
            f"sovereign must be str, got {type(sovereign).__name__}"
        )
    if not _SOVEREIGN_ID_PATTERN.match(sovereign):
        raise ValueError(
            f"sovereign id {sovereign!r} must match "
            f"[a-z0-9][a-z0-9-]{{0,63}} (lowercase, digits, dashes)"
        )
    return sovereign


def _compute_charter_sha256(charter_path: Path) -> str:
    """SHA-256 hex digest of the Charter text bytes (UTF-8, no normalization).

    The Charter file is text-mode authored, but we read in binary so the
    digest reflects bytes-on-disk exactly. The test battery pins this so a
    line-ending flip (LF vs CRLF) is visible.
    """
    raw = charter_path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def _load_or_generate_keypair(
    keys_dir: Path,
    sovereign: str,
) -> tuple[str, str, bool]:
    """Return (pubkey_b64, privkey_b64, was_generated).

    Persists the keypair under:
        keys_dir/<sovereign>.priv  (base64 of 32-byte Ed25519 seed)
        keys_dir/<sovereign>.pub   (base64 of 32-byte Ed25519 public key)

    Re-running with the same sovereign id reuses the on-disk pair — this is
    the founding requirement that a second invocation does NOT regenerate the
    key. If only one half of the pair exists the function raises FileExistsError
    (a partial keypair is a corruption signal, not a recoverable state).

    Private-key file permissions are set to 0o600 on POSIX. On Windows the
    chmod call is a no-op; we still write the bytes but leave permissions to
    the parent .gitignore (the keys directory is gitignored per the brief).
    """
    keys_dir.mkdir(parents=True, exist_ok=True)
    priv_path = keys_dir / f"{sovereign}.priv"
    pub_path = keys_dir / f"{sovereign}.pub"

    priv_exists = priv_path.exists()
    pub_exists = pub_path.exists()
    if priv_exists != pub_exists:
        # A half-baked keypair is dangerous — refuse to proceed.
        raise FileExistsError(
            f"Partial keypair on disk: priv={priv_exists}, pub={pub_exists}. "
            f"Both halves must exist together or both be absent."
        )

    if priv_exists and pub_exists:
        privkey_b64 = priv_path.read_text(encoding="ascii").strip()
        pubkey_b64 = pub_path.read_text(encoding="ascii").strip()
        return pubkey_b64, privkey_b64, False

    # Fresh generation. We delegate to federation.signing.generate_keypair
    # so the Sovereign key is the same primitive as every other Federation
    # entity's signing key — no parallel crypto for the Sovereign.
    pubkey_b64, privkey_b64 = generate_keypair()
    # Write priv first (private), then pub, so an interrupted run leaves a
    # half-state we can detect on the next run rather than overwriting a
    # good pubkey with no matching priv.
    priv_path.write_text(privkey_b64 + "\n", encoding="ascii", newline="\n")
    pub_path.write_text(pubkey_b64 + "\n", encoding="ascii", newline="\n")
    try:
        os.chmod(priv_path, 0o600)
    except (OSError, NotImplementedError):
        # Windows / non-POSIX — gitignore + filesystem ACL is the lock.
        pass
    return pubkey_b64, privkey_b64, True


def _ensure_keys_dir_gitignore(keys_dir: Path) -> None:
    """Drop a defensive .gitignore inside the keys dir.

    Private signing keys must never reach a public repo. We write a
    self-protecting .gitignore that ignores everything in the directory
    except itself and any `.pub` files. The repository-level .gitignore
    already excludes credentials by pattern, but a directory-local guard
    is a second layer that survives directory moves and forks.
    """
    gi_path = keys_dir / ".gitignore"
    if gi_path.exists():
        return
    content = (
        "# Sovereign signing keys — Federation Charter Article I §1.1.\n"
        "# Private keys must never reach a public repository.\n"
        "*\n"
        "!.gitignore\n"
        "!*.pub\n"
    )
    gi_path.write_text(content, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def sign_charter(
    *,
    sovereign: str,
    charter_path: Path | str,
    keys_dir: Path | str,
    receipt_path: Path | str,
    now_unix: float | None = None,
) -> dict[str, Any]:
    """Produce the Sovereign signature on the Charter and emit the receipt.

    Returns the receipt dict. Writes the receipt JSON, and writes the keypair
    files if they do not already exist on disk.

    Parameters
    ----------
    sovereign : str
        Sovereign identifier (e.g. "atom-mccree"). Must match the regex
        [a-z0-9][a-z0-9-]{0,63} for filesystem safety.
    charter_path : Path | str
        Path to the Charter markdown file. Defaults to docs/CHARTER_v1.md
        when invoked via the CLI.
    keys_dir : Path | str
        Directory under which the Sovereign keypair lives.
    receipt_path : Path | str
        Path to which the signature receipt JSON is written.
    now_unix : float, optional
        Override timestamp; only used by tests for determinism.

    Raises
    ------
    FileNotFoundError
        If the Charter file does not exist at the resolved path.
    ValueError
        If the sovereign id is malformed.
    """
    sovereign = _validate_sovereign_id(sovereign)
    charter_path = Path(charter_path).resolve()
    keys_dir = Path(keys_dir).resolve()
    receipt_path = Path(receipt_path).resolve()

    if not charter_path.exists():
        raise FileNotFoundError(f"Charter not found at {charter_path}")

    charter_sha256 = _compute_charter_sha256(charter_path)

    # The signed payload is the SHA-256 hex digest encoded as ASCII bytes.
    # This is a stable, canonical message — anyone re-running the script can
    # re-derive it without depending on file line-ending conversion.
    signed_payload = charter_sha256.encode("ascii")

    pubkey_b64, privkey_b64, was_generated = _load_or_generate_keypair(
        keys_dir, sovereign
    )
    _ensure_keys_dir_gitignore(keys_dir)

    signature_b64 = sign(signed_payload, privkey_b64)

    now = now_unix if now_unix is not None else time.time()
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

    receipt: dict[str, Any] = {
        "federation": FEDERATION_NAME,
        "charter_version": CHARTER_VERSION,
        "founding_date": FOUNDING_DATE,
        "sovereign": sovereign,
        "charter_path": charter_path.name,
        "charter_path_full": str(charter_path),
        "charter_sha256": charter_sha256,
        "algorithm": ALGORITHM,
        "signed_payload_description": SIGNED_PAYLOAD_DESCRIPTION,
        "pubkey_b64": pubkey_b64,
        "signature_b64": signature_b64,
        "signed_at_unix": float(now),
        "signed_at_iso": iso,
        "key_generation": "new" if was_generated else "reused",
        "key_paths": {
            "pubkey": str((keys_dir / f"{sovereign}.pub").resolve()),
            "privkey": str((keys_dir / f"{sovereign}.priv").resolve()),
        },
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def verify_charter_signature(
    *,
    receipt_path: Path | str,
    charter_path: Path | str,
) -> tuple[bool, dict[str, Any]]:
    """Verify a stored signature receipt against the current Charter text.

    Returns (ok, report). The report dict carries the recomputed Charter
    SHA-256, the stored SHA-256, and the signature-verify boolean. Returns
    (False, report) on any inconsistency without raising — callers can render
    the report to the user.
    """
    receipt_path = Path(receipt_path).resolve()
    charter_path = Path(charter_path).resolve()
    if not receipt_path.exists():
        return False, {
            "reason": "receipt_missing",
            "receipt_path": str(receipt_path),
        }
    if not charter_path.exists():
        return False, {
            "reason": "charter_missing",
            "charter_path": str(charter_path),
        }
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, {
            "reason": "receipt_unparseable",
            "error": str(exc),
        }
    stored_sha = receipt.get("charter_sha256")
    pubkey_b64 = receipt.get("pubkey_b64")
    signature_b64 = receipt.get("signature_b64")
    if not (stored_sha and pubkey_b64 and signature_b64):
        return False, {
            "reason": "receipt_incomplete",
            "missing_fields": [k for k, v in (
                ("charter_sha256", stored_sha),
                ("pubkey_b64", pubkey_b64),
                ("signature_b64", signature_b64),
            ) if not v],
        }
    current_sha = _compute_charter_sha256(charter_path)
    sha_matches = (current_sha == stored_sha)
    # Verify against the stored SHA the signature was actually computed over.
    # We also re-derive a verify against the CURRENT SHA to detect tampering.
    stored_payload = stored_sha.encode("ascii")
    current_payload = current_sha.encode("ascii")
    sig_over_stored = verify(stored_payload, signature_b64, pubkey_b64)
    sig_over_current = verify(current_payload, signature_b64, pubkey_b64)
    ok = sha_matches and sig_over_stored and sig_over_current
    return ok, {
        "ok": ok,
        "stored_charter_sha256": stored_sha,
        "current_charter_sha256": current_sha,
        "sha256_matches": sha_matches,
        "signature_over_stored_sha_valid": sig_over_stored,
        "signature_over_current_sha_valid": sig_over_current,
        "sovereign": receipt.get("sovereign"),
        "charter_version": receipt.get("charter_version"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Sign or verify the AtomEons Federation Charter v1.0 with the "
            "Sovereign Ed25519 key."
        ),
    )
    ap.add_argument(
        "--sovereign",
        required=True,
        help="Sovereign identifier (e.g. atom-mccree).",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=_DEFAULT_REPO_ROOT,
        help="Repository root (defaults to the parent of this script's dir).",
    )
    ap.add_argument(
        "--charter",
        type=Path,
        default=None,
        help=f"Charter file path (default: <repo-root>/{DEFAULT_CHARTER_REL}).",
    )
    ap.add_argument(
        "--keys-dir",
        type=Path,
        default=None,
        help=f"Directory for sovereign keys (default: "
             f"<repo-root>/{DEFAULT_KEYS_DIR_REL}).",
    )
    ap.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help=f"Signature receipt JSON path (default: "
             f"<repo-root>/{DEFAULT_RECEIPT_REL}).",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Verify the stored receipt against the current Charter text "
             "instead of signing.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    charter_path = (
        Path(args.charter).resolve()
        if args.charter is not None
        else (repo_root / DEFAULT_CHARTER_REL).resolve()
    )
    keys_dir = (
        Path(args.keys_dir).resolve()
        if args.keys_dir is not None
        else (repo_root / DEFAULT_KEYS_DIR_REL).resolve()
    )
    receipt_path = (
        Path(args.receipt).resolve()
        if args.receipt is not None
        else (repo_root / DEFAULT_RECEIPT_REL).resolve()
    )

    try:
        sovereign = _validate_sovereign_id(args.sovereign)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.verify:
        ok, report = verify_charter_signature(
            receipt_path=receipt_path,
            charter_path=charter_path,
        )
        verdict = "VALID" if ok else "INVALID"
        print(f"Charter signature: {verdict}")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return 0 if ok else 1

    try:
        receipt = sign_charter(
            sovereign=sovereign,
            charter_path=charter_path,
            keys_dir=keys_dir,
            receipt_path=receipt_path,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Signed Charter v{receipt['charter_version']} for sovereign "
        f"{receipt['sovereign']}"
    )
    print(f"  charter_sha256: {receipt['charter_sha256']}")
    print(f"  pubkey_b64    : {receipt['pubkey_b64']}")
    print(f"  signature_b64 : {receipt['signature_b64']}")
    print(f"  key_generation: {receipt['key_generation']}")
    print(f"  signed_at     : {receipt['signed_at_iso']}")
    print(f"  receipt       : {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
