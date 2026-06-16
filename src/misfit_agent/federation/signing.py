"""Ed25519 signing primitives for the AtomEons Federation.

Per Black Mamba Layer 12 (Identity) §12.2:
    Per-instance Ed25519 keypair generated at first boot.
    Public key registered with the Federation receipt log.
    Used for signing receipt log entries, CHSG votes, and authenticating
    to other Federation members for cross-examination.

Per Charter Article III §3.5 the receipt log is constitutionally indelible,
which means receipt entries MUST be signed with a verifiable key. This module
provides the three primitives every Federation entity needs:

    generate_keypair() -> (pubkey_b64, privkey_b64)
    sign(message_bytes, privkey_b64) -> signature_b64
    verify(message_bytes, signature_b64, pubkey_b64) -> bool

Encoding contract
-----------------
* Algorithm: Ed25519 (RFC 8032) via cryptography.hazmat.primitives.asymmetric.
* Wire format for keys and signatures: standard base64 (RFC 4648 §4) with
  padding. We do NOT use urlsafe variant — the receipt log is plain text and
  '+' / '/' are fine in JSONL / YAML quoted strings.
* Raw byte sizes: pubkey = 32, privkey = 32 (seed form), signature = 64.

Tier-1 honesty
--------------
The `cryptography` package is a CPython binding over OpenSSL / libsodium-style
primitives. It contains zero LLM weights, zero learned parameters, and zero
network calls. It is a pure cryptography primitive and is therefore Tier-1
clean. See `tests/test_tier1_attestation.py` for the mechanical guard.
"""

from __future__ import annotations

import base64
from typing import Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


__all__ = ["generate_keypair", "sign", "verify"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _b64encode(raw: bytes) -> str:
    """Standard base64 with padding -> ASCII str."""
    return base64.b64encode(raw).decode("ascii")


def _b64decode(text: str) -> bytes:
    """Inverse of _b64encode. Raises ValueError on malformed input."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise ValueError(f"invalid base64 input: {exc}") from exc


def _privkey_from_b64(privkey_b64: str) -> Ed25519PrivateKey:
    raw = _b64decode(privkey_b64)
    if len(raw) != 32:
        raise ValueError(
            f"Ed25519 private key seed must be 32 bytes, got {len(raw)}"
        )
    return Ed25519PrivateKey.from_private_bytes(raw)


def _pubkey_from_b64(pubkey_b64: str) -> Ed25519PublicKey:
    raw = _b64decode(pubkey_b64)
    if len(raw) != 32:
        raise ValueError(
            f"Ed25519 public key must be 32 bytes, got {len(raw)}"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_keypair() -> Tuple[str, str]:
    """Generate a fresh Ed25519 keypair.

    Returns
    -------
    (pubkey_b64, privkey_b64)
        Both are standard base64 strings. The private key is the raw 32-byte
        seed (NOT a PEM-wrapped PKCS#8 blob). Callers must protect the
        private key as a secret — anyone holding it can sign for this entity.

    Examples
    --------
    >>> pub, priv = generate_keypair()
    >>> len(base64.b64decode(pub))
    32
    >>> len(base64.b64decode(priv))
    32
    """
    sk = Ed25519PrivateKey.generate()
    priv_raw = sk.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    pub_raw = sk.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return _b64encode(pub_raw), _b64encode(priv_raw)


def sign(message_bytes: bytes, privkey_b64: str) -> str:
    """Sign `message_bytes` with the Ed25519 private key in `privkey_b64`.

    Parameters
    ----------
    message_bytes : bytes
        The exact bytes to sign. Callers must canonicalize JSON or YAML
        BEFORE calling sign() if they want signatures to round-trip across
        serializers.
    privkey_b64 : str
        Standard-base64 32-byte private key seed (as produced by
        generate_keypair()).

    Returns
    -------
    str
        Standard-base64 encoded 64-byte Ed25519 signature.

    Raises
    ------
    TypeError
        If `message_bytes` is not bytes.
    ValueError
        If `privkey_b64` is not a valid base64 32-byte seed.
    """
    if not isinstance(message_bytes, (bytes, bytearray)):
        raise TypeError(
            f"message_bytes must be bytes, got {type(message_bytes).__name__}"
        )
    sk = _privkey_from_b64(privkey_b64)
    sig = sk.sign(bytes(message_bytes))
    return _b64encode(sig)


def verify(
    message_bytes: bytes,
    signature_b64: str,
    pubkey_b64: str,
) -> bool:
    """Verify `signature_b64` over `message_bytes` under `pubkey_b64`.

    Returns True iff the signature is a valid Ed25519 signature of
    `message_bytes` under the given public key. Returns False on any
    cryptographic failure (invalid signature, wrong key, tampered message).

    Returns False (does NOT raise) on malformed-input failure modes
    (non-base64 strings, wrong byte lengths, non-bytes message). This
    matches the spec's contract — `verify` returns bool, never throws —
    so callers can use it directly in boolean expressions on untrusted
    input from the public ledger.

    Parameters
    ----------
    message_bytes : bytes
    signature_b64 : str
    pubkey_b64 : str

    Returns
    -------
    bool
        True if and only if the signature verifies.
    """
    if not isinstance(message_bytes, (bytes, bytearray)):
        return False
    try:
        pk = _pubkey_from_b64(pubkey_b64)
        sig_raw = _b64decode(signature_b64)
    except (ValueError, TypeError):
        return False
    if len(sig_raw) != 64:
        return False
    try:
        pk.verify(sig_raw, bytes(message_bytes))
    except InvalidSignature:
        return False
    except Exception:
        # Defensive: any other cryptographic-library exception is treated as
        # verification failure. We never propagate exceptions out of verify().
        return False
    return True
