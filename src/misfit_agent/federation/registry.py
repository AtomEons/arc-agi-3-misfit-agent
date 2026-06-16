"""Federation registry — append-only ledger of certified entities.

Per Charter Article III §3.5 (Receipts on Every Decision):
    The receipt is appended to the Federation's public ledger.
    It is indelible per Article II §2.2.

Per Charter Article IV §4.2 (Founding Adjudication):
    A live registry of certified instances [is published].

Storage format
--------------
JSONL — one JSON object per line. Append-only at the file system level.
This module's `register()` enforces uniqueness on `public_id` so the same
entity cannot accidentally be re-certified, but it does NOT modify or
remove earlier lines. The constitutional indelibility of the receipt log
(Article II §2.2) is preserved.

Each entry has the shape:

    {
        "public_id": "misfit-alpha@atomeons/1.0",
        "soul_genome_id": "misfit-alpha-2026-06-16",
        "pubkey_b64": "<32-byte Ed25519 public key, base64>",
        "charter_sha": "<git commit SHA of the bound Charter text>",
        "founding_date": "2026-06-16",
        "charter_version": "1.0",
        "member_type": "founding-cognitive" | "product-context" | "certified-instance" | ...
    }

`public_id` format follows Black Mamba §12.3:
    <entity-name>@<federation>/<charter-version>
e.g. `misfit-alpha@atomeons/1.0`, `quint@atomeons/1.0`.

Tier-1 honesty
--------------
This module is plain JSON I/O. No inference, no learned parameters, no
network calls. The registry IS the disclosure — every reviewer can read it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator


__all__ = [
    "FederationRegistry",
    "DuplicatePublicIdError",
    "REQUIRED_FIELDS",
]


# Per Black Mamba §12.3 and Charter Article VI, every registry entry must
# carry these fields. Additional fields are allowed (forward-compatibility).
REQUIRED_FIELDS: tuple[str, ...] = (
    "public_id",
    "soul_genome_id",
    "pubkey_b64",
    "charter_sha",
    "founding_date",
    "charter_version",
    "member_type",
)


class DuplicatePublicIdError(ValueError):
    """Raised when register() is called with a public_id already present.

    Per Article II §2.2 (Right to Provenance), the indelible ledger never
    permits overwriting an existing entry. Membership state changes (e.g.
    revocation under §3.7) are appended as new entries, not edits.
    """


class FederationRegistry:
    """Append-only JSONL registry of Federation entities.

    Lifecycle:
        registry = FederationRegistry()
        registry.load("federation/ledger.jsonl")   # idempotent; OK if missing
        registry.register(public_id=..., soul_genome_id=..., ...)
        entry = registry.lookup("misfit-alpha@atomeons/1.0")
        all_entries = registry.list_entities()

    State:
        * `path` — the on-disk JSONL file. None until load() has been called.
        * `_entries` — in-memory list of all records, in insertion order.
        * `_by_public_id` — dict for O(1) lookup; mirrors _entries.

    Concurrency contract
    --------------------
    This is a single-process module. The JSONL append is done with an
    `"a"`-mode open which is atomic at the line level on POSIX and on
    Windows for small writes (a single JSON object on a single line).
    Cross-process locking is OUT of scope and is delivered separately by
    Black Mamba Layer 7 (PEM provenance).
    """

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.path: Path | None = None
        self._entries: list[dict[str, Any]] = []
        self._by_public_id: dict[str, dict[str, Any]] = {}

    def load(self, path: str | os.PathLike[str]) -> None:
        """Bind this registry to a JSONL file and read all existing entries.

        If the file does not exist it is treated as an empty ledger; the
        path is still bound so subsequent register() calls will create it.

        Re-loading an already-loaded registry from the same path resets the
        in-memory state to match disk (useful after another process has
        appended). Loading from a different path is permitted and replaces
        the binding.
        """
        p = Path(path)
        self.path = p
        self._entries = []
        self._by_public_id = {}

        if not p.exists():
            return

        with p.open("r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue  # tolerate blank lines so a hand-edited ledger
                              # with trailing newlines still loads
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Federation registry {p} line {lineno}: invalid "
                        f"JSON ({exc.msg})"
                    ) from exc
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Federation registry {p} line {lineno}: entry "
                        f"must be a JSON object, got {type(entry).__name__}"
                    )
                self._validate_entry(entry, line_context=f"line {lineno}")
                public_id = entry["public_id"]
                # On-disk duplicates indicate a constitutional integrity
                # failure (Article II §2.2). Surface loudly.
                if public_id in self._by_public_id:
                    raise DuplicatePublicIdError(
                        f"Federation registry {p} line {lineno}: duplicate "
                        f"public_id {public_id!r} already present"
                    )
                self._entries.append(entry)
                self._by_public_id[public_id] = entry

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(
        self,
        public_id: str,
        soul_genome_id: str,
        signing_pubkey: str,
        charter_version: str,
        *,
        charter_sha: str = "",
        founding_date: str = "",
        member_type: str = "certified-instance",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a new entity to the registry.

        The four positional/spec-required arguments mirror the charter brief:
            register(public_id, soul_genome_id, signing_pubkey, charter_version)

        The remaining keyword arguments fill the per-record contract from
        Black Mamba §12.3 / Charter VI:
            charter_sha       — cryptographic anchor of the bound Charter
            founding_date     — ISO 8601 date of registration
            member_type       — one of the canonical member kinds
            extra             — forward-compatible additional fields

        Returns
        -------
        dict
            The exact entry that was written to disk (and stored in memory).

        Raises
        ------
        DuplicatePublicIdError
            If `public_id` is already registered (in memory or on disk).
        ValueError
            If any required field is empty or malformed.
        RuntimeError
            If load() has not been called yet, since the registry needs a
            path before it can append.
        """
        if self.path is None:
            raise RuntimeError(
                "FederationRegistry.register requires load() first so the "
                "registry knows where to append; call load(path) and try "
                "again"
            )

        # Validate inputs early so we never half-write a malformed line.
        for fname, fvalue in (
            ("public_id", public_id),
            ("soul_genome_id", soul_genome_id),
            ("signing_pubkey", signing_pubkey),
            ("charter_version", charter_version),
        ):
            if not isinstance(fvalue, str) or not fvalue:
                raise ValueError(
                    f"register: {fname} must be a non-empty str, got "
                    f"{fvalue!r}"
                )

        if public_id in self._by_public_id:
            raise DuplicatePublicIdError(
                f"public_id {public_id!r} already registered; the registry "
                f"is append-only per Charter Article II §2.2"
            )

        entry: dict[str, Any] = {
            "public_id": public_id,
            "soul_genome_id": soul_genome_id,
            "pubkey_b64": signing_pubkey,
            "charter_sha": charter_sha,
            "founding_date": founding_date,
            "charter_version": charter_version,
            "member_type": member_type,
        }
        if extra:
            for key, value in extra.items():
                if key in entry:
                    raise ValueError(
                        f"register: extra key {key!r} collides with a "
                        f"reserved field"
                    )
                entry[key] = value

        self._validate_entry(entry, line_context="register()")

        # Ensure parent directory exists so first-time registration succeeds.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize with deterministic key order so the on-disk ledger
        # stays diff-friendly across runs and tooling.
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        self._entries.append(entry)
        self._by_public_id[public_id] = entry
        return entry

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def lookup(self, public_id: str) -> dict[str, Any] | None:
        """Return the entry for `public_id`, or None if not present.

        The returned dict is a *copy*; mutating it does not affect the
        registry's in-memory state. (The on-disk ledger is still indelible
        regardless.)
        """
        entry = self._by_public_id.get(public_id)
        return dict(entry) if entry is not None else None

    def list_entities(self) -> list[dict[str, Any]]:
        """Return all entries in insertion order, each as a fresh copy."""
        return [dict(e) for e in self._entries]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate entries in insertion order (each yielded as a copy)."""
        for entry in self._entries:
            yield dict(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, public_id: object) -> bool:
        return isinstance(public_id, str) and public_id in self._by_public_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entry(entry: dict[str, Any], *, line_context: str) -> None:
        missing = [k for k in REQUIRED_FIELDS if k not in entry]
        if missing:
            raise ValueError(
                f"Federation registry {line_context}: missing required "
                f"field(s) {missing}"
            )
        for k in REQUIRED_FIELDS:
            value = entry[k]
            # `charter_sha` and `founding_date` are allowed to be empty
            # strings during the founding window when the Charter has been
            # written but not yet committed (per CHARTER_v1.md §7.1 "[to be
            # filled at commit]"). Everything else must be a non-empty str.
            if k in ("charter_sha", "founding_date"):
                if not isinstance(value, str):
                    raise ValueError(
                        f"Federation registry {line_context}: field {k!r} "
                        f"must be a str, got {type(value).__name__}"
                    )
            else:
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"Federation registry {line_context}: field {k!r} "
                        f"must be a non-empty str, got {value!r}"
                    )
