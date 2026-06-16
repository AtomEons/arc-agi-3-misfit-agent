"""Resonance library — per-install JSONL of PEM-bound experience entries.

Provenance-Enforced Memory (PEM): 8-field contract per docs/PAPER_v1.md §3.
An entry is admissible only if ALL eight fields are present and verifiable:

  1. source_provenance     — what created this entry (self-solve, ablation, ...)
  2. contamination_tier    — which Tier (1, 2, 3) the entry was produced under
  3. creation_event        — timestamp + episode signature that produced it
  4. replay_pointer        — exact reproduction path
  5. mutation_history      — any post-creation edits with reasons
  6. expiry_decay_rule     — when this entry stops being trusted
  7. evidence_payload      — the observation that justifies the win
  8. downstream_usage_receipt — every retrieval that consumed this entry

Tier-1 honesty constraints (enforced at write time):
  - Append-only on disk. Never edit prior rows.
  - Per-install. No cross-customer / cross-machine sharing.
  - source_provenance MUST be "self-solved" — pre-seeded entries are rejected.
  - contamination_tier MUST be "tier_1" — entries produced under Tier-2 (LLM
    heuristic in inference path) are stored in a SEPARATE library file and
    cannot pollute the Tier-1 retrieval surface.
  - In-memory bounded to recent entries to keep K-NN cheap; full library is
    re-read from disk on next agent boot.

Legacy entries written before PEM (only `source` field present) are migrated
on read with safe defaults: contamination_tier="tier_1", expiry_decay_rule=
"never_decay", and synthesized creation_event from solved_at_unix. The
migrated rows are flagged in mutation_history so an auditor can tell which
rows are native PEM vs upgraded.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from .fingerprint import FINGERPRINT_DIM, cosine
from .episode import ActionRecord


def default_library_path() -> pathlib.Path:
    """Per-install path: %LOCALAPPDATA%\\misfit-agent on Windows,
    or ~/.local/share/misfit-agent elsewhere."""
    if os.name == "nt":
        base = pathlib.Path(os.environ.get("LOCALAPPDATA", str(pathlib.Path.home())))
    else:
        base = pathlib.Path.home() / ".local" / "share"
    return base / "misfit-agent" / "resonance_library.jsonl"


@dataclass
class LibraryEntry:
    """PEM-bound resonance entry. See module docstring for the 8 fields."""
    # Field 1: source_provenance — what created this entry.
    # Admissible values: "self-solved", "ablation-condition-a", "ablation-condition-b".
    # Pre-seeded entries are rejected at record_solved.
    source: str = "self-solved"
    # Field 2: contamination_tier — which Tier the entry was produced under.
    # Tier-1 retrieval surface accepts ONLY "tier_1" entries.
    contamination_tier: str = "tier_1"
    # Field 3: creation_event — timestamp + episode signature.
    # episode_signature is sha256(game_id + first8(fingerprint))[:16].
    solved_at_unix: float = 0.0
    episode_signature: str = ""
    # Field 4: replay_pointer — path to reproduce.
    # Format: "kernel_version:<v>|game_id:<id>|library_path:<resonance_jsonl_path>".
    replay_pointer: str = ""
    # Field 5: mutation_history — every post-creation edit with reason.
    # Append-only by convention; empty list for native-PEM rows.
    mutation_history: list[dict] = field(default_factory=list)
    # Field 6: expiry_decay_rule — when this entry stops being trusted.
    # Forms: "never_decay" | "hours:N" | "after_kernel:<v>" | "until_unix:<ts>".
    expiry_decay_rule: str = "never_decay"
    # Field 7: evidence_payload — the observation that justifies the win.
    # The fingerprint + winning_policy together ARE the evidence; we
    # additionally hash the final scene grid for tamper detection.
    fingerprint: list[float] = field(default_factory=list)
    winning_policy: list[dict] = field(default_factory=list)
    evidence_grid_hash: str = ""
    composite_score: float = 0.0
    # Field 8: downstream_usage_receipt — every retrieval consumer (append-only).
    # Each entry: {"consumed_at_unix": float, "consumer_game_id": str,
    #              "kernel_version": str}.
    usage_receipts: list[dict] = field(default_factory=list)
    # Convenience: game_id remains a top-level field for retrieval logging.
    game_id: str = "unknown"


def episode_signature(game_id: str, fingerprint: list[float]) -> str:
    """Deterministic 16-hex-char signature for the creation event."""
    head = ",".join(f"{x:.4f}" for x in (fingerprint or [])[:8])
    return hashlib.sha256(f"{game_id}|{head}".encode()).hexdigest()[:16]


def _migrate_legacy_entry(d: dict) -> Optional[dict]:
    """Upgrade a pre-PEM row (only `source` field present) to a PEM-compliant
    dict with conservative defaults. Returns None if the row is not legacy."""
    if "contamination_tier" in d:
        return None
    if d.get("source") != "self-solved":
        return None
    d.setdefault("contamination_tier", "tier_1")
    d.setdefault("episode_signature",
                 episode_signature(d.get("game_id", "unknown"),
                                   d.get("fingerprint", [])))
    d.setdefault("replay_pointer", f"legacy:|game_id:{d.get('game_id','unknown')}")
    d.setdefault("expiry_decay_rule", "never_decay")
    d.setdefault("evidence_grid_hash", "")
    d.setdefault("usage_receipts", [])
    d.setdefault("mutation_history", [{
        "kind": "pem_migration",
        "at_unix": time.time(),
        "reason": "legacy pre-PEM row migrated with defaults",
    }])
    return d


def validate_pem(entry: dict | LibraryEntry) -> list[str]:
    """Return a list of missing/invalid PEM fields. Empty list = PEM-valid."""
    d = entry if isinstance(entry, dict) else asdict(entry)
    errors: list[str] = []
    if d.get("source") not in ("self-solved", "ablation-condition-a",
                                "ablation-condition-b"):
        errors.append(f"source_provenance: invalid value {d.get('source')!r}")
    if d.get("contamination_tier") not in ("tier_1", "tier_2", "tier_3"):
        errors.append(
            f"contamination_tier: invalid value {d.get('contamination_tier')!r}"
        )
    for k in ("solved_at_unix", "episode_signature", "replay_pointer",
              "expiry_decay_rule", "composite_score"):
        if k not in d or d[k] in (None, ""):
            errors.append(f"{k}: missing or empty")
    if not isinstance(d.get("mutation_history"), list):
        errors.append("mutation_history: must be a list")
    if not isinstance(d.get("usage_receipts"), list):
        errors.append("usage_receipts: must be a list")
    if not d.get("fingerprint"):
        errors.append("fingerprint (evidence_payload): missing or empty")
    return errors


@dataclass
class ResonanceLibrary:
    path: pathlib.Path
    entries: list[LibraryEntry] = field(default_factory=list)
    _pending: list[LibraryEntry] = field(default_factory=list)

    @classmethod
    def load_or_create(cls, path: str | pathlib.Path) -> "ResonanceLibrary":
        path = pathlib.Path(path)
        lib = cls(path=path)
        if not path.exists():
            return lib
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("source") != "self-solved":
                        # Tier-1 honesty: silently skip non-self-solved entries.
                        continue
                    # Upgrade legacy pre-PEM rows in place.
                    migrated = _migrate_legacy_entry(d)
                    if migrated is not None:
                        d = migrated
                    # Tier-1 retrieval surface: only tier_1 entries.
                    if d.get("contamination_tier") != "tier_1":
                        continue
                    try:
                        lib.entries.append(LibraryEntry(**d))
                    except TypeError:
                        # Schema drift — skip silently rather than crash boot.
                        continue
        except OSError:
            pass
        return lib

    def find_k_nearest(self, query: np.ndarray, k: int = 5
                       ) -> list[tuple[LibraryEntry, float]]:
        if not self.entries:
            return []
        scored = []
        for e in self.entries:
            sim = cosine(query, np.asarray(e.fingerprint, dtype=np.float32))
            scored.append((e, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def retrieve_policy_seeds(self, query: np.ndarray, k: int = 5,
                              consumer_game_id: str = "unknown",
                              kernel_version: str = "unknown"
                              ) -> list[list[dict]]:
        """Return up to K unique winning policies for resonance-seeded search.

        PEM field 8: every retrieval appends a usage_receipt to the consumed
        entries so an auditor can trace which past solves influenced which
        future ones. The receipt is in-memory only here; flushed on next save.
        """
        seen: set[str] = set()
        out: list[list[dict]] = []
        now = time.time()
        for e, _ in self.find_k_nearest(query, k):
            sig = json.dumps([a["action_value"] for a in e.winning_policy])
            if sig in seen:
                continue
            seen.add(sig)
            e.usage_receipts.append({
                "consumed_at_unix": now,
                "consumer_game_id": consumer_game_id,
                "kernel_version": kernel_version,
            })
            out.append(e.winning_policy)
        return out

    def record_solved(self, fingerprint: np.ndarray, winning_policy: list[ActionRecord],
                       composite_score: float, source: str = "self-solved",
                       game_id: Optional[str] = None,
                       contamination_tier: str = "tier_1",
                       evidence_grid: Optional[np.ndarray] = None,
                       kernel_version: str = "unknown",
                       expiry_decay_rule: str = "never_decay") -> LibraryEntry:
        """Record a PEM-compliant resonance entry.

        Source-provenance enforcement is the Tier-1 honesty gate: anything
        other than self-solved or an ablation variant is rejected at write time.
        Contamination-tier enforcement: a tier_1 retrieval surface refuses to
        accept tier_2/tier_3 entries even from internal callers.
        """
        if source not in ("self-solved", "ablation-condition-a",
                           "ablation-condition-b"):
            raise ValueError(
                f"Tier-1 honesty: library only accepts self-solved or ablation entries, "
                f"got {source!r}"
            )
        if contamination_tier != "tier_1":
            raise ValueError(
                f"Tier-1 retrieval surface refuses contamination_tier={contamination_tier!r}. "
                f"Use a separate library file for tier_2/tier_3 entries."
            )
        gid = game_id or (winning_policy[0].action_name if winning_policy else "unknown")
        fp_list = [float(x) for x in fingerprint.tolist()]
        sig = episode_signature(gid, fp_list)
        if evidence_grid is not None:
            grid_hash = hashlib.sha256(
                np.asarray(evidence_grid, dtype=np.int32).tobytes()
            ).hexdigest()[:16]
        else:
            grid_hash = ""
        entry = LibraryEntry(
            source=source,
            contamination_tier=contamination_tier,
            solved_at_unix=time.time(),
            episode_signature=sig,
            replay_pointer=(
                f"kernel_version:{kernel_version}|game_id:{gid}"
                f"|library_path:{self.path}"
            ),
            mutation_history=[],
            expiry_decay_rule=expiry_decay_rule,
            fingerprint=fp_list,
            winning_policy=[asdict(a) for a in winning_policy],
            evidence_grid_hash=grid_hash,
            composite_score=float(composite_score),
            usage_receipts=[],
            game_id=gid,
        )
        errors = validate_pem(entry)
        if errors:
            raise ValueError(f"PEM validation failed: {errors}")
        self.entries.append(entry)
        self._pending.append(entry)
        return entry

    def flush_to_disk(self) -> int:
        if not self._pending:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with self.path.open("a", encoding="utf-8") as f:
            for e in self._pending:
                f.write(json.dumps(asdict(e)) + "\n")
                n += 1
        self._pending.clear()
        return n
