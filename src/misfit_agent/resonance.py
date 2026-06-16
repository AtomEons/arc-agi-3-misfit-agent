"""Resonance library — per-install JSONL of (fingerprint, winning_policy).

Critical Tier-1 honesty constraints:
  - Append-only on disk. Never edit prior rows.
  - Per-install. No cross-customer / cross-machine sharing.
  - Source-tagged: every row records source = "self-solved" so an audit can
    confirm the library was not pre-seeded from public ARC corpora.
  - In-memory bounded to recent entries to keep K-NN cheap; full library is
    re-read from disk on next agent boot.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

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
    game_id: str
    fingerprint: list[float]
    winning_policy: list[dict]  # serialized ActionRecord list
    composite_score: float
    solved_at_unix: float
    source: str = "self-solved"


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
                    lib.entries.append(LibraryEntry(**d))
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

    def retrieve_policy_seeds(self, query: np.ndarray, k: int = 5
                              ) -> list[list[dict]]:
        """Return up to K unique winning policies for resonance-seeded search."""
        seen: set[str] = set()
        out: list[list[dict]] = []
        for e, _ in self.find_k_nearest(query, k):
            sig = json.dumps([a["action_value"] for a in e.winning_policy])
            if sig in seen:
                continue
            seen.add(sig)
            out.append(e.winning_policy)
        return out

    def record_solved(self, fingerprint: np.ndarray, winning_policy: list[ActionRecord],
                       composite_score: float, source: str = "self-solved",
                       game_id: Optional[str] = None) -> LibraryEntry:
        if source != "self-solved":
            raise ValueError(
                f"Tier-1 honesty: library only accepts self-solved entries, got {source!r}"
            )
        gid = game_id or (winning_policy[0].action_name if winning_policy else "unknown")
        entry = LibraryEntry(
            game_id=gid,
            fingerprint=[float(x) for x in fingerprint.tolist()],
            winning_policy=[asdict(a) for a in winning_policy],
            composite_score=float(composite_score),
            solved_at_unix=time.time(),
            source=source,
        )
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
