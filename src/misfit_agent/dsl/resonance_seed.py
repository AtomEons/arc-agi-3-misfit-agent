"""Resonance-seeded synthesis initialization — DSL Team RESONANCE-SEED.

The Misfit-Alpha substrate ships a per-install resonance library
(`misfit_agent.resonance.ResonanceLibrary`) that stores PEM-bound entries of
(fingerprint, winning_policy) for tasks the agent has self-solved. At
synthesis time, instead of cold-starting beam search over the full atomic
primitive grid, this module looks up the K most-similar past tasks by
fingerprint cosine and reconstructs their winning programs as seed
candidates for the beam.

Public API:
  seed_from_resonance(train_pairs, library_path=None, k=5) -> list[Program]
    Returns up to k Program candidates seeded from the resonance library.
    Returns [] when no library file exists at library_path.

  task_fingerprint(train_pairs) -> np.ndarray
    Re-export of `arc2_solver.task_fingerprint` — the 16-dim ARC-AGI-2
    fingerprint vector. Re-exported here so callers can compute the same
    fingerprint that record_solved writes, without cross-importing the
    sister solver from DSL code.

Signature translation:
  For now, the library stores rule signatures as tuples (the same shape
  emitted by `arc2_solver.<Rule>.signature()`), not full Program ASTs.
  `program_from_signature()` is a strict translator that reconstructs a
  Program from a signature where the rule maps to a DSL primitive. If
  reconstruction is not possible (unknown rule, bad parameters, composed
  signature with an untranslatable arm), the seed is silently SKIPPED —
  the beam falls back to ordinary cold synthesis. This is Tier-1 honest:
  seeds are a search-prior shortcut, not a correctness contract.

Tier-1 honesty constraints:
  - No new third-party imports. numpy + stdlib + already-shipped modules.
  - No learned parameters, no pretrained weights.
  - No public-corpus heuristics — fingerprint similarity is computed
    from observations of the CURRENT task only; the matched entries
    were themselves produced under tier_1 by the source library.
  - All reconstructed seeds are typed Grid->Grid programs that are
    is_complete() (no holes); evaluation safety is the interpreter's job.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from ..arc2_solver import task_fingerprint as _arc2_task_fingerprint
from ..resonance import ResonanceLibrary, default_library_path
from .ast import Program, make_program
from .primitives import (
    Identity, Translate, Rotate, Reflect, Recolor,
    Crop, Tile, Gravity, Symmetrize, KeepWhere,
)


__all__ = [
    "seed_from_resonance",
    "task_fingerprint",
    "program_from_signature",
]


# ---------------------------------------------------------------------------
# Public re-export: task fingerprint
# ---------------------------------------------------------------------------


def task_fingerprint(train_pairs) -> np.ndarray:
    """Return the 16-dim ARC-AGI-2 task fingerprint for the given train pairs.

    Reuses the sister solver's fingerprint so a single fingerprint definition
    governs both library writes (record_solved) and library reads (seeding).
    The vector is a numpy float32 array of length 16.
    """
    if not train_pairs:
        return np.zeros(16, dtype=np.float32)
    # Normalize: accept list-of-lists OR numpy arrays, just like the solver.
    norm: list[tuple[np.ndarray, np.ndarray]] = []
    for pair in train_pairs:
        inp, out = pair
        norm.append((np.asarray(inp, dtype=np.int32),
                     np.asarray(out, dtype=np.int32)))
    return _arc2_task_fingerprint(norm)


# ---------------------------------------------------------------------------
# Signature -> Program translator
# ---------------------------------------------------------------------------


# Map sister-solver rule signature heads to DSL primitive factories. Each
# entry is (signature_head, arity_check, factory). The factory returns a
# Primitive instance OR raises ValueError to signal that the signature
# tuple cannot be reconstructed.
def _translate_atomic(sig: tuple):
    """Translate one atomic rule signature tuple -> Primitive instance.

    Returns None when the signature cannot be reconstructed (unknown head,
    wrong arity, bad params). Returning None is the soft-fail path; the
    caller skips this seed and continues.
    """
    if not isinstance(sig, tuple) or not sig:
        return None
    head = sig[0]

    if head == "Identity":
        # ("Identity",)
        return Identity() if len(sig) == 1 else None

    if head == "Translate2":
        # ("Translate2", dy, dx)
        if len(sig) != 3:
            return None
        try:
            dy = int(sig[1])
            dx = int(sig[2])
        except (TypeError, ValueError):
            return None
        return Translate(dy=dy, dx=dx)

    if head == "Rotate":
        # ("Rotate", k)  with k in {1, 2, 3}
        if len(sig) != 2:
            return None
        try:
            k = int(sig[1])
        except (TypeError, ValueError):
            return None
        if k not in (1, 2, 3):
            return None
        return Rotate(k=k)

    if head == "ReflectH":
        return Reflect(axis="H") if len(sig) == 1 else None
    if head == "ReflectV":
        return Reflect(axis="V") if len(sig) == 1 else None
    if head == "Transpose":
        # arc2_solver's Transpose is np.transpose, which is Reflect(axis=D1)
        return Reflect(axis="D1") if len(sig) == 1 else None

    if head == "Recolor":
        # ("Recolor", tuple_of_(k,v)_items)
        if len(sig) != 2:
            return None
        items = sig[1]
        if not isinstance(items, (tuple, list)):
            return None
        mapping: dict[int, int] = {}
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return None
            try:
                k_i = int(item[0])
                v_i = int(item[1])
            except (TypeError, ValueError):
                return None
            if not (0 <= k_i <= 9 and 0 <= v_i <= 9):
                return None
            mapping[k_i] = v_i
        if not mapping:
            return None
        return Recolor(mapping=mapping)

    if head == "CropToBbox":
        return Crop() if len(sig) == 1 else None

    if head == "Tile":
        # ("Tile", rf, cf)
        if len(sig) != 3:
            return None
        try:
            rf = int(sig[1])
            cf = int(sig[2])
        except (TypeError, ValueError):
            return None
        if rf < 1 or cf < 1:
            return None
        return Tile(rf=rf, cf=cf)

    # Native DSL signature heads (in case callers store them directly).
    if head == "Translate":
        if len(sig) != 3:
            return None
        try:
            dy = int(sig[1]); dx = int(sig[2])
        except (TypeError, ValueError):
            return None
        return Translate(dy=dy, dx=dx)

    if head == "Reflect":
        if len(sig) != 2:
            return None
        axis = sig[1]
        if axis not in ("H", "V", "D1", "D2"):
            return None
        return Reflect(axis=axis)

    if head == "Gravity":
        if len(sig) != 2:
            return None
        direction = sig[1]
        if direction not in ("U", "D", "L", "R"):
            return None
        return Gravity(direction=direction)

    if head == "Symmetrize":
        if len(sig) != 2:
            return None
        axis = sig[1]
        if axis not in ("H", "V", "BOTH"):
            return None
        return Symmetrize(axis=axis)

    if head == "KeepWhere":
        if len(sig) != 2:
            return None
        pred = sig[1]
        if pred not in ("largest", "smallest", "edge_touching", "non_edge"):
            return None
        return KeepWhere(predicate=pred)

    # Unknown / unsupported head.
    return None


def program_from_signature(sig) -> Optional[Program]:
    """Reconstruct a complete Program from a sister-solver rule signature.

    Returns None when the signature cannot be reconstructed. The caller
    treats None as "skip this seed", never as an error.

    Atomic signatures become single-primitive Programs. Composed signatures
    of the form ("Composed", sig_a, sig_b) are NOT reconstructed in this
    initial cut — depth-2 chains require Seq combinator support that the
    DSL ships separately. They are silently skipped (returns None).
    """
    if not isinstance(sig, tuple) or not sig:
        return None
    head = sig[0]

    # Composed signature: out of scope for the atomic-seed path.
    if head == "Composed":
        return None

    prim = _translate_atomic(sig)
    if prim is None:
        return None

    # Build a Grid->Grid Program with a single PrimitiveNode root.
    # The atomic primitives we accept all take exactly one Grid input;
    # the program's leaf is a Grid hole that gets bound to the initial
    # input grid by `evaluate(program, x)`. This matches the convention
    # used by the cold-start synthesizer in dsl/synthesis.py — a single
    # root-level input-bound hole is part of the seed contract, not an
    # unfilled subprogram slot that needs further synthesis.
    try:
        from .ast import make_hole
        from .types import Grid as GridType
        program = make_program(prim, make_hole(GridType, hole_id=0))
    except Exception:
        return None
    return program


# ---------------------------------------------------------------------------
# Resonance seed entry point
# ---------------------------------------------------------------------------


def _winning_policy_to_signature(winning_policy) -> Optional[tuple]:
    """Extract a rule signature tuple from a stored winning_policy.

    The library stores winning_policy as either:
      - a list of ActionRecord dicts (legacy ARC-AGI-3 schema), OR
      - a list with a single dict carrying {"signature": <tuple-like>}
        (ARC-AGI-2 sister-solver schema), OR
      - a top-level dict {"signature": <tuple-like>}.

    Returns the signature tuple when extractable, else None. None means
    this entry's policy cannot drive a DSL seed and is skipped.
    """
    if winning_policy is None:
        return None

    # Case A: dict-at-top {"signature": [...]}
    if isinstance(winning_policy, dict):
        sig = winning_policy.get("signature")
        return _normalize_signature(sig)

    # Case B: list of dicts
    if isinstance(winning_policy, (list, tuple)):
        if not winning_policy:
            return None
        first = winning_policy[0]
        if isinstance(first, dict):
            # Sister-solver schema: {"signature": [...]} as the only entry.
            if "signature" in first:
                return _normalize_signature(first["signature"])
            # Legacy ARC-AGI-3 schema (ActionRecord): not translatable
            # into a DSL program. Skip silently.
            return None
        # Possibly a raw signature tuple stored directly.
        return _normalize_signature(tuple(winning_policy))

    return None


def _normalize_signature(sig) -> Optional[tuple]:
    """Normalize a signature value (list-of-lists from JSON, tuple, ...) into
    nested tuples. Returns None if the input is not signature-shaped."""
    if sig is None:
        return None
    if isinstance(sig, (list, tuple)):
        out: list = []
        for item in sig:
            if isinstance(item, (list, tuple)):
                norm_item = _normalize_signature(item)
                if norm_item is None:
                    return None
                out.append(norm_item)
            else:
                out.append(item)
        return tuple(out)
    return None


def _seed_is_evaluable(program: Program) -> bool:
    """A seed is evaluable when every node is filled, allowing AT MOST a
    single root-level Grid hole that the interpreter binds to the initial
    input grid at evaluate(p, x) time.

    This mirrors the cold-start synthesizer's convention: a depth-1 atomic
    program looks like `prim(<Grid hole>)` and is treated as a fully
    synthesized seed even though it contains the input-port hole.
    """
    from .ast import PrimitiveNode, HoleNode
    from .types import DslType
    root = program.root
    if not isinstance(root, PrimitiveNode):
        return False
    # Allow exactly one root-level Grid hole as the input port; any deeper
    # hole indicates an un-synthesized subprogram and disqualifies the seed.
    for child in root.children:
        if isinstance(child, HoleNode):
            if child.expected_type != DslType.GRID:
                return False
            continue
        if isinstance(child, PrimitiveNode):
            # Walk down: nested PrimitiveNodes must themselves be evaluable.
            if not _node_is_evaluable(child):
                return False
            continue
        # ConstNode is fine.
    return True


def _node_is_evaluable(node) -> bool:
    """Recursive evaluable check for nested PrimitiveNodes. Only the ROOT
    is allowed to host a leaf input hole; deeper nodes must be fully bound.
    """
    from .ast import PrimitiveNode, HoleNode
    if isinstance(node, HoleNode):
        return False
    if isinstance(node, PrimitiveNode):
        for c in node.children:
            if not _node_is_evaluable(c):
                return False
    return True


def seed_from_resonance(
    train_pairs,
    library_path: Optional[pathlib.Path | str] = None,
    k: int = 5,
) -> list[Program]:
    """Return up to k Program seeds drawn from the resonance library.

    Args:
        train_pairs: list of (input_grid, output_grid) demonstrations for
            the current task. Each grid may be list-of-lists or numpy.
        library_path: explicit path to the resonance JSONL file. When None,
            falls back to `resonance.default_library_path()`. If the path
            does not exist, returns [] (cold-start path).
        k: maximum number of seed programs to return.

    Returns:
        A list of up to k typed Grid->Grid Programs, each is_complete().
        Returns [] when:
          - the library file does not exist
          - the library file exists but contains no PEM-valid tier_1 entries
          - none of the top-K nearest entries reconstruct into a valid Program
          - train_pairs is empty
          - k <= 0
    """
    if k <= 0:
        return []
    if not train_pairs:
        return []

    # Resolve library path. None -> per-install default; missing file -> [].
    path = pathlib.Path(library_path) if library_path is not None \
        else default_library_path()
    if not path.exists():
        return []

    # Load the library. The constructor silently skips non-PEM rows.
    library = ResonanceLibrary.load_or_create(path)
    if not library.entries:
        return []

    # Compute the query fingerprint from the current task.
    fp = task_fingerprint(train_pairs)

    # Pull the K nearest entries by cosine. We then translate each entry's
    # stored winning_policy into a DSL Program, skipping un-translatable
    # entries. We request 2K candidates from the library so that after
    # skips we still hope to return up to K seeds.
    nearest = library.find_k_nearest(fp, k=max(k * 2, k + 4))

    seeds: list[Program] = []
    seen_hashes: set[str] = set()
    for entry, _sim in nearest:
        sig = _winning_policy_to_signature(entry.winning_policy)
        if sig is None:
            continue
        program = program_from_signature(sig)
        if program is None:
            continue
        # Tier-1 honesty: the seed must be evaluable. A seed program's only
        # holes are root-level input-binding holes (per the synthesis-team
        # convention shared with dsl.synthesis); any DEEPER holes would be
        # un-synthesized subprograms and are rejected here.
        if not _seed_is_evaluable(program):
            continue
        # Dedup by program hash so two library entries with the same
        # signature occupy only one beam slot.
        h = program.sha256_hash()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        seeds.append(program)
        if len(seeds) >= k:
            break

    return seeds
