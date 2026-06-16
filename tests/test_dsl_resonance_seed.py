"""TEAM RESONANCE-SEED — resonance-seeded synthesis initialization.

The required guarantees, mechanically tested:
  - seed_from_resonance returns empty list when no library exists
  - seed_from_resonance returns at most k programs
  - Each seed is a valid Program (every internal node bound; only the root
    input-port Grid hole permitted, per the dsl.synthesis convention)
  - task_fingerprint returns a 16-dim float array
  - Two identical train_pairs yield identical fingerprints

Extra coverage that earns its place:
  - seed_from_resonance returns [] on empty train_pairs
  - seed_from_resonance honors k=0 (returns [])
  - signature reconstruction round-trips for every supported rule head
  - unknown signature heads are skipped (None), not error
  - Composed signatures are skipped (atomic-only first cut)
  - seeds returned from a real library are evaluable via interpreter.evaluate
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.dsl.resonance_seed import (
    seed_from_resonance,
    task_fingerprint,
    program_from_signature,
)
from misfit_agent.dsl.ast import Program, PrimitiveNode, HoleNode
from misfit_agent.dsl.primitives import (
    Identity, Translate, Rotate, Reflect, Recolor, Crop, Tile,
    Gravity, Symmetrize, KeepWhere,
)
from misfit_agent.dsl.interpreter import evaluate
from misfit_agent.dsl.types import DslType


# ---------------------------------------------------------------------------
# Library file helpers
# ---------------------------------------------------------------------------


def _make_pem_entry(fingerprint, signature, game_id="seed_test",
                    solved_at_unix=1_700_000_000.0):
    """Construct a PEM-valid library row that stores its signature in the
    `winning_policy` field. This matches the sister-solver-friendly schema
    that the resonance team will adopt for ARC-AGI-2.
    """
    fp_list = [float(x) for x in fingerprint.tolist()] if hasattr(
        fingerprint, "tolist") else list(fingerprint)
    return {
        "source": "self-solved",
        "contamination_tier": "tier_1",
        "solved_at_unix": solved_at_unix,
        "episode_signature": "abcdef1234567890",
        "replay_pointer": "kernel_version:test|game_id:" + game_id,
        "mutation_history": [],
        "expiry_decay_rule": "never_decay",
        "fingerprint": fp_list,
        "winning_policy": [{"signature": list(signature)}],
        "evidence_grid_hash": "deadbeefdeadbeef",
        "composite_score": 1.0,
        "usage_receipts": [],
        "game_id": game_id,
    }


def _write_library(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _identity_train_pairs(n: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """A small Identity task — input == output."""
    pairs = []
    rng = np.random.default_rng(seed=11)
    for i in range(n):
        g = rng.integers(low=0, high=5, size=(3 + i, 3 + i),
                         dtype=np.int32)
        pairs.append((g, g.copy()))
    return pairs


def _rotate_train_pairs(k: int = 2,
                        n: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """A small Rotate task — output = np.rot90(input, k=k)."""
    pairs = []
    rng = np.random.default_rng(seed=23)
    for i in range(n):
        g = rng.integers(low=0, high=5, size=(3 + i, 3 + i),
                         dtype=np.int32)
        pairs.append((g, np.rot90(g, k=k).copy()))
    return pairs


# ---------------------------------------------------------------------------
# Required: missing library -> []
# ---------------------------------------------------------------------------


def test_seed_returns_empty_when_no_library_exists(tmp_path: Path):
    """A nonexistent library path is a cold-start, not an error."""
    missing = tmp_path / "no_such_library.jsonl"
    assert not missing.exists()
    seeds = seed_from_resonance(_identity_train_pairs(), library_path=missing,
                                 k=5)
    assert seeds == []


def test_seed_returns_empty_when_train_pairs_empty(tmp_path: Path):
    """Empty train pairs is a no-op — no query, no seeds."""
    lib = tmp_path / "lib.jsonl"
    fp = np.zeros(16, dtype=np.float32)
    _write_library(lib, [_make_pem_entry(fp, ("Identity",))])
    seeds = seed_from_resonance([], library_path=lib, k=5)
    assert seeds == []


def test_seed_returns_empty_when_k_is_zero(tmp_path: Path):
    """k=0 trivially produces no seeds."""
    lib = tmp_path / "lib.jsonl"
    train = _identity_train_pairs()
    fp = task_fingerprint(train)
    _write_library(lib, [_make_pem_entry(fp, ("Identity",))])
    assert seed_from_resonance(train, library_path=lib, k=0) == []


# ---------------------------------------------------------------------------
# Required: at most k programs
# ---------------------------------------------------------------------------


def test_seed_returns_at_most_k_programs(tmp_path: Path):
    """A library with 10 distinct winning signatures must still be capped
    at k=3 when k=3 is requested."""
    lib = tmp_path / "lib.jsonl"
    train = _identity_train_pairs()
    fp = task_fingerprint(train)
    # Build 10 distinct signatures.
    signatures = [
        ("Identity",),
        ("Translate2", 1, 0),
        ("Translate2", 0, 1),
        ("Translate2", -1, 0),
        ("Translate2", 0, -1),
        ("Rotate", 1),
        ("Rotate", 2),
        ("Rotate", 3),
        ("ReflectH",),
        ("ReflectV",),
    ]
    entries = [_make_pem_entry(fp, s, game_id=f"task_{i}")
               for i, s in enumerate(signatures)]
    _write_library(lib, entries)

    for k in (1, 2, 3, 5):
        seeds = seed_from_resonance(train, library_path=lib, k=k)
        assert isinstance(seeds, list)
        assert len(seeds) <= k, \
            f"asked for k={k} seeds, got {len(seeds)}"


# ---------------------------------------------------------------------------
# Required: each seed is a valid Program
# ---------------------------------------------------------------------------


def test_each_seed_is_a_valid_program(tmp_path: Path):
    """Returned objects are Program instances with PrimitiveNode roots.
    Their only permitted hole is the root-level Grid input port (mirroring
    the cold-start synthesizer's convention)."""
    lib = tmp_path / "lib.jsonl"
    train = _identity_train_pairs()
    fp = task_fingerprint(train)
    signatures = [
        ("Identity",),
        ("Rotate", 2),
        ("ReflectH",),
        ("Translate2", 0, 1),
        ("CropToBbox",),
    ]
    _write_library(lib, [_make_pem_entry(fp, s, game_id=f"g{i}")
                          for i, s in enumerate(signatures)])
    seeds = seed_from_resonance(train, library_path=lib, k=5)
    assert len(seeds) > 0, "expected at least one seed from a 5-entry library"

    for seed in seeds:
        # Type check
        assert isinstance(seed, Program), \
            f"seed must be a Program, got {type(seed).__name__}"
        assert isinstance(seed.root, PrimitiveNode), \
            f"seed root must be a PrimitiveNode, got {type(seed.root).__name__}"

        # Output type must be Grid (every DSL-translatable rule we accept
        # is Grid->Grid).
        assert seed.output_type() == DslType.GRID, \
            f"seed must be Grid->Grid; got output={seed.output_type()}"

        # The only allowed hole is a root-level Grid input port. No deeper
        # holes — those would indicate an un-synthesized subprogram.
        _assert_no_deep_holes(seed.root)

        # Round-trip: the seed must be evaluable on the first train input
        # via the interpreter, producing a Grid.
        out = evaluate(seed, train[0][0])
        assert isinstance(out, np.ndarray), \
            f"seed evaluation must yield an ndarray (Grid); got {type(out)}"


def _assert_no_deep_holes(node) -> None:
    """Walk the AST and fail if any non-root hole appears. The root
    PrimitiveNode is permitted exactly one Grid hole at depth 1."""
    if isinstance(node, PrimitiveNode):
        for child in node.children:
            if isinstance(child, HoleNode):
                # Allowed at depth-1 only.
                assert child.expected_type == DslType.GRID, \
                    "permitted leaf hole must be Grid-typed"
                continue
            if isinstance(child, PrimitiveNode):
                # Deeper PrimitiveNodes must themselves be hole-free.
                for sub in child.children:
                    assert not isinstance(sub, HoleNode), \
                        "deep holes are not permitted in a seed program"


# ---------------------------------------------------------------------------
# Required: task_fingerprint returns 16-dim float array
# ---------------------------------------------------------------------------


def test_task_fingerprint_returns_16dim_float_array():
    train = _identity_train_pairs()
    fp = task_fingerprint(train)
    assert isinstance(fp, np.ndarray), \
        f"fingerprint must be ndarray; got {type(fp).__name__}"
    assert fp.shape == (16,), \
        f"fingerprint must be 16-dim; got shape {fp.shape}"
    assert np.issubdtype(fp.dtype, np.floating), \
        f"fingerprint must be float; got dtype {fp.dtype}"


def test_task_fingerprint_empty_pairs_returns_zero_vector():
    """An empty task has no observable signal — return the zero vector,
    not None or an exception."""
    fp = task_fingerprint([])
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (16,)
    assert np.all(fp == 0.0)


# ---------------------------------------------------------------------------
# Required: two identical train_pairs yield identical fingerprints
# ---------------------------------------------------------------------------


def test_identical_train_pairs_yield_identical_fingerprints():
    """Determinism: fingerprint must be a pure function of the train pairs.
    Different (deep-copied) inputs with the same data produce the same vec.
    """
    train_a = _rotate_train_pairs(k=2, n=3)
    train_b = [(inp.copy(), out.copy()) for inp, out in train_a]
    fp_a = task_fingerprint(train_a)
    fp_b = task_fingerprint(train_b)
    assert np.array_equal(fp_a, fp_b), \
        f"identical inputs must yield identical fingerprints; got " \
        f"diff norm = {float(np.linalg.norm(fp_a - fp_b))}"


def test_different_train_pairs_yield_different_fingerprints():
    """Sanity counter-test: meaningfully different tasks shouldn't collide."""
    fp_id = task_fingerprint(_identity_train_pairs())
    fp_rot = task_fingerprint(_rotate_train_pairs(k=2, n=2))
    # They might overlap on a few dimensions but not be byte-identical.
    assert not np.array_equal(fp_id, fp_rot), \
        "identity and rotate tasks should not produce identical fingerprints"


# ---------------------------------------------------------------------------
# Signature -> Program reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig, prim_cls, attrs", [
    (("Identity",), Identity, {}),
    (("Translate2", 1, -2), Translate, {"dy": 1, "dx": -2}),
    (("Rotate", 1), Rotate, {"k": 1}),
    (("Rotate", 2), Rotate, {"k": 2}),
    (("Rotate", 3), Rotate, {"k": 3}),
    (("ReflectH",), Reflect, {"axis": "H"}),
    (("ReflectV",), Reflect, {"axis": "V"}),
    (("Transpose",), Reflect, {"axis": "D1"}),
    (("Recolor", ((1, 2), (3, 4))), Recolor, {"mapping": {1: 2, 3: 4}}),
    (("CropToBbox",), Crop, {}),
    (("Tile", 2, 3), Tile, {"rf": 2, "cf": 3}),
    (("Translate", -1, 1), Translate, {"dy": -1, "dx": 1}),
    (("Reflect", "D2"), Reflect, {"axis": "D2"}),
    (("Gravity", "D"), Gravity, {"direction": "D"}),
    (("Symmetrize", "BOTH"), Symmetrize, {"axis": "BOTH"}),
    (("KeepWhere", "largest"), KeepWhere, {"predicate": "largest"}),
])
def test_program_from_signature_supports_every_known_head(sig, prim_cls, attrs):
    program = program_from_signature(sig)
    assert program is not None, f"signature {sig} should reconstruct"
    assert isinstance(program.root, PrimitiveNode)
    prim = program.root.primitive
    assert isinstance(prim, prim_cls), \
        f"expected {prim_cls.__name__}, got {type(prim).__name__}"
    for k, v in attrs.items():
        assert getattr(prim, k) == v, \
            f"primitive attribute {k}: expected {v}, got {getattr(prim, k)}"


@pytest.mark.parametrize("sig", [
    (),
    None,
    ("UnknownHead",),
    ("Rotate", 0),
    ("Rotate", 7),
    ("Rotate", "not-an-int"),
    ("Translate2", 1),        # wrong arity
    ("Translate2", 1, 2, 3),  # wrong arity
    ("Tile", 0, 0),           # bad factors
    ("Recolor", ()),          # empty mapping
    ("Recolor", "not-a-tuple"),
    ("Recolor", ((1, 99),)),  # out-of-range color
    ("Reflect", "Q"),         # bad axis
    ("Gravity", "X"),         # bad direction
    ("Symmetrize", "X"),      # bad axis
    ("KeepWhere", "biggest"), # not a recognized predicate
    ("Composed", ("Identity",), ("Rotate", 2)),  # composed skipped
])
def test_program_from_signature_returns_none_for_unsupported(sig):
    """Soft-fail: unknown / malformed / out-of-scope signatures yield None,
    never an exception. The seed loop then silently skips them."""
    result = program_from_signature(sig)
    assert result is None, \
        f"signature {sig!r} should return None; got {result!r}"


# ---------------------------------------------------------------------------
# End-to-end: library round-trip produces evaluable seeds
# ---------------------------------------------------------------------------


def test_end_to_end_rotate_seed_evaluates_correctly(tmp_path: Path):
    """Plant a Rotate(k=2) entry in the library; query with a Rotate(k=2)
    task; verify the returned seed actually predicts the correct output."""
    lib = tmp_path / "lib.jsonl"
    train = _rotate_train_pairs(k=2, n=2)
    fp = task_fingerprint(train)
    _write_library(lib, [
        _make_pem_entry(fp, ("Rotate", 2), game_id="planted_rot2"),
    ])
    seeds = seed_from_resonance(train, library_path=lib, k=1)
    assert len(seeds) == 1, f"expected 1 seed; got {len(seeds)}"
    seed = seeds[0]
    assert isinstance(seed.root.primitive, Rotate)
    assert seed.root.primitive.k == 2

    # Verify the seed actually produces the right output on every train pair.
    for inp, out in train:
        pred = evaluate(seed, inp)
        assert np.array_equal(pred, out), \
            "Rotate(k=2) seed must reproduce the training output"


def test_legacy_action_record_winning_policy_is_skipped(tmp_path: Path):
    """Legacy ARC-AGI-3 winning_policy entries (ActionRecord dicts) cannot
    be translated into a DSL program — they must be silently skipped, not
    crash the seeder."""
    lib = tmp_path / "lib.jsonl"
    train = _identity_train_pairs()
    fp = task_fingerprint(train)
    # Legacy schema row: winning_policy is a list of ActionRecord dicts,
    # no "signature" field anywhere.
    legacy_entry = _make_pem_entry(fp, ("Identity",))
    legacy_entry["winning_policy"] = [{
        "action_name": "ACTION6",
        "action_value": 6,
        "data": {"x": 5, "y": 5},
        "pre_levels_completed": 0,
        "post_levels_completed": 1,
        "cells_changed": 12,
        "triggered_win": True,
    }]
    # And a clean DSL-translatable row alongside it.
    clean_entry = _make_pem_entry(fp, ("Identity",), game_id="clean")
    _write_library(lib, [legacy_entry, clean_entry])

    seeds = seed_from_resonance(train, library_path=lib, k=5)
    # We must not crash, AND we must still surface the clean entry.
    assert len(seeds) == 1, \
        f"clean entry should produce one seed; got {len(seeds)}"
    assert isinstance(seeds[0].root.primitive, Identity)
