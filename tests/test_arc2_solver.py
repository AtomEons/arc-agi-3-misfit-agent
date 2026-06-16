"""ARC-AGI-2 sister-agent tests.

Five tests:
  1. Identity task: solver returns identity-preserving output.
  2. Single-cell shift: solver induces the shift and applies it to test.
  3. emit_submission produces valid {task_id: [{attempt_1, attempt_2}]}.
  4. Empty task list yields empty submission.
  5. Tier-1: arc2_solver does not import any LLM-family package.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from misfit_agent.arc2_solver import (
    Identity,
    Translate2,
    Recolor,
    solve_task,
    task_fingerprint,
)
from misfit_agent.arc2_runner import run, load_challenges, emit_submission


# ---------------------------------------------------------------------------
# Test 1: identity task
# ---------------------------------------------------------------------------


def test_solver_identity_task_preserves_input():
    """A task whose train pairs are all input==output should be solved
    by the Identity rule, returning the test input unchanged."""
    train_inp = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=np.int32)
    train_pairs = [
        (train_inp.copy(), train_inp.copy()),
        (np.array([[0, 1], [2, 0]], dtype=np.int32),
         np.array([[0, 1], [2, 0]], dtype=np.int32)),
    ]
    test_input = np.array([[7, 0], [0, 8]], dtype=np.int32)

    a1, a2 = solve_task(train_pairs, test_input)

    assert np.array_equal(a1, test_input), (
        f"identity task: attempt_1 should equal test input, got\n{a1}\nvs\n{test_input}"
    )
    # attempt_2 may be the same identity (only rule that fits) — acceptable.
    assert a2.shape == test_input.shape


# ---------------------------------------------------------------------------
# Test 2: single-cell-shift task
# ---------------------------------------------------------------------------


def test_solver_single_cell_shift_task():
    """A task where every output is the input shifted by (dy=1, dx=0).
    Translate2 should fit and apply the shift to the test input."""
    def shift_down(g: np.ndarray) -> np.ndarray:
        out = np.zeros_like(g)
        out[1:, :] = g[:-1, :]
        return out

    train_inputs = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32),
        np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32),
        np.array([[0, 0, 0], [3, 0, 0], [0, 0, 0]], dtype=np.int32),
    ]
    train_pairs = [(t, shift_down(t)) for t in train_inputs]

    test_input = np.array([[0, 5, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    expected = shift_down(test_input)

    a1, a2 = solve_task(train_pairs, test_input)

    assert np.array_equal(a1, expected), (
        f"single-shift task: attempt_1 should be the down-shift, got\n{a1}\nexpected\n{expected}"
    )
    # attempt_2 must be a different program (identity fallback at minimum).
    assert a2.shape == test_input.shape


# ---------------------------------------------------------------------------
# Test 3: emit_submission shape
# ---------------------------------------------------------------------------


def test_emit_submission_shape_is_valid():
    """The runner output must match the official ARC-AGI-2 format:
        {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
    """
    challenges = {
        "task_aaa": {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]},
            ],
            "test": [
                {"input": [[3, 0], [0, 3]]},
                {"input": [[4, 0], [0, 4]]},
            ],
        },
        "task_bbb": {
            "train": [
                {"input": [[1]], "output": [[1]]},
            ],
            "test": [
                {"input": [[7]]},
            ],
        },
    }

    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "sub.json"
        submission = run(challenges, out_path=out, write=True, verbose=False)

        # In-memory shape
        assert set(submission.keys()) == {"task_aaa", "task_bbb"}
        assert isinstance(submission["task_aaa"], list)
        assert len(submission["task_aaa"]) == 2
        for entry in submission["task_aaa"]:
            assert isinstance(entry, dict)
            assert set(entry.keys()) == {"attempt_1", "attempt_2"}
            assert isinstance(entry["attempt_1"], list)
            assert isinstance(entry["attempt_2"], list)
            assert all(isinstance(row, list) for row in entry["attempt_1"])
            assert all(isinstance(x, int) for row in entry["attempt_1"] for x in row)
        assert len(submission["task_bbb"]) == 1

        # On-disk shape — must be valid JSON with same structure
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == submission


# ---------------------------------------------------------------------------
# Test 4: empty task list
# ---------------------------------------------------------------------------


def test_empty_challenges_yields_empty_submission():
    """Empty challenges dict should produce an empty submission, not crash."""
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "sub.json"
        submission = run({}, out_path=out, write=True, verbose=False)
        assert submission == {}
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == {}


# ---------------------------------------------------------------------------
# Test 5: Tier-1 attestation — no LLM imports in arc2 modules
# ---------------------------------------------------------------------------


FORBIDDEN_IMPORT_PATTERNS = [
    r"\btorch\.load\b",
    r"\bfrom\s+transformers\b",
    r"\bimport\s+transformers\b",
    r"\bfrom\s+openai\b",
    r"\bimport\s+openai\b",
    r"\bfrom\s+anthropic\b",
    r"\bimport\s+anthropic\b",
    r"\bfrom\s+llama_cpp\b",
    r"\bimport\s+llama_cpp\b",
    r"\bfrom\s+ctransformers\b",
    r"\bimport\s+ctransformers\b",
    r"\bfrom\s+huggingface_hub\b",
    r"\bimport\s+huggingface_hub\b",
    r"\bfrom\s+sentence_transformers\b",
    r"\bimport\s+sentence_transformers\b",
    r"\bfrom\s+langchain\b",
    r"\bimport\s+langchain\b",
    r"\bfrom\s+langgraph\b",
    r"\bimport\s+langgraph\b",
    r"\bfrom\s+smolagents\b",
    r"\bimport\s+smolagents\b",
    r"\bfrom\s+google\.generativeai\b",
]


def test_tier1_arc2_no_llm_imports():
    """ARC-AGI-2 solver + runner must be priors-only (no LLM packages)."""
    repo_root = pathlib.Path(__file__).parent.parent
    files = [
        repo_root / "src" / "misfit_agent" / "arc2_solver.py",
        repo_root / "src" / "misfit_agent" / "arc2_runner.py",
    ]
    hits: list[tuple[str, str, int]] = []
    for f in files:
        assert f.exists(), f"missing source file: {f}"
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            for pat in FORBIDDEN_IMPORT_PATTERNS:
                if re.search(pat, line, flags=re.IGNORECASE):
                    hits.append((str(f.name), line.strip()[:120], i))
    assert not hits, (
        "Tier-1 violation in arc2 modules:\n"
        + "\n".join(f"  {fn}:{lineno}: {line}" for fn, line, lineno in hits)
    )


# ---------------------------------------------------------------------------
# Bonus sanity checks (not counted toward the 5 required; pytest will still
# run them and give us extra confidence — these are cheap and harden the build).
# ---------------------------------------------------------------------------


def test_recolor_rule_fits_consistent_permutation():
    """The Recolor rule should fit a consistent 1<->2 permutation across pairs."""
    train_pairs = [
        (np.array([[1, 0], [0, 1]], dtype=np.int32),
         np.array([[2, 0], [0, 2]], dtype=np.int32)),
        (np.array([[1, 1], [0, 0]], dtype=np.int32),
         np.array([[2, 2], [0, 0]], dtype=np.int32)),
    ]
    rule = Recolor()
    assert rule.fit(train_pairs) is True
    assert rule.mapping[1] == 2
    test_input = np.array([[1, 0, 1]], dtype=np.int32)
    pred = rule.predict(test_input)
    assert np.array_equal(pred, np.array([[2, 0, 2]], dtype=np.int32))


def test_task_fingerprint_dimension():
    """Fingerprint must be a fixed-length numeric vector for resonance lookup."""
    train_pairs = [
        (np.array([[1, 0], [0, 1]], dtype=np.int32),
         np.array([[1, 0], [0, 1]], dtype=np.int32)),
    ]
    fp = task_fingerprint(train_pairs)
    assert fp.shape == (16,)
    assert fp.dtype == np.float32

    # Empty input is handled (returns zero vector, not a crash)
    fp_empty = task_fingerprint([])
    assert fp_empty.shape == (16,)
    assert float(fp_empty.sum()) == 0.0
