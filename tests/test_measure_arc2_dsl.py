"""Tests for scripts/measure_arc2_dsl.py — the DSL-engine measurement driver.

Covers:
- The script is importable as a module
- run_one_task on an identity task returns correct=True
- Empty split (no tasks) returns 0/0 with no error
- Budget exceeded returns early with the partial result
- Receipt file is valid JSON-lines (one JSON object per line)
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for p in (str(SRC), str(SCRIPTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_module():
    """Import scripts/measure_arc2_dsl.py as a module."""
    # Try package import first (scripts as a namespace package), then file-based.
    try:
        return importlib.import_module("measure_arc2_dsl")
    except ModuleNotFoundError:
        pass
    try:
        return importlib.import_module("scripts.measure_arc2_dsl")
    except ModuleNotFoundError:
        pass
    spec = importlib.util.spec_from_file_location(
        "measure_arc2_dsl", SCRIPTS / "measure_arc2_dsl.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_importable():
    mod = _import_module()
    assert hasattr(mod, "run_one_task")
    assert hasattr(mod, "measure")
    assert hasattr(mod, "main")


def test_run_one_task_identity():
    """Synthesize on an identity task — solver should produce Identity, attempt_1 correct."""
    mod = _import_module()
    g1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    g2 = np.array([[5, 6, 7], [8, 9, 0]], dtype=np.int32)
    train_pairs = [(g1, g1.copy()), (g2, g2.copy())]
    test_input = np.array([[1, 1], [2, 2]], dtype=np.int32)
    gold = test_input.copy()
    result = mod.run_one_task(
        train_pairs=train_pairs,
        test_inputs=[test_input],
        golds=[gold],
        budget_per_task=3.0,
        max_depth=2,
        beam_width=4,
    )
    assert result["test_inputs"] == 1
    assert result["test_inputs_solved"] == 1
    assert result["task_solved_any"] is True
    assert result["task_solved_all"] is True
    assert result["per_input"][0]["any_correct"] is True


def test_empty_split_returns_zero_no_error(tmp_path):
    """measure() with an empty challenge dict should produce a valid 0/0 footer."""
    mod = _import_module()
    # Monkey-patch _load_split to return empty dicts.
    original = mod._load_split
    try:
        mod._load_split = lambda split: ({}, {}, tmp_path / "dummy.json")
        out_path = tmp_path / "empty_receipt.jsonl"
        footer = mod.measure(
            split="training",
            limit=10,
            budget_per_task=0.5,
            max_depth=1,
            beam_width=2,
            out_path=out_path,
        )
    finally:
        mod._load_split = original
    assert footer["tasks_total"] == 0
    assert footer["tasks_solved_all"] == 0
    assert footer["tasks_solved_any"] == 0
    assert footer["test_inputs_total"] == 0
    assert footer["task_solve_rate_all"] == 0.0
    assert out_path.exists()


def test_budget_exceeded_returns_partial():
    """A very small budget should still return a structurally valid result."""
    mod = _import_module()
    # Use a non-trivial pair so synthesis takes some real time.
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    train_pairs = [(g, g.copy())]
    test_input = g.copy()
    gold = g.copy()
    t0 = time.time()
    result = mod.run_one_task(
        train_pairs=train_pairs,
        test_inputs=[test_input],
        golds=[gold],
        budget_per_task=0.001,  # microscopic budget
        max_depth=1,
        beam_width=1,
    )
    elapsed = time.time() - t0
    # Result must be structurally valid even with a tiny budget.
    assert "task_solved_all" in result
    assert "task_solved_any" in result
    assert "wall_clock_s" in result
    assert isinstance(result["per_input"], list)
    assert result["test_inputs"] == 1
    # Should not run for many seconds despite the tiny budget.
    assert elapsed < 10.0


def test_receipt_is_valid_jsonl(tmp_path):
    """Every line of the receipt must parse as JSON."""
    mod = _import_module()
    # Build a tiny synthetic split: 2 identity tasks.
    g1 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    g2 = np.array([[2, 2], [3, 3]], dtype=np.int32)
    fake_challenges = {
        "task_a": {
            "train": [{"input": g1.tolist(), "output": g1.tolist()}],
            "test": [{"input": g1.tolist()}],
        },
        "task_b": {
            "train": [{"input": g2.tolist(), "output": g2.tolist()}],
            "test": [{"input": g2.tolist()}],
        },
    }
    fake_solutions = {
        "task_a": [g1.tolist()],
        "task_b": [g2.tolist()],
    }
    original = mod._load_split
    try:
        mod._load_split = lambda split: (fake_challenges, fake_solutions, tmp_path / "x.json")
        out_path = tmp_path / "valid_receipt.jsonl"
        footer = mod.measure(
            split="training",
            limit=10,
            budget_per_task=2.0,
            max_depth=2,
            beam_width=4,
            out_path=out_path,
        )
    finally:
        mod._load_split = original

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3  # header + 2 task rows + footer
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["kind"] == "measurement_header"
    assert parsed[-1]["kind"] == "measurement_footer"
    task_rows = [r for r in parsed if r.get("kind") == "task"]
    assert len(task_rows) == 2
    # Footer aggregates must match the task rows.
    assert footer["tasks_total"] == 2
    assert footer["tasks_solved_all"] >= 1  # at least one identity task should solve
