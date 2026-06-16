"""Tests for scripts/ablate_resonance.py — TEAM ABLATION-RESONANCE.

Required coverage from the brief:
  - Library accumulation phase produces a non-empty library
  - Condition A runs without library reads (verified by mocking the
    library file)
  - Condition B reads from the library
  - The delta calculation is correct
  - Receipt has all expected fields
  - Markdown output is well-formed

We exercise the harness on a synthetic 10-task subset (8 accumulation + 2
held-out) made of identity / translate / recolor / reflect tasks the DSL
can actually solve. This keeps the test suite under 30 seconds while still
testing every code path the real 1000-task run will execute.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_module():
    """Import scripts/ablate_resonance.py as a module."""
    try:
        return importlib.import_module("ablate_resonance")
    except ModuleNotFoundError:
        pass
    try:
        return importlib.import_module("scripts.ablate_resonance")
    except ModuleNotFoundError:
        pass
    spec = importlib.util.spec_from_file_location(
        "ablate_resonance", SCRIPTS / "ablate_resonance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic task fixtures
# ---------------------------------------------------------------------------


def _identity_task(task_id: str, grid_list: list[list[int]]) -> dict:
    """Task whose output == input. Solvable by Identity primitive."""
    grid = grid_list
    return {
        "_id": task_id,
        "train": [
            {"input": grid, "output": grid},
            {"input": grid, "output": grid},
        ],
        "test": [{"input": grid}],
        "_gold": [grid],
    }


def _reflect_h_task(task_id: str, grid_list: list[list[int]]) -> dict:
    """Task whose output is reflection along H axis. Solvable by Reflect."""
    grid = np.asarray(grid_list, dtype=np.int32)
    out = np.flipud(grid).tolist()
    inp = grid.tolist()
    return {
        "_id": task_id,
        "train": [
            {"input": inp, "output": out},
            {"input": inp, "output": out},
        ],
        "test": [{"input": inp}],
        "_gold": [out],
    }


def _make_synthetic_dataset() -> tuple[dict, dict]:
    """Build a tiny challenges/solutions pair with 10 tasks the DSL can solve.

    Mix of identity and reflection so the accumulation phase will produce
    library entries with atomic-translatable signatures (Identity / ReflectH).
    """
    g1 = [[1, 2], [3, 4]]
    g2 = [[1, 0, 1], [0, 1, 0]]
    g3 = [[2, 2], [3, 3]]
    g4 = [[5, 6, 7], [8, 9, 0]]
    g5 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    tasks: list[dict] = [
        _identity_task("task_aa_id", g1),
        _identity_task("task_ab_id", g2),
        _identity_task("task_ac_id", g3),
        _identity_task("task_ad_id", g4),
        _identity_task("task_ae_id", g5),
        _reflect_h_task("task_af_rh", g1),
        _reflect_h_task("task_ag_rh", g2),
        _reflect_h_task("task_ah_rh", g3),
        _identity_task("task_ai_id", g4),  # held-out
        _identity_task("task_aj_id", g5),  # held-out
    ]

    challenges: dict = {}
    solutions: dict = {}
    for t in tasks:
        tid = t.pop("_id")
        gold = t.pop("_gold")
        challenges[tid] = t
        solutions[tid] = gold
    return challenges, solutions


@pytest.fixture
def synthetic_dataset():
    return _make_synthetic_dataset()


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------


def test_module_importable():
    mod = _import_module()
    # Public API the test brief depends on.
    assert hasattr(mod, "run_ablation")
    assert hasattr(mod, "accumulate_library")
    assert hasattr(mod, "run_condition")
    assert hasattr(mod, "deterministic_split")
    assert hasattr(mod, "classify_verdict")
    assert hasattr(mod, "render_markdown")
    assert hasattr(mod, "main")


def test_deterministic_split_is_stable():
    mod = _import_module()
    ids = ["t_b", "t_a", "t_c", "t_e", "t_d"]
    a1, h1, s1 = mod.deterministic_split(ids, accumulation_size=3,
                                           holdout_size=2)
    a2, h2, s2 = mod.deterministic_split(ids, accumulation_size=3,
                                           holdout_size=2)
    assert a1 == ["t_a", "t_b", "t_c"]
    assert h1 == ["t_d", "t_e"]
    assert a1 == a2 and h1 == h2 and s1 == s2  # deterministic
    # Re-shuffling the input does NOT change the split (sorted internally).
    a3, h3, s3 = mod.deterministic_split(["t_e", "t_d", "t_c", "t_b", "t_a"],
                                          accumulation_size=3, holdout_size=2)
    assert a3 == a1 and h3 == h1 and s3 == s1


def test_deterministic_split_rejects_negative_sizes():
    mod = _import_module()
    with pytest.raises(ValueError):
        mod.deterministic_split(["a", "b"], accumulation_size=-1,
                                  holdout_size=1)
    with pytest.raises(ValueError):
        mod.deterministic_split(["a", "b"], accumulation_size=1,
                                  holdout_size=-1)


# ---------------------------------------------------------------------------
# Accumulation phase
# ---------------------------------------------------------------------------


def test_accumulation_produces_non_empty_library(tmp_path, synthetic_dataset):
    """Brief: 'Library accumulation phase produces a non-empty library'."""
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    accumulation_ids = sorted(challenges.keys())[:8]
    lib_path = tmp_path / "accum_library.jsonl"

    stats = mod.accumulate_library(
        accumulation_ids=accumulation_ids,
        challenges=challenges,
        solutions=solutions,
        library_path=lib_path,
        budget_per_task=2.0,
        max_depth=1,
        beam_width=4,
        refine_max_iters=1,
        verbose=False,
    )

    assert stats["tasks_attempted"] == 8
    assert stats["tasks_solved"] >= 1, "at least one identity task should solve"
    assert stats["entries_written"] >= 1
    # The library file must exist on disk after the flush.
    assert lib_path.exists(), "accumulation must flush library to disk"
    # Each line must be valid JSON with a PEM-bound shape.
    lines = lib_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    for ln in lines:
        d = json.loads(ln)
        assert d["source"] == "self-solved"
        assert d["contamination_tier"] == "tier_1"
        assert "fingerprint" in d
        assert isinstance(d["winning_policy"], list)


def test_accumulation_atomic_entries_seedable(tmp_path, synthetic_dataset):
    """At least one accumulated entry must round-trip through the seed loader
    so Condition B is non-trivially testable."""
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    accumulation_ids = sorted(challenges.keys())[:8]
    lib_path = tmp_path / "accum_library.jsonl"

    stats = mod.accumulate_library(
        accumulation_ids=accumulation_ids,
        challenges=challenges,
        solutions=solutions,
        library_path=lib_path,
        budget_per_task=2.0,
        max_depth=1,
        beam_width=4,
        refine_max_iters=1,
        verbose=False,
    )
    assert stats["atomic_entries_written"] >= 1, (
        "no atomic-translatable entries means resonance can never compound"
    )


# ---------------------------------------------------------------------------
# Condition A — resonance OFF
# ---------------------------------------------------------------------------


def test_condition_a_skips_library_reads(tmp_path, synthetic_dataset):
    """Brief: 'Condition A runs without library reads (verify by mocking
    the library file).'

    We point the harness at a library path that does NOT exist AND we
    monkey-patch seed_from_resonance to raise if called. If Condition A
    is honest, the run completes without ever calling the seed loader.
    """
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    holdout_ids = sorted(challenges.keys())[8:10]
    bogus_library = tmp_path / "this_file_does_not_exist.jsonl"
    assert not bogus_library.exists()

    call_log: list[str] = []

    def _spy_seed_from_resonance(*args, **kwargs):
        call_log.append("called")
        raise AssertionError(
            "Condition A must not consult seed_from_resonance"
        )

    with mock.patch.object(mod, "seed_from_resonance",
                            _spy_seed_from_resonance):
        result = mod.run_condition(
            holdout_ids=holdout_ids,
            challenges=challenges,
            solutions=solutions,
            library_path=bogus_library,
            use_resonance=False,
            budget_per_task=2.0,
            max_depth=1,
            beam_width=4,
            resonance_k=0,
            refine_max_iters=1,
            verbose=False,
        )

    assert call_log == [], "Condition A leaked a library read"
    # Structure: result must be aggregate dict.
    assert "solved_count" in result
    assert "mean_wall_clock_s" in result
    assert "seeded_task_count" in result
    assert result["seeded_task_count"] == 0
    assert result["false_rhyme_failures"] == 0
    assert isinstance(result["per_task"], list)
    assert len(result["per_task"]) == len(holdout_ids)


# ---------------------------------------------------------------------------
# Condition B — resonance ON
# ---------------------------------------------------------------------------


def test_condition_b_reads_from_library(tmp_path, synthetic_dataset):
    """Brief: 'Condition B reads from library'."""
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    accumulation_ids = sorted(challenges.keys())[:8]
    holdout_ids = sorted(challenges.keys())[8:10]
    lib_path = tmp_path / "library.jsonl"

    # First populate the library so there's something to read.
    mod.accumulate_library(
        accumulation_ids=accumulation_ids,
        challenges=challenges,
        solutions=solutions,
        library_path=lib_path,
        budget_per_task=2.0,
        max_depth=1,
        beam_width=4,
        refine_max_iters=1,
    )
    assert lib_path.exists()

    call_log: list[dict] = []
    real_seed = mod.seed_from_resonance

    def _spy_seed_from_resonance(train_pairs, library_path=None, k=5):
        call_log.append({"library_path": str(library_path), "k": k})
        return real_seed(train_pairs, library_path=library_path, k=k)

    with mock.patch.object(mod, "seed_from_resonance",
                            _spy_seed_from_resonance):
        result = mod.run_condition(
            holdout_ids=holdout_ids,
            challenges=challenges,
            solutions=solutions,
            library_path=lib_path,
            use_resonance=True,
            budget_per_task=2.0,
            max_depth=1,
            beam_width=4,
            resonance_k=5,
            refine_max_iters=1,
        )

    assert len(call_log) == len(holdout_ids), (
        "Condition B must call seed_from_resonance once per holdout task"
    )
    # Every call must reference the supplied library path.
    for call in call_log:
        assert call["library_path"] == str(lib_path)
        assert call["k"] == 5
    # The aggregate must reflect that at least some tasks were seeded.
    assert result["seeded_task_count"] >= 1, (
        "library has entries but Condition B never used a seed — "
        "the read path is dead"
    )


# ---------------------------------------------------------------------------
# Delta calculation
# ---------------------------------------------------------------------------


def test_classify_verdict_compounds():
    mod = _import_module()
    cond_a = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 1.0,
              "seeded_task_count": 0}
    cond_b = {"solved_count": 7, "mean_wall_clock_per_seeded_task_s": 0.4,
              "seeded_task_count": 7}
    assert mod.classify_verdict(cond_a, cond_b) == "compounds"


def test_classify_verdict_hurts():
    mod = _import_module()
    cond_a = {"solved_count": 7, "mean_wall_clock_per_seeded_task_s": 1.0,
              "seeded_task_count": 0}
    cond_b = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 0.4,
              "seeded_task_count": 5}
    assert mod.classify_verdict(cond_a, cond_b) == "hurts"


def test_classify_verdict_theater():
    mod = _import_module()
    cond_a = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 1.0,
              "seeded_task_count": 0}
    cond_b = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 0.99,
              "seeded_task_count": 5}
    assert mod.classify_verdict(cond_a, cond_b) == "theater"


def test_classify_verdict_no_seeds():
    mod = _import_module()
    cond_a = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 1.0,
              "seeded_task_count": 0}
    cond_b = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 1.0,
              "seeded_task_count": 0}
    assert mod.classify_verdict(cond_a, cond_b) == "no_seeds"


def test_classify_verdict_speedup_with_same_solves():
    """Solves equal but Condition B is materially faster on seeded tasks."""
    mod = _import_module()
    cond_a = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 2.0,
              "seeded_task_count": 0}
    cond_b = {"solved_count": 5, "mean_wall_clock_per_seeded_task_s": 0.5,
              "seeded_task_count": 5}
    assert mod.classify_verdict(cond_a, cond_b) == "compounds"


def test_run_ablation_computes_correct_delta(tmp_path, synthetic_dataset):
    """End-to-end: solve_delta and wall_clock_delta must match by-hand math."""
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    receipt_path = tmp_path / "receipt.json"
    markdown_path = tmp_path / "ABLATION.md"

    payload = mod.run_ablation(
        challenges=challenges,
        solutions=solutions,
        accumulation_size=8,
        holdout_size=2,
        budget_per_task=2.0,
        max_depth=1,
        beam_width=4,
        resonance_k=5,
        refine_max_iters=1,
        library_path=tmp_path / "lib.jsonl",
        receipt_path=receipt_path,
        markdown_path=markdown_path,
        verbose=False,
    )

    expected_delta = (
        payload["condition_b"]["solved_count"]
        - payload["condition_a"]["solved_count"]
    )
    assert payload["solve_delta"] == expected_delta
    expected_wall = round(
        payload["condition_a"]["mean_wall_clock_s"]
        - payload["condition_b"]["mean_wall_clock_s"], 4
    )
    assert payload["wall_clock_delta_s"] == expected_wall


# ---------------------------------------------------------------------------
# Receipt completeness
# ---------------------------------------------------------------------------


REQUIRED_RECEIPT_FIELDS = [
    "team",
    "kind",
    "recorded_at_utc",
    "paper_section",
    "tier_1_attestation_clean",
    "constraints",
    "accumulation_size",
    "holdout_size",
    "split_sha256",
    "accumulation_ids",
    "holdout_ids",
    "library_path",
    "synthesis_config",
    "accumulation",
    "condition_a",
    "condition_b",
    "solve_delta",
    "wall_clock_delta_s",
    "verdict",
]


def test_receipt_has_all_expected_fields(tmp_path, synthetic_dataset):
    """Brief: 'Receipt has all expected fields.'"""
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    receipt_path = tmp_path / "receipt.json"
    markdown_path = tmp_path / "ABLATION.md"

    payload = mod.run_ablation(
        challenges=challenges,
        solutions=solutions,
        accumulation_size=8,
        holdout_size=2,
        budget_per_task=2.0,
        max_depth=1,
        beam_width=4,
        resonance_k=5,
        refine_max_iters=1,
        library_path=tmp_path / "lib.jsonl",
        receipt_path=receipt_path,
        markdown_path=markdown_path,
    )

    # Top-level required fields.
    for field in REQUIRED_RECEIPT_FIELDS:
        assert field in payload, f"missing field: {field}"

    # Tier-1 constraints are explicit and all four are False.
    assert payload["constraints"]["llm_in_inference_path"] is False
    assert payload["constraints"]["pretrained_weights"] is False
    assert payload["constraints"]["internet_at_eval"] is False
    assert payload["constraints"]["public_corpus_heuristics"] is False

    # Synthesis config records every knob the brief named.
    cfg = payload["synthesis_config"]
    for k in ("budget_per_task_s", "max_depth", "beam_width",
              "resonance_k", "refine_max_iters"):
        assert k in cfg

    # Accumulation aggregates.
    acc = payload["accumulation"]
    for k in ("tasks_attempted", "tasks_solved",
              "entries_written", "atomic_entries_written"):
        assert k in acc

    # Each condition has the brief's required aggregates.
    for cond_key in ("condition_a", "condition_b"):
        cond = payload[cond_key]
        for k in ("solved_count", "mean_wall_clock_s",
                  "mean_wall_clock_per_seeded_task_s",
                  "false_rhyme_failures", "per_task",
                  "seeded_task_count"):
            assert k in cond, f"{cond_key} missing field: {k}"
        assert isinstance(cond["per_task"], list)

    # Verdict is one of the documented classifications.
    assert payload["verdict"] in (
        "compounds", "hurts", "theater", "no_seeds",
    )

    # Receipt is also valid JSON on disk.
    on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert on_disk["team"] == "ABLATION-RESONANCE"
    assert on_disk["solve_delta"] == payload["solve_delta"]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_output_is_well_formed(tmp_path, synthetic_dataset):
    """Brief: 'Markdown output is well-formed.'"""
    mod = _import_module()
    challenges, solutions = synthetic_dataset
    receipt_path = tmp_path / "receipt.json"
    markdown_path = tmp_path / "ABLATION.md"

    mod.run_ablation(
        challenges=challenges,
        solutions=solutions,
        accumulation_size=8,
        holdout_size=2,
        budget_per_task=2.0,
        max_depth=1,
        beam_width=4,
        resonance_k=5,
        refine_max_iters=1,
        library_path=tmp_path / "lib.jsonl",
        receipt_path=receipt_path,
        markdown_path=markdown_path,
    )

    assert markdown_path.exists()
    text = markdown_path.read_text(encoding="utf-8")

    # Section headers the report contract names.
    assert "# Resonance ablation" in text
    assert "## Split" in text
    assert "## Accumulation phase" in text
    assert "## Held-out results" in text
    assert "## Verdict" in text
    assert "## Honest constraints" in text

    # Table row markers for the main result table.
    assert "| Solved count |" in text
    assert "| Mean wall clock per task (s) |" in text
    assert "| Tasks seeded from library |" in text
    assert "| Mean wall clock per seeded task (s) |" in text
    assert "| False-rhyme failures" in text

    # Honesty footer.
    assert "No LLM" in text
    assert "self-solved" in text
    assert "Tier" in text or "tier" in text

    # The verdict line is present and quotes one of the documented values.
    assert "classification:" in text
    assert any(v in text for v in
               ("compounds", "hurts", "theater", "no_seeds"))


def test_render_markdown_handles_no_seeds_branch():
    """The verdict='no_seeds' message must render without crashing."""
    mod = _import_module()
    payload = {
        "accumulation_size": 8,
        "holdout_size": 2,
        "split_sha256": "deadbeef",
        "accumulation": {
            "tasks_attempted": 8, "tasks_solved": 0,
            "entries_written": 0, "atomic_entries_written": 0,
        },
        "condition_a": {
            "solved_count": 0, "mean_wall_clock_s": 1.0,
            "seeded_task_count": 0,
            "mean_wall_clock_per_seeded_task_s": 0.0,
            "false_rhyme_failures": 0,
        },
        "condition_b": {
            "solved_count": 0, "mean_wall_clock_s": 1.0,
            "seeded_task_count": 0,
            "mean_wall_clock_per_seeded_task_s": 0.0,
            "false_rhyme_failures": 0,
        },
        "solve_delta": 0,
        "wall_clock_delta_s": 0.0,
        "verdict": "no_seeds",
    }
    md = mod.render_markdown(payload)
    assert "no_seeds" in md
    assert "## Verdict" in md
    # Specific guidance message for the no-seeds verdict.
    assert "no usable seeds" in md.lower()
