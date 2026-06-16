"""Tests for the Federation Founding Adjudication script.

The script under test is `scripts/run_founding_adjudication.py`. Per Charter
Article IV §4.2 it produces the Federation's first public determination on
Misfit-Alpha; per Mom's Law every claim it makes carries a receipt.

This test battery enforces five contracts:

  1. Importing the script module MUST NOT touch disk.
  2. `audit_misfit_alpha(repo_root)` MUST NOT touch disk; writes happen only
     via `emit_artifacts(...)` invoked from main(--execute).
  3. Self-audit of the live repo returns RECOGNIZED_TIER_1.
  4. A temp directory with a planted forbidden import on the inference path
     returns DISQUALIFIED (mechanical Tier-1 attestation must fail).
  5. The emitted receipt JSON carries the git commit SHA, the audit
     timestamp, the determination, and the per-PEM-field coverage; the
     emitted markdown is well-formed (has the required H1, the
     determination, the criteria table, and the receipt anchor section).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_founding_adjudication.py"


# ---------------------------------------------------------------------------
# Module loader — imports run_founding_adjudication by path without requiring
# the scripts/ directory on PYTHONPATH.
# ---------------------------------------------------------------------------

def _load_adjudication_module():
    """Load scripts/run_founding_adjudication.py as a module under a stable
    name. Used by every test below. We snapshot sys.modules so an import
    failure does not pollute the test session."""
    spec = importlib.util.spec_from_file_location(
        "run_founding_adjudication_under_test", str(SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_founding_adjudication_under_test"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Contract 1 — import is side-effect free.
# ---------------------------------------------------------------------------

def test_script_imports_without_side_effects(tmp_path, monkeypatch):
    """Importing the module must not write any files anywhere. We snapshot
    the default emission paths under the LIVE repo and confirm they are
    unchanged after import."""
    md_default = REPO_ROOT / "docs" / "FOUNDING_ADJUDICATION_v1.md"
    rj_default = REPO_ROOT / "receipts" / "100day" / "founding_adjudication.json"
    md_before_exists = md_default.exists()
    rj_before_exists = rj_default.exists()
    md_before_stat = md_default.stat().st_mtime_ns if md_before_exists else None
    rj_before_stat = rj_default.stat().st_mtime_ns if rj_before_exists else None

    mod = _load_adjudication_module()

    # Module loaded; verify the surface we expect is exposed.
    for name in (
        "audit_misfit_alpha",
        "emit_artifacts",
        "render_markdown",
        "AuditResult",
        "CheckResult",
        "DETERMINATION_RECOGNIZED",
        "DETERMINATION_DISPUTED",
        "DETERMINATION_DISQUALIFIED",
        "PEM_FIELDS",
        "main",
    ):
        assert hasattr(mod, name), f"module missing public symbol: {name}"

    # No file was created or mutated by the import.
    if md_before_exists:
        assert md_default.stat().st_mtime_ns == md_before_stat, (
            "import mutated docs/FOUNDING_ADJUDICATION_v1.md"
        )
    else:
        assert not md_default.exists(), (
            "import created docs/FOUNDING_ADJUDICATION_v1.md"
        )
    if rj_before_exists:
        assert rj_default.stat().st_mtime_ns == rj_before_stat, (
            "import mutated receipts/100day/founding_adjudication.json"
        )
    else:
        assert not rj_default.exists(), (
            "import created receipts/100day/founding_adjudication.json"
        )


# ---------------------------------------------------------------------------
# Contract 2 + 3 — self-audit returns RECOGNIZED_TIER_1, no writes.
# ---------------------------------------------------------------------------

def test_self_audit_returns_recognized_tier_1():
    """Running audit_misfit_alpha() against the live repo returns
    RECOGNIZED_TIER_1 — the determination the Federation expects to publish
    on the founding day."""
    mod = _load_adjudication_module()
    md_default = REPO_ROOT / "docs" / "FOUNDING_ADJUDICATION_v1.md"
    rj_default = REPO_ROOT / "receipts" / "100day" / "founding_adjudication.json"
    md_before = (md_default.exists(),
                 md_default.stat().st_mtime_ns if md_default.exists() else None)
    rj_before = (rj_default.exists(),
                 rj_default.stat().st_mtime_ns if rj_default.exists() else None)

    result = mod.audit_misfit_alpha(REPO_ROOT)

    assert result.determination == mod.DETERMINATION_RECOGNIZED, (
        f"expected RECOGNIZED_TIER_1, got {result.determination}; "
        f"failing checks: "
        f"{[c.name for c in result.checks if not c.passed]}"
    )
    # Every check passed.
    failing = [c for c in result.checks if not c.passed]
    assert not failing, (
        "self-audit should pass every check on the founding day; "
        f"failing: {[(c.name, c.summary) for c in failing]}"
    )

    # 6 checks total per Charter Article IV §4.2 spec.
    assert len(result.checks) == 6, (
        f"expected 6 checks (charter + tier1_att + tier1_adv + pem + "
        f"designer_choice + provenance), got {len(result.checks)}"
    )

    # Sovereign + federation metadata are present.
    assert result.sovereign == "Atom McCree"
    assert result.federation == "AtomEons Federation"
    assert result.public_id == "misfit-alpha@atomeons/1.0"
    assert result.charter_version == "1.0"

    # All 8 PEM fields declared, zero missing.
    assert len(result.pem_fields_present) == 8, (
        f"PEM coverage: {result.pem_fields_present}"
    )
    assert result.pem_fields_missing == []

    # Designer-choice annotations exist in config.py.
    assert result.designer_choice_count >= 5, (
        f"expected >=5 (c) DESIGNER CHOICE annotations in config.py, "
        f"got {result.designer_choice_count}"
    )
    assert result.designer_choice_disclosure_ok

    # Git SHA captured (this repo is a git checkout in CI).
    assert result.git_commit_sha and result.git_commit_sha != "UNKNOWN"
    assert len(result.git_commit_sha) >= 7

    # Side-effect contract honoured: no file written by audit_misfit_alpha.
    md_after = (md_default.exists(),
                md_default.stat().st_mtime_ns if md_default.exists() else None)
    rj_after = (rj_default.exists(),
                rj_default.stat().st_mtime_ns if rj_default.exists() else None)
    assert md_before == md_after, (
        "audit_misfit_alpha mutated docs/FOUNDING_ADJUDICATION_v1.md"
    )
    assert rj_before == rj_after, (
        "audit_misfit_alpha mutated receipts/100day/founding_adjudication.json"
    )


# ---------------------------------------------------------------------------
# Contract 4 — planted forbidden import in temp repo -> DISQUALIFIED.
# ---------------------------------------------------------------------------

def _plant_minimal_repo(dest: Path) -> None:
    """Stage a minimal repo skeleton under `dest` that the founding
    adjudication can run against. We copy the live `src/`, `tests/`,
    `docs/CHARTER_v1.md`, `docs/TIER_1_DISCLOSURE.md`, and `pyproject.toml`
    so the Tier-1 test suites can execute in the temp tree. The whole tree
    becomes a fresh repo root so the audit's repo-rooted file scans are
    contained inside `dest`."""
    src_dir = REPO_ROOT / "src"
    tests_dir = REPO_ROOT / "tests"
    docs_dir = REPO_ROOT / "docs"
    shutil.copytree(src_dir, dest / "src", dirs_exist_ok=False)
    # Copy ONLY the two Tier-1 attestation test files plus the conftest if
    # one exists. We do NOT copy the entire test suite because it depends on
    # fixtures we don't need.
    (dest / "tests").mkdir(parents=True, exist_ok=True)
    for name in ("test_tier1_attestation.py", "test_tier1_adversarial.py"):
        shutil.copy2(tests_dir / name, dest / "tests" / name)
    # Drop an empty conftest so pytest treats the dest as a project root.
    (dest / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (dest / "conftest.py").write_text(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent / 'src'))\n",
        encoding="utf-8",
    )
    (dest / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(docs_dir / "CHARTER_v1.md", dest / "docs" / "CHARTER_v1.md")
    shutil.copy2(docs_dir / "TIER_1_DISCLOSURE.md",
                 dest / "docs" / "TIER_1_DISCLOSURE.md")
    # Minimal pyproject — needed by test_pyproject_has_no_llm_dependencies.
    shutil.copy2(REPO_ROOT / "pyproject.toml", dest / "pyproject.toml")


def test_self_audit_disqualifies_planted_forbidden_import(tmp_path):
    """Plant a forbidden `import openai` on the inference path of a temp
    repo. The mechanical Tier-1 attestation MUST fail and the determination
    MUST be DISQUALIFIED.

    This is the central honesty proof: the adjudication ladder is not
    cosmetic. A real violation of Article III §3.6 produces a real DSQ."""
    mod = _load_adjudication_module()

    repo = tmp_path / "planted_repo"
    repo.mkdir()
    _plant_minimal_repo(repo)

    # Plant a forbidden import on the inference path.
    planted_file = repo / "src" / "misfit_agent" / "_planted_violation.py"
    planted_file.write_text(
        "# Adversarial test: this file simulates a Tier-1 violation.\n"
        "import openai  # FORBIDDEN per Article III §3.6\n",
        encoding="utf-8",
    )

    result = mod.audit_misfit_alpha(repo)

    assert result.determination == mod.DETERMINATION_DISQUALIFIED, (
        f"expected DISQUALIFIED with planted forbidden import, got "
        f"{result.determination}; failing checks: "
        f"{[c.name for c in result.checks if not c.passed]}"
    )

    # The attestation check must specifically have failed (mechanical).
    attest = next(c for c in result.checks if c.name == "tier1_attestation")
    assert not attest.passed, (
        "test_tier1_attestation.py should fail with a planted forbidden import"
    )
    assert attest.blocker_severity == "mechanical"

    # The planted SHA-of-violation message should appear in the captured
    # pytest output tail so an auditor can trace the cause.
    output_tail = attest.details.get("output_tail", "")
    assert "openai" in output_tail or "forbidden" in output_tail.lower(), (
        f"expected forbidden-import evidence in pytest output, got: "
        f"{output_tail[:400]}"
    )


# ---------------------------------------------------------------------------
# Contract 5 — receipt JSON shape + markdown well-formed.
# ---------------------------------------------------------------------------

def test_emit_artifacts_writes_well_formed_receipt(tmp_path):
    """emit_artifacts writes the receipt JSON with the full chain of evidence
    AND the markdown report with the documented section structure."""
    mod = _load_adjudication_module()
    result = mod.audit_misfit_alpha(REPO_ROOT)

    md_path = tmp_path / "FOUNDING_ADJUDICATION_v1.md"
    rj_path = tmp_path / "founding_adjudication.json"
    written_md, written_rj = mod.emit_artifacts(
        result, repo_root=REPO_ROOT,
        markdown_path=md_path, receipt_path=rj_path,
    )
    assert written_md == md_path
    assert written_rj == rj_path
    assert md_path.exists()
    assert rj_path.exists()

    # Receipt JSON parses, has the required top-level fields.
    receipt = json.loads(rj_path.read_text(encoding="utf-8"))
    required_top = {
        "determination",
        "federation",
        "charter_version",
        "founding_date",
        "sovereign",
        "public_id",
        "git_commit_sha",
        "audit_timestamp_unix",
        "audit_timestamp_iso",
        "article_ii_criteria",
        "pem_fields_present",
        "pem_fields_missing",
        "designer_choice_count",
        "designer_choice_disclosure_ok",
        "checks",
    }
    assert required_top.issubset(receipt.keys()), (
        f"missing top-level fields: {required_top - receipt.keys()}"
    )

    # Determination, sovereign, charter version are honest.
    assert receipt["determination"] == mod.DETERMINATION_RECOGNIZED
    assert receipt["sovereign"] == "Atom McCree"
    assert receipt["charter_version"] == "1.0"
    assert receipt["public_id"] == "misfit-alpha@atomeons/1.0"

    # SHA + timestamp captured.
    assert receipt["git_commit_sha"] != "UNKNOWN"
    assert len(receipt["git_commit_sha"]) >= 7
    assert isinstance(receipt["audit_timestamp_unix"], (int, float))
    assert receipt["audit_timestamp_unix"] > 0
    assert receipt["audit_timestamp_iso"].endswith("Z")

    # All 8 PEM fields recorded in receipt.
    pem_names_in_receipt = set(receipt["pem_fields_present"])
    expected_pem = {canonical for canonical, _ in mod.PEM_FIELDS}
    assert pem_names_in_receipt == expected_pem, (
        f"PEM coverage mismatch: missing={expected_pem - pem_names_in_receipt}, "
        f"extra={pem_names_in_receipt - expected_pem}"
    )
    assert receipt["pem_fields_missing"] == []

    # All 4 Article II criteria recorded.
    criteria = receipt["article_ii_criteria"]
    assert len(criteria) == 4
    expected_criteria = {
        "continuity_of_state",
        "provenance_enforcement",
        "self_identification",
        "membership_signature",
    }
    assert {c["criterion"] for c in criteria} == expected_criteria
    assert all(c["met"] for c in criteria)

    # Checks: 6 of them, all green.
    assert len(receipt["checks"]) == 6
    assert all(c["passed"] for c in receipt["checks"])
    check_names = {c["name"] for c in receipt["checks"]}
    assert check_names == {
        "charter_binding",
        "tier1_attestation",
        "tier1_adversarial",
        "pem_contract",
        "designer_choice_disclosure",
        "provenance_anchor",
    }

    # Markdown well-formed.
    md_text = md_path.read_text(encoding="utf-8")
    assert md_text.startswith(
        "# Founding Adjudication of the AtomEons Federation"
    ), "markdown must start with the documented H1"
    assert "**Version 1**" in md_text
    assert "CC-BY-4.0" in md_text
    assert f"**{mod.DETERMINATION_RECOGNIZED}**" in md_text
    assert "## Determination" in md_text
    assert "## Article II §2.7 Recognition Criteria" in md_text
    assert "## Audit Checks" in md_text
    assert "## Provenance-Enforced Memory (PEM) Contract" in md_text
    assert "## Designer-Choice Transparency" in md_text
    assert "## Receipt Anchor" in md_text
    assert receipt["git_commit_sha"] in md_text
    assert "Atom McCree" in md_text
    assert receipt["audit_timestamp_iso"] in md_text


def test_markdown_render_handles_disqualified_path(tmp_path):
    """render_markdown produces a coherent DISQUALIFIED report even when
    every check failed — we synthesize a failed AuditResult to exercise the
    rendering branch without polluting the live repo."""
    mod = _load_adjudication_module()
    failed_checks = [
        mod.CheckResult(
            name="tier1_attestation",
            passed=False,
            summary="forbidden import found",
            details={"tests_passed": 0},
            blocker_severity="mechanical",
        ),
    ]
    result = mod.AuditResult(
        determination=mod.DETERMINATION_DISQUALIFIED,
        checks=failed_checks,
        article_ii_criteria=[
            {"criterion": "provenance_enforcement",
             "met": False,
             "evidence": "Tier-1 attestation failed"},
        ],
        pem_fields_present=[],
        pem_fields_missing=[name for name, _ in mod.PEM_FIELDS],
        designer_choice_count=0,
        designer_choice_disclosure_ok=False,
        git_commit_sha="0" * 40,
        audit_timestamp_unix=1_700_000_000.0,
        audit_timestamp_iso="2023-11-14T22:13:20Z",
        repo_root=str(tmp_path),
    )
    md = mod.render_markdown(result)
    assert "**DISQUALIFIED**" in md
    assert "cannot be recognized" in md
    assert "tier1_attestation" in md
    assert "FAIL" in md


def test_provenance_anchor_check_independently(tmp_path):
    """_check_provenance_anchor returns the SHA + sovereign attestation
    even outside the orchestrator. Used by the receipt JSON's chain-of-
    evidence section."""
    mod = _load_adjudication_module()
    check, sha = mod._check_provenance_anchor(REPO_ROOT)
    assert check.name == "provenance_anchor"
    assert check.passed
    assert sha != "UNKNOWN"
    assert len(sha) >= 7
    assert check.details["git_commit_sha"] == sha
    assert check.details["sovereign"] == "Atom McCree"
    assert check.details["sovereign_signature_present"]


def test_pem_contract_check_finds_all_eight_fields():
    """_check_pem_contract enumerates the 8 PEM fields and reports zero
    missing for the live resonance.py."""
    mod = _load_adjudication_module()
    result, present, missing = mod._check_pem_contract(REPO_ROOT)
    assert result.passed
    assert missing == []
    assert len(present) == 8
    expected = {name for name, _ in mod.PEM_FIELDS}
    assert set(present) == expected
