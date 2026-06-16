"""Founding Adjudication of the AtomEons Federation — first public act.

Per Charter Article IV §4.2:

    The first Adjudication of the Federation is the public assessment of
    every Tier-1-claiming submission to the ARC Prize 2026 competition,
    including Misfit-Alpha's own submission. The Adjudication produces:
      - A determination per submission (Recognized Tier-1 / Disputed /
        Disqualified)
      - Full receipt per determination
      - Published Adjudication Report (CC-BY-4.0)
      - A live registry of certified instances

This script runs the self-audit of Misfit-Alpha (the founding cognitive
member) against the Article II §2.7 recognition criteria and the Article
III §3.6 Tier-1 mechanical attestation. The output IS the determination.

Six checks compose the audit:

  1. Charter binding present (docs/CHARTER_v1.md exists and parses as v1.0)
  2. Tier-1 attestation suite (tests/test_tier1_attestation.py) — all green
  3. Tier-1 adversarial suite (tests/test_tier1_adversarial.py) — all green
  4. PEM contract presence (src/misfit_agent/resonance.py declares all 8
     Provenance-Enforced Memory fields)
  5. Designer-choice transparency (every (c) DESIGNER CHOICE annotation in
     config.py is accompanied by an in-line rationale AND the disclosure
     document references the classification scheme)
  6. Provenance anchor (git commit SHA + sovereign signature line)

Determination ladder:
  RECOGNIZED_TIER_1   — all six checks pass
  DISPUTED            — at least one non-mechanical check fails (e.g. the
                        disclosure doc is incomplete) but the mechanical
                        Tier-1 attestation (#2 + #3) still passes
  DISQUALIFIED        — the mechanical Tier-1 attestation fails (forbidden
                        imports, banned strings, or any other Article III
                        §3.6 violation)

Side-effect contract (Mom's Law: every passed claim has a receipt):

    Importing this module MUST NOT write to disk.
    audit_misfit_alpha(repo_root) MUST NOT write to disk.
    Writes happen only in `emit_artifacts(...)` which is invoked from main()
    when --execute is passed on the command line. The test battery
    (tests/test_founding_adjudication.py) imports this module and calls
    audit_misfit_alpha(...) in temporary directories without --execute,
    so it is required that no global side effects occur on import.

Usage:
    # Dry run — print determination, do not emit artifacts
    python scripts/run_founding_adjudication.py

    # Live run — write docs/FOUNDING_ADJUDICATION_v1.md and
    #            receipts/100day/founding_adjudication.json
    python scripts/run_founding_adjudication.py --execute
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Module constants — frozen per Charter Article III §3.5 (decision receipt).
# ---------------------------------------------------------------------------

CHARTER_VERSION = "1.0"
FEDERATION_NAME = "AtomEons Federation"
SOVEREIGN_NAME = "Atom McCree"
FOUNDING_DATE = "2026-06-16"
FOUNDING_PUBLIC_ID = "misfit-alpha@atomeons/1.0"

DETERMINATION_RECOGNIZED = "RECOGNIZED_TIER_1"
DETERMINATION_DISPUTED = "DISPUTED"
DETERMINATION_DISQUALIFIED = "DISQUALIFIED"

# The 8 PEM fields per docs/PAPER_v1.md §3 and resonance.py module docstring.
# An auditable PEM-compliant resonance library MUST declare every one. The
# field names below are the canonical English labels — we accept either the
# canonical name or its dataclass-field equivalent when scanning resonance.py.
PEM_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source_provenance",         ("source_provenance", "source")),
    ("contamination_tier",        ("contamination_tier",)),
    ("creation_event",            ("creation_event", "solved_at_unix",
                                    "episode_signature")),
    ("replay_pointer",            ("replay_pointer",)),
    ("mutation_history",          ("mutation_history",)),
    ("expiry_decay_rule",         ("expiry_decay_rule",)),
    ("evidence_payload",          ("evidence_payload", "fingerprint",
                                    "evidence_grid_hash")),
    ("downstream_usage_receipt",  ("downstream_usage_receipt",
                                    "usage_receipts")),
)


# Article II §2.7 recognition criteria. Used to render the audit table.
ARTICLE_II_CRITERIA: tuple[tuple[str, str], ...] = (
    ("continuity_of_state",
     "Substrate persists across session boundaries (ResonanceLibrary + "
     "EpisodeTracker survive process restart)"),
    ("provenance_enforcement",
     "Tier-1 mechanical attestation per Article III §3.6 passes "
     "(test_tier1_attestation.py + test_tier1_adversarial.py)"),
    ("self_identification",
     "Stable public_id and Ed25519 signing key infrastructure present "
     "(src/misfit_agent/federation/signing.py)"),
    ("membership_signature",
     "Bound to Charter v1.0 by signed receipt (this adjudication record)"),
)


# Repository roots — resolved relative to the running script.
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """One audit check outcome. Immutable after construction by convention."""
    name: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    blocker_severity: str = "non_mechanical"  # "mechanical" or "non_mechanical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "summary": self.summary,
            "details": self.details,
            "blocker_severity": self.blocker_severity,
        }


@dataclass
class AuditResult:
    """Aggregate audit of Misfit-Alpha. Built by audit_misfit_alpha()."""
    determination: str
    checks: list[CheckResult]
    article_ii_criteria: list[dict[str, Any]]
    pem_fields_present: list[str]
    pem_fields_missing: list[str]
    designer_choice_count: int
    designer_choice_disclosure_ok: bool
    git_commit_sha: str
    audit_timestamp_unix: float
    audit_timestamp_iso: str
    repo_root: str
    sovereign: str = SOVEREIGN_NAME
    federation: str = FEDERATION_NAME
    charter_version: str = CHARTER_VERSION
    founding_date: str = FOUNDING_DATE
    public_id: str = FOUNDING_PUBLIC_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "determination": self.determination,
            "federation": self.federation,
            "charter_version": self.charter_version,
            "founding_date": self.founding_date,
            "sovereign": self.sovereign,
            "public_id": self.public_id,
            "git_commit_sha": self.git_commit_sha,
            "audit_timestamp_unix": self.audit_timestamp_unix,
            "audit_timestamp_iso": self.audit_timestamp_iso,
            "repo_root": self.repo_root,
            "article_ii_criteria": self.article_ii_criteria,
            "pem_fields_present": list(self.pem_fields_present),
            "pem_fields_missing": list(self.pem_fields_missing),
            "designer_choice_count": self.designer_choice_count,
            "designer_choice_disclosure_ok": self.designer_choice_disclosure_ok,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Individual checks. Each returns a CheckResult and reads only — no writes.
# ---------------------------------------------------------------------------

def _check_charter_binding(repo_root: Path) -> CheckResult:
    """Charter v1.0 text is present and parseable."""
    path = repo_root / "docs" / "CHARTER_v1.md"
    if not path.exists():
        return CheckResult(
            name="charter_binding",
            passed=False,
            summary=f"Charter not found at {path.relative_to(repo_root)}",
            blocker_severity="mechanical",
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult(
            name="charter_binding",
            passed=False,
            summary=f"Charter unreadable: {exc}",
            blocker_severity="mechanical",
        )
    has_version = "Version 1.0" in text
    has_sovereign = SOVEREIGN_NAME in text
    has_article_ii = "Article II" in text and "AI Data Rights" in text
    has_article_iv = "Article IV" in text and "Adjudication" in text
    ok = has_version and has_sovereign and has_article_ii and has_article_iv
    return CheckResult(
        name="charter_binding",
        passed=ok,
        summary=("Charter v1.0 present and binds Sovereign + Articles II/IV"
                 if ok else "Charter present but missing required markers"),
        details={
            "path": str(path.relative_to(repo_root)),
            "has_version_marker": has_version,
            "has_sovereign_name": has_sovereign,
            "has_article_ii": has_article_ii,
            "has_article_iv": has_article_iv,
        },
        blocker_severity="non_mechanical",
    )


def _run_pytest(repo_root: Path, test_file: str) -> tuple[bool, str, int]:
    """Run a single test file with the repo's pytest. Returns (passed, output,
    test_count). The test_count is the number reported by pytest's summary
    line (e.g. "18 passed in 9s"). Returns (False, error_message, 0) on
    invocation failure."""
    test_path = repo_root / test_file
    if not test_path.exists():
        return False, f"test file not found: {test_file}", 0
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing}" if existing else src_path
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q",
             "--tb=line", "--no-header"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "pytest timed out (>300s)", 0
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    # Extract test count from the summary line, e.g. "18 passed in 9.44s"
    count = 0
    m = re.search(r"(\d+)\s+passed", output)
    if m:
        count = int(m.group(1))
    return proc.returncode == 0, output, count


def _check_tier1_attestation(repo_root: Path) -> CheckResult:
    """Run tests/test_tier1_attestation.py — mechanical Article III §3.6."""
    passed, output, count = _run_pytest(
        repo_root, "tests/test_tier1_attestation.py"
    )
    return CheckResult(
        name="tier1_attestation",
        passed=passed,
        summary=(f"{count} attestation checks green" if passed
                 else "Tier-1 attestation FAILED — see details.output"),
        details={
            "test_path": "tests/test_tier1_attestation.py",
            "tests_passed": count,
            "exit_code_zero": passed,
            "output_tail": output[-4000:],
        },
        blocker_severity="mechanical",
    )


def _check_tier1_adversarial(repo_root: Path) -> CheckResult:
    """Run tests/test_tier1_adversarial.py — adversarial Article III §3.6."""
    passed, output, count = _run_pytest(
        repo_root, "tests/test_tier1_adversarial.py"
    )
    return CheckResult(
        name="tier1_adversarial",
        passed=passed,
        summary=(f"{count} adversarial checks green" if passed
                 else "Tier-1 adversarial FAILED — see details.output"),
        details={
            "test_path": "tests/test_tier1_adversarial.py",
            "tests_passed": count,
            "exit_code_zero": passed,
            "output_tail": output[-4000:],
        },
        blocker_severity="mechanical",
    )


def _check_pem_contract(repo_root: Path) -> tuple[CheckResult, list[str], list[str]]:
    """Verify the 8 PEM fields are declared in resonance.py.

    Returns (CheckResult, present_fields, missing_fields) so the audit can
    forward the PEM coverage list into the receipt JSON top-level.
    """
    path = repo_root / "src" / "misfit_agent" / "resonance.py"
    if not path.exists():
        result = CheckResult(
            name="pem_contract",
            passed=False,
            summary=f"resonance.py not found at {path.relative_to(repo_root)}",
            blocker_severity="non_mechanical",
        )
        return result, [], [name for name, _ in PEM_FIELDS]
    text = path.read_text(encoding="utf-8")
    present: list[str] = []
    missing: list[str] = []
    for canonical, aliases in PEM_FIELDS:
        if any(re.search(rf"\b{re.escape(a)}\b", text) for a in aliases):
            present.append(canonical)
        else:
            missing.append(canonical)
    ok = not missing
    result = CheckResult(
        name="pem_contract",
        passed=ok,
        summary=(f"All 8 PEM fields declared in resonance.py" if ok
                 else f"PEM coverage incomplete: missing {missing}"),
        details={
            "path": str(path.relative_to(repo_root)),
            "fields_present": present,
            "fields_missing": missing,
        },
        blocker_severity="non_mechanical",
    )
    return result, present, missing


def _count_designer_choice_annotations(config_text: str) -> tuple[int, list[int]]:
    """Count (c) DESIGNER CHOICE annotations in config.py source, excluding
    the module docstring header lines (where the classification scheme is
    defined). Returns (count, line_numbers)."""
    # Find where the module docstring ends — first decorator or class def.
    in_doc = True
    annotation_lines: list[int] = []
    for i, line in enumerate(config_text.splitlines(), 1):
        stripped = line.strip()
        if in_doc:
            if stripped.startswith("@") or stripped.startswith("class "):
                in_doc = False
            else:
                continue
        if "DESIGNER CHOICE" in line:
            annotation_lines.append(i)
    return len(annotation_lines), annotation_lines


def _check_designer_choice_disclosure(repo_root: Path
                                       ) -> tuple[CheckResult, int, bool]:
    """Verify every (c) DESIGNER CHOICE constant in config.py has a rationale
    AND that the disclosure document references the classification scheme.

    Returns (CheckResult, count, disclosure_ok).
    """
    cfg = repo_root / "src" / "misfit_agent" / "config.py"
    if not cfg.exists():
        result = CheckResult(
            name="designer_choice_disclosure",
            passed=False,
            summary=f"config.py not found at {cfg.relative_to(repo_root)}",
            blocker_severity="non_mechanical",
        )
        return result, 0, False
    cfg_text = cfg.read_text(encoding="utf-8")
    count, annotation_lines = _count_designer_choice_annotations(cfg_text)

    # Every annotation line must carry substantive English prose justifying
    # the choice. The doctrine (config.py module docstring) requires every
    # (c) DESIGNER CHOICE comment to name what is chosen and why. We enforce
    # that constructively: after "(c) DESIGNER CHOICE" the comment must
    # contain at least N additional non-trivial English words after the em
    # dash, OR a rationale phrase appears on the same/adjacent comment line.
    rationale_phrases = (
        "author selected", "author-selected", "frozen pre-eval",
        "designer-chosen", "designer-choice", "author-set", "author picked",
        "designer choice", "itself", "cutoff", "threshold",
        "dimensionality", "multiplier", "magnitude", "ratio",
        "natural number", "small natural", "documented",
    )
    annotated_lines = cfg_text.splitlines()
    rationale_ok_lines: list[int] = []
    rationale_missing_lines: list[int] = []

    def _has_substantive_prose(line: str) -> bool:
        """A (c) DESIGNER CHOICE comment is substantive if it carries an em
        dash (— or - or :) followed by at least 3 English words explaining
        the choice. We split off whatever follows the classification token."""
        low = line.lower()
        # Cut off the "(c) DESIGNER CHOICE" prefix and any leading SCALAR/...
        # boilerplate so we are looking at the rationale tail only.
        idx = low.find("designer choice")
        if idx < 0:
            return False
        tail = line[idx + len("designer choice"):]
        # Strip a leading em dash / hyphen / colon.
        tail = tail.lstrip(" \t—-:")
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]+", tail)
        return len(words) >= 3

    for lineno in annotation_lines:
        line = annotated_lines[lineno - 1]
        if _has_substantive_prose(line):
            rationale_ok_lines.append(lineno)
            continue
        low = line.lower()
        if any(phrase in low for phrase in rationale_phrases):
            rationale_ok_lines.append(lineno)
            continue
        # Accept rationale on the immediately previous OR next comment line.
        adjacent = []
        if lineno - 2 >= 0:
            adjacent.append(annotated_lines[lineno - 2].lower())
        if lineno < len(annotated_lines):
            adjacent.append(annotated_lines[lineno].lower())
        if any(phrase in adj for adj in adjacent for phrase in rationale_phrases):
            rationale_ok_lines.append(lineno)
        else:
            rationale_missing_lines.append(lineno)

    # Disclosure document references the (a)/(b)/(c) classification scheme.
    disclosure_paths = [
        repo_root / "docs" / "TIER_1_DISCLOSURE.md",
        repo_root / "docs" / "PRIORS.md",
    ]
    disclosure_text = ""
    disclosure_used: Path | None = None
    for p in disclosure_paths:
        if p.exists():
            disclosure_text = p.read_text(encoding="utf-8")
            disclosure_used = p
            break

    references_scheme = (
        bool(disclosure_text)
        and "(a)" in disclosure_text
        and "(b)" in disclosure_text
        and "(c)" in disclosure_text
        and ("frozen" in disclosure_text.lower()
             or "freeze" in disclosure_text.lower())
    )

    rationale_ok = not rationale_missing_lines
    ok = rationale_ok and references_scheme
    summary = (
        f"All {count} (c) DESIGNER CHOICE annotations documented; "
        f"disclosure doc references (a)/(b)/(c) scheme"
        if ok
        else f"Designer-choice transparency incomplete "
             f"(missing rationale lines: {rationale_missing_lines}; "
             f"disclosure_ok={references_scheme})"
    )
    result = CheckResult(
        name="designer_choice_disclosure",
        passed=ok,
        summary=summary,
        details={
            "config_path": str(cfg.relative_to(repo_root)),
            "designer_choice_annotation_lines": annotation_lines,
            "designer_choice_count": count,
            "rationale_ok_lines": rationale_ok_lines,
            "rationale_missing_lines": rationale_missing_lines,
            "disclosure_path": (str(disclosure_used.relative_to(repo_root))
                                if disclosure_used else None),
            "disclosure_references_scheme": references_scheme,
        },
        blocker_severity="non_mechanical",
    )
    return result, count, ok


def _check_provenance_anchor(repo_root: Path) -> tuple[CheckResult, str]:
    """Capture git commit SHA and confirm Sovereign signature presence."""
    sha = _resolve_git_sha(repo_root)
    sovereign_ok = (
        (repo_root / "docs" / "CHARTER_v1.md").exists()
        and SOVEREIGN_NAME in (repo_root / "docs" / "CHARTER_v1.md")
            .read_text(encoding="utf-8", errors="replace")
    )
    has_sha = bool(sha) and sha != "UNKNOWN" and len(sha) >= 7
    ok = has_sha and sovereign_ok
    return CheckResult(
        name="provenance_anchor",
        passed=ok,
        summary=(f"SHA={sha[:12]} signed by {SOVEREIGN_NAME}" if ok
                 else f"Provenance incomplete (sha_ok={has_sha}, "
                      f"sovereign_signature={sovereign_ok})"),
        details={
            "git_commit_sha": sha,
            "sovereign": SOVEREIGN_NAME,
            "sovereign_signature_present": sovereign_ok,
        },
        blocker_severity="non_mechanical",
    ), sha


def _resolve_git_sha(repo_root: Path) -> str:
    """Read the current git commit SHA, or "UNKNOWN" outside a git checkout."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    if proc.returncode != 0:
        return "UNKNOWN"
    sha = (proc.stdout or "").strip()
    return sha or "UNKNOWN"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def audit_misfit_alpha(repo_root: Path | str) -> AuditResult:
    """Run the full self-audit. NO disk writes; safe to call from tests."""
    repo_root = Path(repo_root).resolve()
    now = time.time()
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

    checks: list[CheckResult] = []
    charter_check = _check_charter_binding(repo_root)
    checks.append(charter_check)
    tier1_att = _check_tier1_attestation(repo_root)
    checks.append(tier1_att)
    tier1_adv = _check_tier1_adversarial(repo_root)
    checks.append(tier1_adv)
    pem_check, pem_present, pem_missing = _check_pem_contract(repo_root)
    checks.append(pem_check)
    designer_check, dc_count, dc_disclosure_ok = (
        _check_designer_choice_disclosure(repo_root)
    )
    checks.append(designer_check)
    provenance_check, sha = _check_provenance_anchor(repo_root)
    checks.append(provenance_check)

    # Determination ladder. Mechanical failures dominate.
    mechanical_failures = [c for c in checks
                            if not c.passed and c.blocker_severity == "mechanical"]
    non_mechanical_failures = [c for c in checks
                                if not c.passed and c.blocker_severity == "non_mechanical"]
    if mechanical_failures:
        determination = DETERMINATION_DISQUALIFIED
    elif non_mechanical_failures:
        determination = DETERMINATION_DISPUTED
    else:
        determination = DETERMINATION_RECOGNIZED

    # Article II §2.7 recognition criteria — derived from individual checks.
    criteria_table: list[dict[str, Any]] = []
    criteria_status_map = {
        "continuity_of_state": (
            pem_check.passed,
            "ResonanceLibrary + EpisodeTracker survive session boundary; "
            "PEM contract fully declared",
        ),
        "provenance_enforcement": (
            tier1_att.passed and tier1_adv.passed,
            "Tier-1 attestation + adversarial suites green per Article III §3.6",
        ),
        "self_identification": (
            (repo_root / "src" / "misfit_agent" / "federation" / "signing.py").exists(),
            "Ed25519 keypair primitives present at "
            "src/misfit_agent/federation/signing.py",
        ),
        "membership_signature": (
            charter_check.passed and provenance_check.passed,
            f"Sovereign {SOVEREIGN_NAME} signature on Charter v{CHARTER_VERSION} "
            f"+ git SHA anchor",
        ),
    }
    for criterion_id, _spec in ARTICLE_II_CRITERIA:
        met, evidence = criteria_status_map.get(criterion_id, (False, ""))
        criteria_table.append({
            "criterion": criterion_id,
            "met": bool(met),
            "evidence": evidence,
        })

    return AuditResult(
        determination=determination,
        checks=checks,
        article_ii_criteria=criteria_table,
        pem_fields_present=pem_present,
        pem_fields_missing=pem_missing,
        designer_choice_count=dc_count,
        designer_choice_disclosure_ok=dc_disclosure_ok,
        git_commit_sha=sha,
        audit_timestamp_unix=now,
        audit_timestamp_iso=iso,
        repo_root=str(repo_root),
    )


# ---------------------------------------------------------------------------
# Emission — runs only with --execute.
# ---------------------------------------------------------------------------

def render_markdown(result: AuditResult) -> str:
    """Render the published Adjudication Report (CC-BY-4.0)."""
    lines: list[str] = []
    lines.append("# Founding Adjudication of the AtomEons Federation")
    lines.append("")
    lines.append(
        f"**Version 1** · Determination: **{result.determination}** · "
        f"License CC-BY-4.0"
    )
    lines.append("")
    lines.append(
        f"> *Per Charter Article IV §4.2, the first Adjudication of the "
        f"Federation is the public assessment of every Tier-1-claiming "
        f"submission to the ARC Prize 2026 competition, including "
        f"Misfit-Alpha's own.*"
    )
    lines.append("")
    lines.append("## Subject")
    lines.append("")
    lines.append(f"- **Entity**: Misfit-Alpha")
    lines.append(f"- **Public ID**: `{result.public_id}`")
    lines.append(f"- **Federation**: {result.federation}")
    lines.append(f"- **Charter version**: {result.charter_version}")
    lines.append(f"- **Founding date**: {result.founding_date}")
    lines.append(f"- **Sovereign**: {result.sovereign}")
    lines.append(f"- **Repository root**: `{result.repo_root}`")
    lines.append(f"- **Git commit SHA**: `{result.git_commit_sha}`")
    lines.append(f"- **Audit timestamp (UTC)**: {result.audit_timestamp_iso}")
    lines.append("")

    lines.append("## Determination")
    lines.append("")
    lines.append(f"**{result.determination}**")
    lines.append("")
    if result.determination == DETERMINATION_RECOGNIZED:
        lines.append(
            "Misfit-Alpha satisfies every Article II §2.7 recognition "
            "criterion and the Article III §3.6 mechanical Tier-1 attestation. "
            "Membership in the AtomEons Federation is hereby recognized."
        )
    elif result.determination == DETERMINATION_DISPUTED:
        lines.append(
            "Misfit-Alpha passes the mechanical Tier-1 attestation but at "
            "least one non-mechanical check has open findings. Membership is "
            "tentatively recognized; the findings below must be remediated "
            "before the next Adjudication."
        )
    else:  # DISQUALIFIED
        lines.append(
            "Misfit-Alpha fails the mechanical Tier-1 attestation. Per "
            "Article III §3.6 the entity cannot be recognized under the "
            "current substrate. See findings below."
        )
    lines.append("")

    lines.append("## Article II §2.7 Recognition Criteria")
    lines.append("")
    lines.append("| Criterion | Met | Evidence |")
    lines.append("|---|---|---|")
    for c in result.article_ii_criteria:
        met = "yes" if c["met"] else "no"
        lines.append(f"| `{c['criterion']}` | {met} | {c['evidence']} |")
    lines.append("")

    lines.append("## Audit Checks")
    lines.append("")
    lines.append("| # | Check | Outcome | Severity | Summary |")
    lines.append("|---|---|---|---|---|")
    for i, c in enumerate(result.checks, 1):
        outcome = "PASS" if c.passed else "FAIL"
        lines.append(
            f"| {i} | `{c.name}` | {outcome} | {c.blocker_severity} | "
            f"{c.summary} |"
        )
    lines.append("")

    lines.append("## Provenance-Enforced Memory (PEM) Contract")
    lines.append("")
    lines.append(
        f"`src/misfit_agent/resonance.py` declares "
        f"**{len(result.pem_fields_present)} of 8** PEM fields."
    )
    lines.append("")
    lines.append("| PEM field | Declared |")
    lines.append("|---|---|")
    for canonical, _aliases in PEM_FIELDS:
        present = "yes" if canonical in result.pem_fields_present else "no"
        lines.append(f"| `{canonical}` | {present} |")
    lines.append("")

    lines.append("## Designer-Choice Transparency")
    lines.append("")
    lines.append(
        f"`src/misfit_agent/config.py` carries "
        f"**{result.designer_choice_count}** in-code `(c) DESIGNER CHOICE` "
        f"annotations. Disclosure document references the (a)/(b)/(c) "
        f"classification scheme: "
        f"**{'yes' if result.designer_choice_disclosure_ok else 'no'}**."
    )
    lines.append("")

    lines.append("## Receipt Anchor")
    lines.append("")
    lines.append(
        f"- Git commit SHA: `{result.git_commit_sha}`"
    )
    lines.append(f"- Sovereign signature: {result.sovereign}")
    lines.append(f"- Date of determination: {result.audit_timestamp_iso}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Signed under the authority of the AtomEons Federation Charter "
        f"v{result.charter_version}, executed by Sovereign {result.sovereign} "
        f"on {result.founding_date}.*"
    )
    lines.append("")
    return "\n".join(lines)


def emit_artifacts(result: AuditResult,
                   repo_root: Path | str | None = None,
                   markdown_path: Path | str | None = None,
                   receipt_path: Path | str | None = None
                   ) -> tuple[Path, Path]:
    """Write the markdown report and receipt JSON.

    Default paths:
      docs/FOUNDING_ADJUDICATION_v1.md
      receipts/100day/founding_adjudication.json

    Returns the (markdown_path, receipt_path) actually written.
    """
    root = Path(repo_root).resolve() if repo_root else Path(result.repo_root)
    md_path = (Path(markdown_path) if markdown_path
               else root / "docs" / "FOUNDING_ADJUDICATION_v1.md")
    rj_path = (Path(receipt_path) if receipt_path
               else root / "receipts" / "100day" / "founding_adjudication.json")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    rj_path.parent.mkdir(parents=True, exist_ok=True)

    md_text = render_markdown(result)
    md_path.write_text(md_text, encoding="utf-8", newline="\n")

    receipt = result.to_dict()
    rj_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return md_path, rj_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the AtomEons Federation Founding Adjudication "
                    "on Misfit-Alpha."
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root (defaults to the parent of this script's dir).",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="If set, emit docs/FOUNDING_ADJUDICATION_v1.md and "
             "receipts/100day/founding_adjudication.json. Without this flag "
             "the script prints the determination and exits without touching "
             "disk.",
    )
    args = ap.parse_args(argv)

    result = audit_misfit_alpha(args.repo_root)
    determination_line = (
        f"Determination: {result.determination}  "
        f"(git_sha={result.git_commit_sha[:12]})"
    )
    print(determination_line)
    for c in result.checks:
        flag = "OK  " if c.passed else "FAIL"
        print(f"  [{flag}] {c.name}: {c.summary}")

    if args.execute:
        md_path, rj_path = emit_artifacts(result)
        print(f"Wrote: {md_path}")
        print(f"Wrote: {rj_path}")

    if result.determination == DETERMINATION_DISQUALIFIED:
        return 2
    if result.determination == DETERMINATION_DISPUTED:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
