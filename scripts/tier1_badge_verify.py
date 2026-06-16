#!/usr/bin/env python3
"""Tier-1 Attestation Badge — canonical verifier.

Usage:
    python tier1_badge_verify.py [path-to-repo]

Exit codes:
    0   PASS — submission meets all five contract clauses
    1   FAIL — reasons printed to stdout
    2   ERROR — verifier could not run (missing file, malformed JSON)

This verifier is canonical for spec v0.1. Run it on any candidate
submission to produce a binding pass/fail. The verifier itself is
Apache-2.0 licensed; copy it freely.

Spec: docs/TIER1_BADGE_SPEC.md
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SPEC_VERSION = "tier1-attestation/v0.1"


def _sha256_self() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _walk_python_sources(root: Path) -> list[Path]:
    skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__",
                 ".pytest_cache", ".mypy_cache", ".ruff_cache",
                 "_research", "_scratch", "tests", "test"}
    return [p for p in root.rglob("*.py")
            if not any(part in skip_dirs for part in p.parts)]


def _walk_for_weight_files(root: Path, extensions: list[str]) -> list[Path]:
    skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    hits: list[Path] = []
    for ext in extensions:
        for p in root.rglob(f"*{ext}"):
            if any(part in skip_dirs for part in p.parts):
                continue
            hits.append(p)
    return hits


def _check_no_imports(root: Path, banned: list[str]) -> list[str]:
    findings: list[str] = []
    patterns = [(name, re.compile(rf"\b(?:from|import)\s+{re.escape(name)}\b"))
                for name in banned]
    for src in _walk_python_sources(root):
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.split("#", 1)[0]
            for name, pat in patterns:
                if pat.search(stripped):
                    rel = src.relative_to(root)
                    findings.append(f"  forbidden import {name!r}: {rel}:{i}: {stripped.strip()[:80]}")
    return findings


def _check_no_model_strings(root: Path, patterns: list[str]) -> list[str]:
    findings: list[str] = []
    compiled = [(pat, re.compile(pat, re.IGNORECASE)) for pat in patterns]
    for src in _walk_python_sources(root):
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.split("#", 1)[0]
            for pat_str, pat in compiled:
                if pat.search(stripped):
                    rel = src.relative_to(root)
                    findings.append(f"  forbidden string {pat_str!r}: {rel}:{i}: {stripped.strip()[:80]}")
    return findings


def _check_no_weight_files(root: Path, extensions: list[str]) -> list[str]:
    hits = _walk_for_weight_files(root, extensions)
    return [f"  weight file present: {p.relative_to(root)}" for p in hits]


def _check_ci_grep_script(root: Path, script_path: str) -> list[str]:
    target = root / script_path
    if not target.exists():
        return [f"  CI grep script missing: {script_path}"]
    text = target.read_text(encoding="utf-8", errors="ignore")
    must_contain = ["torch", "transformers", "openai", "anthropic"]
    missing = [name for name in must_contain if name not in text]
    if missing:
        return [f"  CI grep script {script_path} does not mention: {', '.join(missing)}"]
    return []


def _check_frozen_tag(root: Path, expected_tag: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "tag", "--list"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return [f"  git tag check failed; cannot verify frozen-constants tag"]
    tags = set(result.stdout.split())
    if expected_tag not in tags:
        return [f"  frozen-constants tag missing: {expected_tag!r} not in repo's git tags"]
    return []


def _check_memory_source_tags(root: Path) -> list[str]:
    findings: list[str] = []
    for jsonl in root.rglob("*.jsonl"):
        if any(part in {".git", "_research", "_scratch", "node_modules"} for part in jsonl.parts):
            continue
        if "memory" in jsonl.name.lower() or "library" in jsonl.name.lower() or "resonance" in jsonl.name.lower():
            try:
                for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "source_tag" not in entry and "source" not in entry:
                        rel = jsonl.relative_to(root)
                        findings.append(f"  memory entry missing source_tag: {rel}:{i}")
                        break
            except OSError:
                continue
    return findings


def verify(repo_root: Path) -> tuple[bool, list[str], dict]:
    badge_path = repo_root / "tier1-badge.json"
    if not badge_path.exists():
        return False, [f"tier1-badge.json not found at {badge_path}"], {}
    try:
        badge = json.loads(badge_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, [f"tier1-badge.json malformed: {e}"], {}

    if badge.get("spec") != SPEC_VERSION:
        return False, [f"badge spec mismatch: expected {SPEC_VERSION}, got {badge.get('spec')!r}"], badge

    findings: list[str] = []
    exclusions = badge.get("exclusions", {})
    attest = badge.get("attestation", {})

    findings += _check_no_imports(repo_root, exclusions.get("no_imports", []))
    findings += _check_no_model_strings(repo_root, exclusions.get("no_model_strings", []))
    findings += _check_no_weight_files(repo_root, exclusions.get("no_weight_files", []))

    if attest.get("ci_grep_script"):
        findings += _check_ci_grep_script(repo_root, attest["ci_grep_script"])
    else:
        findings.append("  attestation.ci_grep_script missing from badge")

    if attest.get("frozen_constants_git_tag"):
        findings += _check_frozen_tag(repo_root, attest["frozen_constants_git_tag"])
    else:
        findings.append("  attestation.frozen_constants_git_tag missing from badge")

    if attest.get("memory_source_tag_enforced"):
        findings += _check_memory_source_tags(repo_root)

    if not attest.get("no_network_at_inference"):
        findings.append("  attestation.no_network_at_inference is not True")

    return (len(findings) == 0), findings, badge


def main() -> int:
    repo_root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not repo_root.is_dir():
        print(f"ERROR: not a directory: {repo_root}")
        return 2

    print(f"Tier-1 Attestation Badge — verifier v0.1")
    print(f"Spec: {SPEC_VERSION}")
    print(f"Repo: {repo_root}")
    print()

    passed, findings, badge = verify(repo_root)
    if passed:
        verifier_sha = _sha256_self()
        print(f"  PASS  {badge.get('submission', {}).get('name', '?')} "
              f"@ {badge.get('submission', {}).get('version', '?')}")
        print(f"  verifier_sha: {verifier_sha[:16]}...")
        print()
        print("  Update tier1-badge.json with:")
        print(f'    "verifier_sha": "{verifier_sha}"')
        print(f'    "verified_at_unix": <current-unix-timestamp>')
        return 0

    print(f"  FAIL  {len(findings)} finding(s):")
    for f in findings:
        print(f)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
