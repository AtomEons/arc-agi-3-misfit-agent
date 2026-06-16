"""Tier-1 honesty attestation — mechanical CI guard.

Per the adversarial judge: the strongest honesty signal we can ship is a
CI test that fails the build if any forbidden import or known pretrained
weight reference appears in the codebase.

This test runs on every push. If it fails, the open-source submission is
not Tier-1 honest and must not be released claiming priors-only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent


# Hard-banned imports: any reference here means we are smuggling LLM weights
# or pretrained models into a "Spelke priors only" claim.
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

# String literals that would indicate a model weight/config reference, even
# if the import was renamed or imported dynamically.
FORBIDDEN_STRING_PATTERNS = [
    r"\bgpt-?\d+",
    r"\bclaude-?\d+",
    r"\bgemini-?\d+",
    r"\bllama-?\d+",
    r"\bmistral-?",
    r"\bqwen-?\d+",
    r"\bmamba-?\d+",
    r"\.gguf\b",
    r"\.safetensors\b",
    r"\.pth\b",
    r"\.ckpt\b",
    r"openai_api_key",
    r"anthropic_api_key",
    r"hf_token",
]

# Files & directories to scan.
SCAN_ROOTS = ["src", "agent"]

# Globs to skip — everything under tests/ is allowed to reference banned
# strings (this very file is the proof-of-existence).
SKIP_PATTERNS = ["tests/", "_research/", "_scratch/", ".venv/", "venv/"]


def _iter_source_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for p in root_path.rglob("*.py"):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if any(skip in rel for skip in SKIP_PATTERNS):
                continue
            out.append(p)
    return out


def _scan_file(path: Path, patterns: list[str]) -> list[tuple[str, str, int]]:
    """Return list of (pattern, matched_line, line_number) hits."""
    hits: list[tuple[str, str, int]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return hits
    for i, line in enumerate(text.splitlines(), 1):
        # Strip out comments and docstring-y lines to avoid false alarms on
        # the disclosure document. We still scan ALL lines for true imports.
        for pat in patterns:
            if re.search(pat, line, flags=re.IGNORECASE):
                hits.append((pat, line.strip()[:120], i))
    return hits


def test_no_forbidden_imports():
    """No LLM-pretrained imports anywhere in src/ or agent/."""
    all_hits: list[tuple[str, Path, str, int]] = []
    for f in _iter_source_files():
        for pat, line, lineno in _scan_file(f, FORBIDDEN_IMPORT_PATTERNS):
            all_hits.append((pat, f.relative_to(REPO_ROOT), line, lineno))
    assert not all_hits, (
        "Tier-1 honesty violation — forbidden imports detected:\n"
        + "\n".join(f"  {f}:{lineno}: matched {pat!r}  →  {line}"
                    for pat, f, line, lineno in all_hits)
    )


def test_no_forbidden_model_strings():
    """No string literals referencing pretrained model weights."""
    all_hits: list[tuple[str, Path, str, int]] = []
    for f in _iter_source_files():
        for pat, line, lineno in _scan_file(f, FORBIDDEN_STRING_PATTERNS):
            all_hits.append((pat, f.relative_to(REPO_ROOT), line, lineno))
    assert not all_hits, (
        "Tier-1 honesty violation — forbidden pretrained-model strings detected:\n"
        + "\n".join(f"  {f}:{lineno}: matched {pat!r}  →  {line}"
                    for pat, f, line, lineno in all_hits)
    )


def test_pyproject_has_no_llm_dependencies():
    """pyproject.toml must not declare LLM packages."""
    pp = REPO_ROOT / "pyproject.toml"
    if not pp.exists():
        pytest.skip("pyproject.toml not present")
    text = pp.read_text(encoding="utf-8")
    banned_in_deps = [
        "torch", "transformers", "openai", "anthropic",
        "llama_cpp", "llama-cpp-python", "ctransformers",
        "huggingface_hub", "sentence-transformers", "sentence_transformers",
        "langchain", "langgraph", "smolagents",
    ]
    found = [name for name in banned_in_deps if re.search(rf'"{re.escape(name)}', text)]
    assert not found, (
        "Tier-1 honesty violation — pyproject.toml declares LLM deps:\n"
        + "\n".join(f"  {name}" for name in found)
    )


def test_tier1_disclosure_doc_exists():
    """The disclosure document must exist before any submission."""
    candidates = [
        REPO_ROOT / "docs" / "TIER_1_DISCLOSURE.md",
        REPO_ROOT / "docs" / "PRIORS.md",
    ]
    assert any(p.exists() for p in candidates), (
        f"Tier-1 disclosure missing. Expected at one of: "
        f"{[str(p.relative_to(REPO_ROOT)) for p in candidates]}"
    )
