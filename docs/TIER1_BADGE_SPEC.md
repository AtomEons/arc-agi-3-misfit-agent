# Tier-1 Attestation Badge — Open Spec v0.1

**Adopt for free. Apache-2.0 licensed.** A community standard for ARC-AGI submissions that want to make a defensible "no LLM, no pretrained weights, no public-corpus tuning" claim.

This is the badge:

```
[ TIER-1 ATTESTED · zero pretrained · zero LLM · CI-enforced ]
```

If your submission ships `tier1-badge.json` at the repo root and our verifier returns `PASS`, you may display the badge on your README, your Kaggle notebook, your paper, and your discussion posts. We will not police you. The CI is the policing.

## Why this exists

The 2026 ARC-AGI benchmarks score language-model agents at the noise floor: GPT-5.5 at 0.43%, Opus 4.7 at 0.18% on ARC-AGI-3 semi-private<sup>[1]</sup>. Several non-LLM substrates are competing under quiet Tier-1 conventions but the convention is unspoken. Without a community standard, "no LLM" is just a marketing line. With a standard, it's a verifiable claim.

We publish this spec under Apache-2.0 so that any team — including teams that outscore us — can adopt it without permission, royalty, or attribution beyond the badge text.

## The contract

A submission is **Tier-1 Attested** if and only if:

1. A file `tier1-badge.json` exists at the repository root and validates against the schema below.
2. A test in the repository's CI suite fails the build when any forbidden import or pretrained-weight string appears in source code.
3. The submission's runtime inference path makes no network calls to LLM providers.
4. Score weights, search thresholds, and rule grammar parameters are frozen by a git tag published BEFORE first public-leaderboard submission. Any post-freeze change is recorded in a `CHANGELOG.md` entry naming the finding that justified it.
5. If the submission uses a memory store (replay buffer, resonance library, retrieval index), every entry carries a `source_tag` enforced at write time to reject pre-seeded entries.

That is the whole contract. Five rules.

## `tier1-badge.json` schema

```json
{
  "spec": "tier1-attestation/v0.1",
  "submission": {
    "name": "string — submission display name",
    "version": "string — git tag or SHA",
    "license": "string — SPDX identifier (Apache-2.0, MIT, etc.)",
    "url": "string — repo URL"
  },
  "claim": {
    "tier": 1,
    "competitions": ["arc-prize-2026-arc-agi-3", "arc-prize-2026-arc-agi-2"],
    "rationale_doc": "string — path to disclosure md (e.g. docs/TIER_1_DISCLOSURE.md)"
  },
  "attestation": {
    "ci_grep_script": "string — path to forbidden-import test (e.g. tests/test_tier1_attestation.py)",
    "frozen_constants_git_tag": "string — e.g. constants-frozen-2026-06-30",
    "memory_source_tag_enforced": true,
    "no_network_at_inference": true
  },
  "exclusions": {
    "no_imports": ["torch", "transformers", "openai", "anthropic", "llama_cpp",
                    "huggingface_hub", "sentence_transformers", "langchain",
                    "langgraph", "smolagents"],
    "no_weight_files": [".gguf", ".safetensors", ".pth", ".ckpt"],
    "no_model_strings": ["gpt-?\\d+", "claude-?\\d+", "gemini-?\\d+", "mamba-?\\d+",
                          "llama-?\\d+", "mistral-?", "qwen-?\\d+"]
  },
  "verified_at_unix": 0,
  "verifier_sha": "string — sha256 of the verifier that signed this badge"
}
```

## The canonical verifier

`scripts/tier1_badge_verify.py` in this repository is the canonical verifier. To re-run on any repo:

```bash
python tier1_badge_verify.py /path/to/other-repo
```

Exit code 0 = `PASS`. Non-zero = the reasons appear on stdout.

The verifier:
- Walks `submission.name/`'s source tree
- Greps every `.py` file for the patterns in `exclusions.no_imports` and `exclusions.no_model_strings`
- Walks the same tree for binary files matching `exclusions.no_weight_files`
- Reads the CI grep script path and ensures it runs in the project's test suite
- Confirms the `frozen_constants_git_tag` exists in `git tag --list`
- Reads each `*.jsonl` memory store mentioned in `attestation.memory_source_tag_enforced` and rejects any entry without a `source_tag` field

PASS produces a one-line summary plus a `verifier_sha` you copy back into the badge file.

## Adoption guide for other teams

1. Add `tier1-badge.json` to your repo root with your details.
2. Add a forbidden-import test to your CI. (Copy ours under Apache-2.0; we won't object.)
3. Tag your constants-frozen state: `git tag constants-frozen-YYYY-MM-DD && git push --tags`.
4. Add `source_tag` to your memory store writes.
5. Run our verifier: `python tier1_badge_verify.py .`
6. Display the badge on your README. We suggest:

```markdown
[![Tier-1 Attested](https://img.shields.io/badge/Tier--1-attested-success?style=flat-square)](docs/TIER_1_DISCLOSURE.md)
```

That's it. You're now Tier-1 Attested. We will list every passing submission in `community/attested-submissions.json` if you open a PR.

## Tier-2 and Tier-3 — coming as separate specs

This v0.1 spec covers Tier-1 only. Tier-2 (substrate + frozen bundled LLM heuristic) and Tier-3 (substrate + cloud judge lane) will get their own specs once at least three Tier-1 submissions have adopted v0.1. The goal is for the **disclosure shape** to be standardized before the contaminated-tier claims become common.

## Why we expect the standard to spread

A Tier-1 Attested badge is information. Information that scoring agents care about and reviewers care about. A submission that displays the badge AND scores higher than ours has a stronger claim than ours. A submission that displays the badge AND scores lower is still defensible as a pure-priors approach. A submission that lacks the badge is sending a signal whether it intended to or not.

Adopting it costs the adopting team about 30 minutes. Not adopting it is a choice that grows louder as more teams adopt.

## License

This spec is Apache-2.0. Implementations are at your discretion. The badge text itself ("TIER-1 ATTESTED · zero pretrained · zero LLM · CI-enforced") is unrestricted — display it freely on any submission that passes the verifier.

## References

[1] ARC Prize Foundation. (2026-05-01). *Analyzing GPT-5.5 & Opus 4.7 with ARC-AGI-3.* https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis

---

**Spec maintainers welcome.** Open a PR at https://github.com/AtomEons/arc-agi-3-misfit-agent to propose v0.2.
