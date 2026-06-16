"""Build the Kaggle submission notebook for ARC-AGI-3 Misfit agent (v4).

Generates `notebooks/submission.ipynb` with 5 cells following the proven
StochasticGoose KAGGLE_IS_COMPETITION_RERUN contract:

  Cell 0  (markdown) Tier-1 disclosure — embedded body of docs/TIER_1_DISCLOSURE.md
  Cell 1  (code)     offline pip install from /kaggle/input/.../arc_agi_3_wheels/*.whl
  Cell 2  (code)     %%writefile /kaggle/working/my_agent.py — concatenated substrate
                     of all 12 misfit_agent source files (wave-4 modules included)
  Cell 3  (code)     KAGGLE_IS_COMPETITION_RERUN gateway bootstrap + slim
                     agents/__init__.py rewrite that registers Misfit in
                     AVAILABLE_AGENTS exactly mirroring the patch already at
                     _research/ARC-AGI-3-Agents/agents/__init__.py
  Cell 4  (code)     non-rerun dummy submission.parquet writer (Phase A) +
                     the agent run trigger (only fires in rerun)

The substrate concatenation strategy: take every src/misfit_agent/**/*.py file
in dependency order, strip relative `from .x import` lines (everything lives in
the same flat namespace inside my_agent.py), strip duplicate stdlib imports,
and wrap in `%%writefile`.

Run:    python scripts/build_notebook.py
Output: notebooks/submission.ipynb, notebooks/kernel-metadata.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).parent.parent
SRC = REPO / "src" / "misfit_agent"
NB_OUT = REPO / "notebooks" / "submission.ipynb"
META_OUT = REPO / "notebooks" / "kernel-metadata.json"
DISCLOSURE_DOC = REPO / "docs" / "TIER_1_DISCLOSURE.md"


# Concatenation order — strict dependency order.
# 1) config.py first (no internal deps).
# 2) perceptor.py (depended on by everything below).
# 3) episode.py (depends on perceptor; has a lazy local import of
#    tracker_hungarian that we strip — both classes coexist in the
#    flattened namespace, so the lazy import becomes a no-op resolution).
# 4) fingerprint.py (depends on episode).
# 5) resonance.py (depends on fingerprint).
# 6) rules/no_op.py, rules/translate.py (no internal deps).
# 7) world_model.py (depends on rules + config).
# 8) click_quantizer.py (depends on perceptor).
# 9) tracker_hungarian.py (depends on perceptor + config) — needed BEFORE
#    abstain_policy/mcts_puct/misfit_agent so its symbol exists.
# 10) goal_inducer.py (depends on perceptor).
# 11) abstain_policy.py (depends on episode + world_model + config).
# 12) action_search.py (depends on click_quantizer + episode + perceptor +
#     world_model).
# 13) mcts_puct.py (depends on config + uses world_model.predict via callback).
# 14) misfit_agent.py LAST — imports everything else.
MODULE_ORDER = [
    "config.py",
    "perceptor.py",
    "episode.py",
    "fingerprint.py",
    "resonance.py",
    "rules/no_op.py",
    "rules/translate.py",
    "world_model.py",
    "click_quantizer.py",
    "tracker_hungarian.py",
    "goal_inducer.py",
    "abstain_policy.py",
    "action_search.py",
    "mcts_puct.py",
    "misfit_agent.py",
]


def _strip_relative_imports(text: str) -> str:
    """Remove `from .x import Y` lines AND the leading module docstring.

    All names in a flat-concat file live in the same namespace, so the
    relative-import lines would just be no-ops at best and shadow-bind at
    worst. We also collapse a leading triple-quoted docstring per-module
    so the bundled file isn't littered with 12 docstrings.
    """
    out_lines: list[str] = []
    in_docstring = False
    docstring_quote: str | None = None
    saw_first_real_line = False
    for line in text.splitlines():
        stripped = line.strip()

        # Skip leading module docstring (must be the very first non-empty line).
        if not saw_first_real_line and not stripped:
            continue
        if not saw_first_real_line and (
            stripped.startswith('"""') or stripped.startswith("'''")
        ):
            docstring_quote = stripped[:3]
            # Single-line docstring: `"""one-liner"""`
            if stripped.count(docstring_quote) >= 2 and len(stripped) > 3:
                saw_first_real_line = True
                continue
            in_docstring = True
            saw_first_real_line = True
            continue
        if in_docstring:
            if docstring_quote and docstring_quote in line:
                in_docstring = False
            continue
        if not saw_first_real_line:
            saw_first_real_line = True

        # Strip ALL relative imports (single- and multi-dot).
        if re.match(r"^\s*from\s+\.{1,2}", line):
            continue
        # Strip __all__ declarations — they're per-module and irrelevant
        # to the flattened bundle.
        if re.match(r"^__all__\s*=", line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _dedup_stdlib_imports(text: str) -> str:
    """Collapse repeated stdlib imports across concatenated modules.

    Catches both `import X` and `from X import Y, Z` lines whose stripped
    form already appeared earlier in the bundle. Conservative: only dedups
    exact-string matches to avoid accidentally dropping a real second import
    that genuinely brings new names.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        m = re.match(
            r"^(from\s+[\w.]+\s+import\s+.+|import\s+[\w.]+(?:\s+as\s+\w+)?)$",
            line.strip(),
        )
        if m:
            key = line.strip()
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def _build_agent_module() -> str:
    """Concatenate the misfit substrate into a single self-contained module."""
    header: list[str] = [
        '"""',
        "MisfitAgent — Tier-1 Spelke-priors ARC-AGI-3 agent (built notebook bundle).",
        "",
        "Generated by scripts/build_notebook.py — do not edit by hand.",
        "Source: https://github.com/AtomEons/arc-agi-3-misfit-agent",
        "",
        "This bundle contains all 12 substrate modules flattened into one file:",
        "  config, perceptor, episode, fingerprint, resonance,",
        "  rules/no_op, rules/translate, world_model, click_quantizer,",
        "  tracker_hungarian, goal_inducer, abstain_policy, action_search,",
        "  mcts_puct, misfit_agent.",
        "",
        "NO LLM in the inference path. NO pretrained weights. NO ARC task-family",
        "hardcoding. Hand-authored typed rule grammar over Spelke core knowledge",
        "priors. Mechanically enforced by tests/test_tier1_attestation.py.",
        '"""',
        "",
        "# ----- shared standard-library and ecosystem imports --------------------",
        "from __future__ import annotations",
        "",
        "import copy",
        "import json",
        "import math",
        "import os",
        "import pathlib",
        "import random",
        "import time",
        "from collections import Counter",
        "from dataclasses import asdict, dataclass, field",
        "from typing import Any, Callable, Optional, Sequence",
        "",
        "import numpy as np",
        "",
        "# arcengine + framework supplied by the eval container.",
        "from arcengine import FrameData, GameAction, GameState",
        "from agents.agent import Agent",
        "",
    ]

    parts: list[str] = list(header)

    for rel_path in MODULE_ORDER:
        p = SRC / rel_path
        if not p.exists():
            raise FileNotFoundError(f"missing module: {rel_path}")
        text = p.read_text(encoding="utf-8")
        text = _strip_relative_imports(text)
        parts.append("")
        parts.append("# ============================================================")
        parts.append(f"# MODULE: {rel_path}")
        parts.append("# ============================================================")
        parts.append("")
        parts.append(text)

    body = "\n".join(parts)
    body = _dedup_stdlib_imports(body)

    # Framework expects MyAgent to subclass Agent. The Misfit class above
    # already extends Agent; expose it under the canonical alias so the
    # slim __init__.py rewrite can register it.
    body += (
        "\n\n"
        "# ============================================================\n"
        "# Framework entry point — Misfit registered as MyAgent.\n"
        "# ============================================================\n"
        "MyAgent = Misfit\n"
    )
    return body


def _load_disclosure_markdown() -> str:
    """Return the cell-0 markdown body.

    Embeds the body of docs/TIER_1_DISCLOSURE.md verbatim so the disclosure
    is in-notebook (Kaggle reviewers don't need to follow an external link).
    A leading attribution + repo link is prepended.
    """
    if not DISCLOSURE_DOC.exists():
        # Fail loudly — Tier-1 disclosure is non-optional.
        raise FileNotFoundError(
            f"missing required disclosure doc: {DISCLOSURE_DOC}"
        )
    body = DISCLOSURE_DOC.read_text(encoding="utf-8")
    header = (
        "# Misfit Agent — ARC-AGI-3 Tier-1 Submission\n"
        "\n"
        "> **Source:** https://github.com/AtomEons/arc-agi-3-misfit-agent  \n"
        "> **License:** Apache-2.0  \n"
        "> **Tier-1 attestation:** Spelke core knowledge object priors + "
        "hand-authored typed rule grammar. **No pretrained model weights of any kind. "
        "No language model in the inference path.**\n"
        "\n"
        "> Mechanically enforced by `tests/test_tier1_attestation.py` — fails CI "
        "if any of `transformers`, `openai`, `anthropic`, `llama_cpp`, "
        "`huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, "
        "`smolagents` are imported.\n"
        "\n"
        "---\n"
        "\n"
    )
    return header + body


def _build_notebook(agent_source: str, disclosure_md: str) -> dict[str, Any]:
    """Construct the .ipynb JSON document."""
    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    def md_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    # ----- Cell 1: offline wheel install (hedged path) ----------------------
    # Kaggle has moved competition mounts between /kaggle/input/<slug>/ and
    # /kaggle/input/competitions/<slug>/ historically. Try both at runtime so
    # the same notebook survives either layout.
    cell1_install = (
        "# Cell 1 — offline wheel install from the bundled competition data.\n"
        "# Internet is OFF (per kernel-metadata.json). Wheels live in the\n"
        "# competition dataset; mount path varies between Kaggle layouts so\n"
        "# resolve at runtime then hand the resolved dir to !pip install.\n"
        "import os\n"
        "_WHEEL_CANDIDATES = [\n"
        "    '/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',\n"
        "    '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',\n"
        "]\n"
        "_wheel_dir = next((p for p in _WHEEL_CANDIDATES if os.path.isdir(p)), None)\n"
        "assert _wheel_dir, f'no wheel dir found at any of: {_WHEEL_CANDIDATES}'\n"
        "print(f'misfit: wheel mount resolved to {_wheel_dir}')\n"
        "!pip install --no-index --find-links {_wheel_dir} arc-agi python-dotenv\n"
    )

    # ----- Cell 2: %%writefile of the entire substrate ----------------------
    cell2_writefile = "%%writefile /kaggle/working/my_agent.py\n" + agent_source

    # ----- Cell 3: KAGGLE_IS_COMPETITION_RERUN bootstrap + slim init -------
    # This cell:
    #   1) detects KAGGLE_IS_COMPETITION_RERUN (Phase B)
    #   2) waits for the eval gateway to come up
    #   3) copies the bundled ARC-AGI-3-Agents framework to a writable location
    #   4) drops our compiled my_agent.py into agents/templates/
    #   5) REWRITES agents/__init__.py to the slim version that registers Misfit
    #      under AVAILABLE_AGENTS — exactly mirrors the patch already applied
    #      to _research/ARC-AGI-3-Agents/agents/__init__.py
    #   6) writes the .env file pointing at gateway:8001 with RECORDINGS_DIR set
    #
    # IMPORTANT: the slim __init__.py body uses Python triple-single-quotes so
    # the embedded heredoc doesn't collide with the outer triple-double-quotes
    # of this build script.
    slim_init_body = (
        '"""Slim init — only Random + Misfit, no langgraph/smolagents/openai deps.\n'
        "\n"
        "This is the Kaggle-deploy rewrite. Mirrors the Day-12 architect plan\n"
        "slim version at _research/ARC-AGI-3-Agents/agents/__init__.py.\n"
        '"""\n'
        "from typing import Type, cast\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .recorder import Recorder\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import MyAgent\n"
        "\n"
        "load_dotenv()\n"
        "\n"
        "# Misfit is registered under BOTH `myagent` (canonical framework slug)\n"
        "# and `misfit` (operator-friendly alias). Random stays for sanity runs.\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n"
        '    "random": cast(Type[Agent], Random),\n'
        '    "myagent": cast(Type[Agent], MyAgent),\n'
        '    "misfit": cast(Type[Agent], MyAgent),\n'
        "}\n"
        "\n"
        "# add all recording files as valid agent names (Playback)\n"
        "for rec in Recorder.list():\n"
        "    AVAILABLE_AGENTS[rec] = Playback\n"
        "\n"
        '__all__ = ["Swarm", "Random", "MyAgent", "Agent", "Recorder", "Playback", "AVAILABLE_AGENTS"]\n'
    )

    env_body = (
        "SCHEME=http\n"
        "HOST=gateway\n"
        "PORT=8001\n"
        "ARC_API_KEY=test-key-123\n"
        "ARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    )

    cell3_rerun = (
        "# Cell 3 — KAGGLE_IS_COMPETITION_RERUN bootstrap + slim init rewrite.\n"
        "#\n"
        "# Detects rerun mode (Phase B = real eval). Phase A skips this whole\n"
        "# block and falls through to Cell 4 which writes the dummy parquet.\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
        "    # Wait for the eval gateway to come up before doing anything else.\n"
        "    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\\n"
        "          --retry-max-time 600 http://gateway:8001/api/games\n"
        "\n"
        "    # Copy the bundled framework to a writable location (hedged path).\n"
        "    _FW_CANDIDATES = [\n"
        "        '/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents',\n"
        "        '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents',\n"
        "    ]\n"
        "    _fw_src = next((p for p in _FW_CANDIDATES if os.path.isdir(p)), None)\n"
        "    assert _fw_src, f'no ARC-AGI-3-Agents found at any of: {_FW_CANDIDATES}'\n"
        "    !cp -r {_fw_src} /kaggle/working/ARC-AGI-3-Agents\n"
        "\n"
        "    # Install our agent module into agents/templates/.\n"
        "    !cp /kaggle/working/my_agent.py \\\n"
        "        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py\n"
        "\n"
        "    # ------------------------------------------------------------------\n"
        "    # SLIM agents/__init__.py — only Random + Misfit registered.\n"
        "    # langgraph / smolagents / openai are NOT in the offline wheel set;\n"
        "    # eagerly importing them crashes the framework load before MyAgent\n"
        "    # gets registered. This rewrite mirrors the patch already applied\n"
        "    # to _research/ARC-AGI-3-Agents/agents/__init__.py in the source repo.\n"
        "    # ------------------------------------------------------------------\n"
        "    _slim_init = " + repr(slim_init_body) + "\n"
        "    Path('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py').write_text(\n"
        "        _slim_init, encoding='utf-8'\n"
        "    )\n"
        "\n"
        "    # .env — gateway endpoint per the StochasticGoose rerun contract.\n"
        "    _env_body = " + repr(env_body) + "\n"
        "    Path('/kaggle/working/ARC-AGI-3-Agents/.env').write_text(\n"
        "        _env_body, encoding='utf-8'\n"
        "    )\n"
        "\n"
        "    print('Phase B bootstrap complete — agent run trigger fires in Cell 4.')\n"
        "else:\n"
        "    print('Not in KAGGLE_IS_COMPETITION_RERUN; Phase A dummy parquet path in Cell 4.')\n"
    )

    # ----- Cell 4: dummy submission.parquet (Phase A) + run trigger (Phase B)
    cell4_dummy_and_run = (
        "# Cell 4 — Phase A dummy submission.parquet + Phase B agent run trigger.\n"
        "#\n"
        "# Phase A (Kaggle save-and-validate, KAGGLE_IS_COMPETITION_RERUN unset):\n"
        "#   Just write the dummy parquet so Kaggle accepts the submission.\n"
        "#\n"
        "# Phase B (real rerun, KAGGLE_IS_COMPETITION_RERUN=1):\n"
        "#   Actually run the agent against the gateway. The framework writes its\n"
        "#   own submission.parquet via Recorder so we don't write a dummy here.\n"
        "import os\n"
        "\n"
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
        "    # Phase B — agent run trigger.\n"
        "    !cd /kaggle/working/ARC-AGI-3-Agents && \\\n"
        "        MPLBACKEND=agg \\\n"
        "        python main.py --agent myagent\n"
        "else:\n"
        "    # Phase A — dummy parquet so Kaggle accepts the save.\n"
        "    import pandas as pd\n"
        "    pd.DataFrame(\n"
        "        data=[['1_0', '1', True, 1]],\n"
        "        columns=['row_id', 'game_id', 'end_of_game', 'score'],\n"
        "    ).to_parquet('/kaggle/working/submission.parquet', index=False)\n"
        "    print('Wrote dummy submission.parquet for Phase A validation.')\n"
    )

    return {
        "cells": [
            md_cell(disclosure_md),
            code_cell(cell1_install),
            code_cell(cell2_writefile),
            code_cell(cell3_rerun),
            code_cell(cell4_dummy_and_run),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {
                "accelerator": "none",
                "dataSources": [],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _build_kernel_metadata(repo_user: str = "atommccree") -> dict[str, Any]:
    """Kaggle kernel-metadata.json — enables Phase A push via `kaggle kernels push`.

    Per the operator brief:
      - id_no    : atommccree/agi-in-a-video-shop-atom-eons-nostalgia
      - internet : false (offline-only; wheels bundled)
      - gpu      : false (Tier-1 substrate is CPU-only; T4 not required)
      - code_file: submission.ipynb
      - data     : the competition dataset itself contains arc_agi_3_wheels;
                   no extra dataset_sources entry required.
    """
    return {
        "id": f"{repo_user}/agi-in-a-video-shop-atom-eons-nostalgia",
        "title": "AGI in a Video Shop — AtomEons Nostalgia (Misfit Tier-1)",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
    }


def main() -> None:
    NB_OUT.parent.mkdir(parents=True, exist_ok=True)
    agent_source = _build_agent_module()
    disclosure_md = _load_disclosure_markdown()
    nb = _build_notebook(agent_source, disclosure_md)
    NB_OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    META_OUT.write_text(json.dumps(_build_kernel_metadata(), indent=2), encoding="utf-8")
    print(f"wrote {NB_OUT}  ({NB_OUT.stat().st_size:,} bytes)")
    print(f"wrote {META_OUT}")
    print(f"agent module: {len(agent_source.splitlines()):,} lines")
    print(f"modules concatenated: {len(MODULE_ORDER)}")


if __name__ == "__main__":
    main()
