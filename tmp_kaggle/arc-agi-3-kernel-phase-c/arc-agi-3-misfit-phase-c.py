"""ARC-AGI-3 Misfit Phase C — FULL SUBSTRATE.

Strategy upgrade over Phase A (Random) and Phase B (state-change heuristic):
  - Full Misfit substrate: WorldModel.fit_with_refinement (HRM-style outer
    refinement, +13pp from arcprize.org 2025-08-15 analysis), MCTS-PUCT
    with 200 rollouts, ResonanceLibrary K-NN, HungarianTracker for
    CONTINUITY prior on object correspondences, GoalInducer for ranked
    goal hypotheses, AbstainPolicy three-conjunction abstain trigger,
    EpisodeTracker, fingerprint_episode, perceive_frame Spelke priors,
    click_quantizer for action-space discretization.
  - Tier-1 strict throughout (CI-grep attested): no LLM in inference, no
    pretrained weights, no learned parameters at eval. Spelke core
    knowledge priors only + experience-only resonance.

Top score on ARC-AGI-3 as of 2026-06-17: 1.21% (Tufa Labs). Catchable.
Phase C is our real shot at climbing.

Project: Double Mamba — AGI Synergy Unit.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
COMP_INPUT = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
_INPUT_BASE = Path("/kaggle/input")


def _find_substrate_root() -> Path | None:
    """Search /kaggle/input for the misfit_agent package (or its zip)."""
    for p in _INPUT_BASE.rglob("misfit_agent.py"):
        # Found misfit_agent.py — parent's parent is the substrate root
        if p.parent.name == "misfit_agent":
            return p.parent.parent
    # Fallback: look for the dataset directory by user/slug
    candidates = [
        _INPUT_BASE / "datasets" / "atommccree" / "atomeons-misfit-substrate",
        _INPUT_BASE / "atomeons-misfit-substrate",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _setup_offline_install():
    wheel_dir = COMP_INPUT / "arc_agi_3_wheels"
    if not wheel_dir.is_dir():
        print(f"[setup] wheel dir not present at {wheel_dir}; skip install")
        return
    print(f"[setup] installing offline from {wheel_dir}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         "--find-links", str(wheel_dir), "arc-agi", "python-dotenv",
         "numpy", "scipy"],
        check=False,
    )


# ---------------------------------------------------------------------------
# The Phase C agent — wraps Misfit substrate as MyAgent.
# ---------------------------------------------------------------------------


def _build_my_agent_source(substrate_root: Path) -> str:
    """Build the my_agent.py source. Imports our Misfit class and
    re-exports as MyAgent per StochasticGoose contract."""
    return f'''"""AtomEons Misfit Phase-C — full Spelke-priors substrate.

Wraps the Misfit class from the vendored AtomEons substrate (full
WorldModel.fit_with_refinement + MCTS-PUCT + ResonanceLibrary +
HungarianTracker + GoalInducer + AbstainPolicy + perceive_frame +
click_quantizer) as the Kaggle-required MyAgent.

Tier-1 strict. Project: Double Mamba — AGI Synergy Unit.
"""

import sys

# Mount the substrate package
_SUBSTRATE_ROOT = "{substrate_root}"
if _SUBSTRATE_ROOT not in sys.path:
    sys.path.insert(0, _SUBSTRATE_ROOT)

try:
    from misfit_agent.misfit_agent import Misfit
except ImportError as exc:
    print(f"[fatal] could not import Misfit substrate: {{exc}}")
    raise


class MyAgent(Misfit):
    """Per StochasticGoose contract:
       - class name MUST be MyAgent
       - inherits from agents.agent.Agent (via Misfit -> Agent)
    """
    pass
'''


def _write_agent(substrate_root: Path):
    WORK.mkdir(parents=True, exist_ok=True)
    src = _build_my_agent_source(substrate_root)
    target = WORK / "my_agent.py"
    target.write_text(src)
    print(f"[setup] wrote agent wrapper to {target}")
    print(f"[setup] substrate root: {substrate_root}")


def _bootstrap_rerun(substrate_root: Path):
    print("[bootstrap] waiting for gateway")
    subprocess.run(
        ["curl", "--fail", "--retry", "999", "--retry-all-errors",
         "--retry-delay", "5", "--retry-max-time", "600",
         "http://gateway:8001/api/games"],
        check=False,
    )

    src_framework = COMP_INPUT / "ARC-AGI-3-Agents"
    dst_framework = WORK / "ARC-AGI-3-Agents"
    if not src_framework.is_dir():
        print(f"[bootstrap] FATAL: framework not present at {src_framework}")
        return False
    if not dst_framework.exists():
        shutil.copytree(src_framework, dst_framework)
    print(f"[bootstrap] framework copied to {dst_framework}")

    templates_dir = dst_framework / "agents" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(WORK / "my_agent.py", templates_dir / "my_agent.py")

    init_path = dst_framework / "agents" / "__init__.py"
    init_path.write_text(
        "from typing import Type, cast\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import MyAgent\n"
        "\n"
        "load_dotenv()\n"
        "\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = "
        "{\"random\": Random, \"myagent\": MyAgent}\n"
    )

    env_path = dst_framework / ".env"
    env_path.write_text(
        "SCHEME=http\n"
        "HOST=gateway\n"
        "PORT=8001\n"
        "ARC_API_KEY=test-key-123\n"
        "ARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        f"RECORDINGS_DIR={WORK}/server_recording\n"
    )

    os.environ["MPLBACKEND"] = "agg"
    # Ensure substrate path is available to the running agent
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{substrate_root}{os.pathsep}{pythonpath}"
    rc = subprocess.run(
        [sys.executable, "main.py", "--agent", "myagent"],
        cwd=str(dst_framework), check=False,
    )
    print(f"[bootstrap] agent run finished with code {rc.returncode}")
    return True


def _write_dummy_parquet():
    import pandas as pd
    df = pd.DataFrame(
        data=[["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    )
    out = WORK / "submission.parquet"
    df.to_parquet(str(out), index=False)
    print(f"[done] non-rerun dummy parquet written to {out}")


def main():
    print("[boot] AtomEons Misfit Phase-C — FULL SUBSTRATE")
    print("[boot] Project: Double Mamba — AGI Synergy Unit")
    print("[boot] Tier-1 strict: WorldModel + MCTS-PUCT + Resonance + Hungarian + GoalInducer + AbstainPolicy + perceive_frame")

    _setup_offline_install()

    # Defensive: find the substrate root by searching /kaggle/input
    print(f"[diag] /kaggle/input contents: {sorted(os.listdir('/kaggle/input'))}")
    substrate_root = _find_substrate_root()
    if substrate_root is None:
        print("[fatal] could not locate misfit_agent substrate; "
              "falling back to dummy parquet")
        _write_dummy_parquet()
        return
    print(f"[diag] substrate found at: {substrate_root}")

    _write_agent(substrate_root)

    is_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    print(f"[boot] rerun mode: {is_rerun}")
    if is_rerun:
        _bootstrap_rerun(substrate_root)
    else:
        _write_dummy_parquet()


if __name__ == "__main__":
    main()
