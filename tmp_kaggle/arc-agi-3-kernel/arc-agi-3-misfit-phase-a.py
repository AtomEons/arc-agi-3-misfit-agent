"""ARC-AGI-3 Misfit Phase A — leaderboard entry.

Strategy: ship the Random baseline via the proven StochasticGoose 3-cell
contract. This LANDS us on the leaderboard with a working submission.
Phase B (next push) swaps Random for the full Misfit substrate
(Spelke priors + MCTS-PUCT + resonance library).

Top score as of 2026-06-17: 1.21% (Tufa Labs). Top is catchable —
substrate-augmented entries should clear 2%.

Project: Double Mamba — AGI Synergy Unit.
"""

import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
COMP_INPUT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"


def _setup_offline_install():
    """Cell-0 equivalent: install arc-agi + dotenv from bundled wheels."""
    wheel_dir = f"{COMP_INPUT}/arc_agi_3_wheels"
    if not os.path.isdir(wheel_dir):
        print(f"[setup] wheel dir not present at {wheel_dir}; skip install")
        return
    print(f"[setup] installing offline from {wheel_dir}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         "--find-links", wheel_dir, "arc-agi", "python-dotenv"],
        check=False,  # don't fail if already installed
    )


# ---------------------------------------------------------------------------
# The agent source — gets %%written to /kaggle/working/my_agent.py
# ---------------------------------------------------------------------------

MY_AGENT_SOURCE = '''"""AtomEons Misfit Phase-A — Random baseline.

Tier-1 strict baseline (no LLM, no pretrained weights, no learned params
at eval). Will be replaced by the full Misfit substrate in Phase B.

Project: Double Mamba — AGI Synergy Unit.
"""

import random
from typing import Optional, Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


class MyAgent(Agent):
    """Phase-A random-action baseline. Per StochasticGoose contract:
       - class name MUST be MyAgent
       - inherits from agents.agent.Agent
       - choose_action gets FrameData, returns GameAction
       - is_done gets FrameData (+ optional frames), returns bool
    """

    MAX_ACTIONS = float("inf")
    HARD_WALL_CLOCK_SECONDS = 8 * 3600 + 50 * 60

    def choose_action(self, latest_frame: FrameData,
                       frames: Optional[list] = None) -> GameAction:
        raw_actions = getattr(latest_frame, "available_actions", None) or []
        # gateway returns list[int]; templates expect GameAction enum
        normalized = []
        for a in raw_actions:
            try:
                if hasattr(a, "value"):
                    normalized.append(GameAction(a))
                else:
                    normalized.append(GameAction(int(a)))
            except (ValueError, TypeError):
                continue
        if not normalized:
            return GameAction.ACTION1
        return random.choice(normalized)

    def is_done(self, latest_frame: FrameData,
                 frames: Optional[list] = None) -> bool:
        if latest_frame is None:
            return False
        state = getattr(latest_frame, "state", None)
        if state == GameState.WIN:
            return True
        levels = getattr(latest_frame, "levels_completed", None)
        if levels is not None and levels > 0 and state == GameState.WIN:
            return True
        return False
'''


def _write_agent():
    """Cell-1 equivalent: write my_agent.py to /kaggle/working/."""
    os.makedirs(WORK, exist_ok=True)
    target = f"{WORK}/my_agent.py"
    with open(target, "w") as f:
        f.write(MY_AGENT_SOURCE)
    print(f"[setup] wrote agent to {target}")


def _bootstrap_rerun():
    """Cell-2 equivalent: the critical rerun-mode bootstrap.

    Per StochasticGoose contract:
      1. Wait for gateway at http://gateway:8001
      2. Copy framework from /kaggle/input/.../ARC-AGI-3-Agents to /kaggle/working/
      3. Copy my_agent.py into agents/templates/
      4. SLIM __init__.py rewrite (skip langgraph / smolagents etc.)
      5. .env with gateway endpoint
      6. Run main.py --agent myagent
    """
    # Wait for gateway
    print("[bootstrap] waiting for gateway")
    subprocess.run(
        ["curl", "--fail", "--retry", "999", "--retry-all-errors",
         "--retry-delay", "5", "--retry-max-time", "600",
         "http://gateway:8001/api/games"],
        check=False,
    )

    src_framework = f"{COMP_INPUT}/ARC-AGI-3-Agents"
    dst_framework = f"{WORK}/ARC-AGI-3-Agents"
    if not os.path.isdir(src_framework):
        print(f"[bootstrap] FATAL: framework not present at {src_framework}")
        return False
    if not os.path.isdir(dst_framework):
        shutil.copytree(src_framework, dst_framework)
    print(f"[bootstrap] framework copied to {dst_framework}")

    # Drop our agent into templates/
    templates_dir = f"{dst_framework}/agents/templates"
    os.makedirs(templates_dir, exist_ok=True)
    shutil.copy(f"{WORK}/my_agent.py", f"{templates_dir}/my_agent.py")
    print(f"[bootstrap] my_agent.py copied to templates/")

    # SLIM __init__.py rewrite — load-bearing
    init_path = f"{dst_framework}/agents/__init__.py"
    with open(init_path, "w") as f:
        f.write(
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
    print(f"[bootstrap] slim __init__.py rewritten")

    # .env
    env_path = f"{dst_framework}/.env"
    with open(env_path, "w") as f:
        f.write(
            "SCHEME=http\n"
            "HOST=gateway\n"
            "PORT=8001\n"
            "ARC_API_KEY=test-key-123\n"
            "ARC_BASE_URL=http://gateway:8001/\n"
            "OPERATION_MODE=online\n"
            "ENVIRONMENTS_DIR=\n"
            f"RECORDINGS_DIR={WORK}/server_recording\n"
        )
    print(f"[bootstrap] .env written")

    # Run agent
    os.environ["MPLBACKEND"] = "agg"
    print("[bootstrap] starting agent run")
    rc = subprocess.run(
        [sys.executable, "main.py", "--agent", "myagent"],
        cwd=dst_framework,
        check=False,
    )
    print(f"[bootstrap] agent run finished with code {rc.returncode}")
    return True


def _write_dummy_parquet():
    """Cell-3 equivalent: non-rerun-mode dummy parquet for Phase-A validation."""
    import pandas as pd
    df = pd.DataFrame(
        data=[["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    )
    out = f"{WORK}/submission.parquet"
    df.to_parquet(out, index=False)
    print(f"[done] non-rerun dummy parquet written to {out}")


def main():
    print("[boot] AtomEons Misfit Phase-A — ARC-AGI-3 leaderboard entry")
    print("[boot] Project: Double Mamba — AGI Synergy Unit")

    _setup_offline_install()
    _write_agent()

    is_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    print(f"[boot] rerun mode: {is_rerun}")

    if is_rerun:
        _bootstrap_rerun()
    else:
        _write_dummy_parquet()


if __name__ == "__main__":
    main()
