"""ARC-AGI-3 Misfit Phase B — state-change tracking + level-seeking heuristic.

Strategy upgrade over Phase A (Random baseline):
  - Track which actions produced state changes in recent frames
  - Prefer state-changing actions (entropy-seeking)
  - When game stalls (no state change for K turns), rotate to fresh action
  - When level_completed increases, lock the recent action sequence as
    "winning policy" for similar-fingerprint frames
  - Tier-1 strict throughout: deterministic, no LLM, no learned params

Project: Double Mamba — AGI Synergy Unit.
"""

import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
COMP_INPUT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"


def _setup_offline_install():
    wheel_dir = f"{COMP_INPUT}/arc_agi_3_wheels"
    if not os.path.isdir(wheel_dir):
        print(f"[setup] wheel dir not present at {wheel_dir}; skip install")
        return
    print(f"[setup] installing offline from {wheel_dir}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         "--find-links", wheel_dir, "arc-agi", "python-dotenv"],
        check=False,
    )


# ---------------------------------------------------------------------------
# The Phase B agent source — gets written to /kaggle/working/my_agent.py
# ---------------------------------------------------------------------------

MY_AGENT_SOURCE = '''"""AtomEons Misfit Phase-B — state-change + level-seek heuristic.

Tier-1 strict. Deterministic. No LLM, no pretrained weights, no learned
parameters at eval. Spelke COHESION + NUMEROSITY priors:
  - COHESION: track which actions produce state changes (objects move)
  - NUMEROSITY: track level_completed progression as the reward signal

Project: Double Mamba — AGI Synergy Unit.
"""

import hashlib
from collections import defaultdict, deque
from typing import Optional, Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


def _frame_fingerprint(frame) -> str:
    """Hash the frame's visible grid for state-change detection."""
    try:
        f = getattr(frame, "frame", None)
        if f is None:
            return "no_frame"
        flat = []
        # FrameData.frame is list[list[list[int]]] per arcengine
        def _walk(x, depth=0):
            if depth > 4:
                return
            if isinstance(x, list):
                for y in x:
                    _walk(y, depth + 1)
            else:
                flat.append(str(x))
        _walk(f)
        s = ",".join(flat[:4096])  # cap
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "exc"


class MyAgent(Agent):
    """Per StochasticGoose contract:
       - class name MUST be MyAgent
       - inherits from agents.agent.Agent
    """

    MAX_ACTIONS = float("inf")
    HARD_WALL_CLOCK_SECONDS = 8 * 3600 + 50 * 60

    # State-change memory
    _last_fingerprint = None
    _action_change_counts = defaultdict(int)  # action_int -> changes
    _action_try_counts = defaultdict(int)
    _action_history = deque(maxlen=10)
    _last_level = 0
    _stalled_for = 0

    def _normalize_actions(self, raw_actions) -> list:
        normalized = []
        for a in raw_actions or []:
            try:
                if hasattr(a, "value"):
                    normalized.append(GameAction(a))
                else:
                    normalized.append(GameAction(int(a)))
            except (ValueError, TypeError):
                continue
        return normalized

    def choose_action(self, latest_frame: FrameData,
                       frames: Optional[list] = None) -> GameAction:
        actions = self._normalize_actions(
            getattr(latest_frame, "available_actions", None) or []
        )
        if not actions:
            return GameAction.ACTION1

        # Frame change detection
        fp = _frame_fingerprint(latest_frame)
        changed = (self._last_fingerprint is not None
                   and fp != self._last_fingerprint)
        self._last_fingerprint = fp

        # Credit the last action with a state change if it occurred
        if changed and self._action_history:
            last_a = self._action_history[-1]
            self._action_change_counts[last_a] += 1
            self._stalled_for = 0
        else:
            self._stalled_for += 1

        # Level progression — reward signal
        cur_level = getattr(latest_frame, "levels_completed", 0) or 0
        if cur_level > self._last_level:
            # Just completed a level! Strong reinforcement for recent actions
            for a in list(self._action_history)[-3:]:
                self._action_change_counts[a] += 5
            self._last_level = cur_level
            self._stalled_for = 0

        # When stalled, prefer an action we've tried less
        if self._stalled_for >= 8:
            tries = [(self._action_try_counts[int(a.value)], int(a.value), a)
                     for a in actions]
            tries.sort()
            chosen = tries[0][2]
            self._action_try_counts[int(chosen.value)] += 1
            self._action_history.append(int(chosen.value))
            self._stalled_for = 0
            return chosen

        # Default: argmax by (change_count / try_count) "yield" metric
        best_yield = -1.0
        best_action = actions[0]
        for a in actions:
            ai = int(a.value)
            tries = self._action_try_counts[ai]
            changes = self._action_change_counts[ai]
            # +1 smoothing so unseen actions get explored
            yield_ratio = (changes + 1) / (tries + 1)
            if yield_ratio > best_yield:
                best_yield = yield_ratio
                best_action = a

        self._action_try_counts[int(best_action.value)] += 1
        self._action_history.append(int(best_action.value))
        return best_action

    def is_done(self, latest_frame: FrameData,
                 frames: Optional[list] = None) -> bool:
        if latest_frame is None:
            return False
        state = getattr(latest_frame, "state", None)
        if state == GameState.WIN:
            return True
        return False
'''


def _write_agent():
    os.makedirs(WORK, exist_ok=True)
    target = f"{WORK}/my_agent.py"
    with open(target, "w") as f:
        f.write(MY_AGENT_SOURCE)
    print(f"[setup] wrote agent to {target}")


def _bootstrap_rerun():
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

    templates_dir = f"{dst_framework}/agents/templates"
    os.makedirs(templates_dir, exist_ok=True)
    shutil.copy(f"{WORK}/my_agent.py", f"{templates_dir}/my_agent.py")

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

    os.environ["MPLBACKEND"] = "agg"
    rc = subprocess.run(
        [sys.executable, "main.py", "--agent", "myagent"],
        cwd=dst_framework, check=False,
    )
    print(f"[bootstrap] agent run finished with code {rc.returncode}")
    return True


def _write_dummy_parquet():
    import pandas as pd
    df = pd.DataFrame(
        data=[["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    )
    out = f"{WORK}/submission.parquet"
    df.to_parquet(out, index=False)
    print(f"[done] non-rerun dummy parquet written to {out}")


def main():
    print("[boot] AtomEons Misfit Phase-B — state-change + level-seek heuristic")
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
