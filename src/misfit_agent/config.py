"""Frozen configuration — every magic number lives here with provenance.

Per the judge panel's must-fix #2: every threshold must be classified as
  (a) DERIVED FROM A PRIOR  — geometric truth or Spelke core knowledge
  (b) BUDGET HEURISTIC      — math from the scoring rule or wall-clock
  (c) TUNED ON PUBLIC GAMES — public-corpus prior (must be disclosed)

If a value is in category (c), it must be disclosed in TIER_1_DISCLOSURE.md
AND frozen before any private-set submission.

This file MUST be committed to git before Day-10 evaluation so threshold
sweeps cannot be done silently in the final week.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerConfig:
    # Hungarian matching cost weights (Day-4 deliverable).
    # Classification: (a) DERIVED FROM PRIOR — generic geometric distance / topology / color identity.
    # These weights encode the relative importance of position vs shape vs identity for cohesion.
    # NOT tuned on any specific game grid. If we sweep them on public games, we must move them to (c).
    alpha_centroid_dist: float = 1.0
    beta_shape_hamming: float = 0.5
    gamma_color_mismatch: float = 2.0


@dataclass(frozen=True)
class WorldModelConfig:
    # (a) DERIVED FROM PRIOR — same-observation requires same-prediction is the consistency axiom.
    min_observations_for_trust: int = 3
    # (b) BUDGET HEURISTIC — wall-clock per-step simulator target (µs).
    target_us_per_step: int = 50
    # (b) BUDGET HEURISTIC — episodic memory ceiling (states per game).
    max_states_per_game: int = 10_000


@dataclass(frozen=True)
class PseudocountConfig:
    # (a) DERIVED FROM PRIOR — standard novelty-bonus form alpha / sqrt(N+1).
    novelty_alpha: float = 1.0


@dataclass(frozen=True)
class AbstainConfig:
    # (b) BUDGET HEURISTIC — derive from quadratic scoring math.
    # Score = (human/agent)^2. Half-life of marginal score gain is at agent_actions = 2*human_baseline.
    # Default 25 = empirical band where pseudocount novelty typically plateaus on small grids.
    # MUST be re-derived from a published calculation, not asserted — judge auditor flag.
    min_actions_before_abstain: int = 25
    # (a) DERIVED FROM PRIOR — when world-model variance > 20% of predictions, hypothesis class is wrong.
    world_model_variance_threshold: float = 0.20
    # (a) DERIVED FROM PRIOR — pseudocount slope below this = exploration plateau.
    novelty_plateau_slope: float = 0.05


@dataclass(frozen=True)
class MCTSConfig:
    # (b) BUDGET HEURISTIC — PUCT exploration constant; canonical value in AlphaZero literature.
    c_puct: float = 1.41
    # (b) BUDGET HEURISTIC — per-action search depth cap.
    max_depth: int = 6
    # (b) BUDGET HEURISTIC — rollouts per real action; trades quality vs wall clock.
    rollouts_per_action: int = 200
    # (b) BUDGET HEURISTIC — hard timeout per choose_action call (ms).
    hard_timeout_ms: int = 500


@dataclass(frozen=True)
class BudgetConfig:
    # (b) BUDGET HEURISTIC — Kaggle's 9h hard cap minus 5 min cold-start overhead minus 5 min safety.
    wall_clock_kill_seconds: int = 8 * 3600 + 50 * 60   # 8h50m
    # (a) DERIVED FROM PRIOR — the agents.agent.Agent default. Server-side cap may be 80;
    # MUST verify empirically on Day 1 (judge Kaggle-reality must-fix).
    # Raised to 400 in plan but treated as advisory until verified.
    max_actions_per_game: int = 80
    # (b) BUDGET HEURISTIC — anticipated 110 private games per Kaggle eval batch.
    expected_total_games: int = 110


@dataclass(frozen=True)
class FingerprintConfig:
    # (b) BUDGET HEURISTIC — total signature dimensionality.
    total_dim: int = 50


@dataclass(frozen=True)
class FrozenConfig:
    """Singleton view over all module configs.

    To change a value here AFTER first submission, the operator must:
      1. Tag the prior submission's git SHA
      2. Document the change in CHANGELOG.md with rationale
      3. Note which judge must-fix justified the change
    """
    tracker: TrackerConfig = TrackerConfig()
    world_model: WorldModelConfig = WorldModelConfig()
    pseudocount: PseudocountConfig = PseudocountConfig()
    abstain: AbstainConfig = AbstainConfig()
    mcts: MCTSConfig = MCTSConfig()
    budget: BudgetConfig = BudgetConfig()
    fingerprint: FingerprintConfig = FingerprintConfig()


CONFIG = FrozenConfig()
