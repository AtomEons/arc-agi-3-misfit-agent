"""Frozen configuration — every magic number lives here with provenance.

Per the judge panel's must-fix #2 AND the Gemini Consigliere audit
(2026-06-16, `docs/HANDOFF_GEMINI_CONSIGLIERE.md`), every threshold must
be classified as one of:

  (a) DERIVED FROM A PRIOR  — geometric truth, Spelke core knowledge, OR
                               a published-form constant whose SHAPE is the
                               prior (specific scalar value still designer).
  (b) BUDGET HEURISTIC      — math from the scoring rule, wall-clock, or
                               framework defaults.
  (c) DESIGNER CHOICE       — author-selected scalar frozen pre-evaluation.
                               MUST be disclosed in TIER_1_DISCLOSURE.md
                               AND in PAPER §7 Limitations.

The Gemini Consigliere correctly flagged that prior versions of this file
classified designer-selected scalars as (a) DERIVED FROM PRIOR. We have
re-classified to (c) DESIGNER CHOICE for the specific scalar values, while
preserving (a) for the prior FORM where applicable.

This file MUST be committed to git before the first Phase B submission.
Constants-frozen git tag policy applies: a `constants-frozen-<date>` tag
is published before each Kaggle Phase B submit, and any change AFTER that
tag is recorded in CHANGELOG.md with the judge finding that justified it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerConfig:
    # Hungarian matching cost weights.
    # FORM: linear weighting of (centroid distance, shape Hamming, color mismatch)
    #       is admissible as (a) — generic geometric / topological distance.
    # SCALARS: α=1.0, β=0.5, γ=2.0 are (c) DESIGNER CHOICE — author selected
    #          these ratios pre-evaluation; not derived from data, not tuned
    #          on public games. Frozen pre-eval per Gemini Consigliere audit.
    alpha_centroid_dist: float = 1.0
    beta_shape_hamming: float = 0.5
    gamma_color_mismatch: float = 2.0


@dataclass(frozen=True)
class WorldModelConfig:
    # FORM: (a) — consistency axiom (same observation → same prediction)
    # SCALAR (3): (c) DESIGNER CHOICE — author picked a small natural number
    #             as the minimum trust threshold pre-evaluation.
    min_observations_for_trust: int = 3
    # (b) BUDGET HEURISTIC — wall-clock per-step simulator target (µs).
    target_us_per_step: int = 50
    # (b) BUDGET HEURISTIC — episodic memory ceiling (states per game).
    max_states_per_game: int = 10_000


@dataclass(frozen=True)
class PseudocountConfig:
    # FORM: (a) — standard count-based novelty bonus α / √(N+1)
    #             (Bellemare et al. 2016).
    # SCALAR (1.0): (c) DESIGNER CHOICE — the α multiplier itself.
    novelty_alpha: float = 1.0


@dataclass(frozen=True)
class AbstainConfig:
    # (b) BUDGET HEURISTIC — derived from quadratic scoring break-even math.
    # Per-level score = (human_baseline_actions / agent_actions)², capped 1.15×.
    # Marginal score lift goes to zero as agent_actions → 2 × human_baseline.
    # Below that, additional actions return more than they cost in budget.
    # Above, they cost more than they return. 25 is a workable floor that
    # absorbs cases where the per-game human baseline is unavailable.
    # GEMINI AUDIT NOTE: Gemini flagged this as "overt data leak" but it is
    # NOT tuned on public games — it is a constant derived from the published
    # scoring formula. We defend the (b) classification but acknowledge the
    # specific scalar (25) is designer-chosen and disclose it as such.
    min_actions_before_abstain: int = 25
    # FORM: (a) — the high-variance signal indicates wrong hypothesis class.
    # SCALAR (0.20): (c) DESIGNER CHOICE — the variance threshold itself.
    world_model_variance_threshold: float = 0.20
    # FORM: (a) — slope-near-zero indicates novelty plateau.
    # SCALAR (0.05): (c) DESIGNER CHOICE — the plateau slope cutoff.
    novelty_plateau_slope: float = 0.05


@dataclass(frozen=True)
class MCTSConfig:
    # FORM: (a) — PUCT formula is canonical AlphaZero.
    # SCALAR (1.41): (b) BUDGET HEURISTIC — published-canonical c_puct value
    #                ≈ √2, widely adopted.
    c_puct: float = 1.41
    # (b) BUDGET HEURISTIC — per-action search depth cap.
    max_depth: int = 6
    # (b) BUDGET HEURISTIC — rollouts per real action; trades quality vs wall clock.
    rollouts_per_action: int = 200
    # (b) BUDGET HEURISTIC — hard timeout per choose_action call (ms).
    hard_timeout_ms: int = 500
    # FORM: (a) — sparse positive reward at WIN, small per-action cost, small
    #             novelty bonus is standard MCTS shaping (Pathak ICM lineage).
    # SCALARS (+10 / −0.01 / +0.10): (c) DESIGNER CHOICE — author-set scalars
    #                                 frozen pre-eval. Per Gemini audit, the
    #                                 specific magnitudes are not derived.
    reward_win: float = 10.0
    reward_per_action: float = -0.01
    reward_novel_fingerprint: float = 0.10


@dataclass(frozen=True)
class BudgetConfig:
    # (b) BUDGET HEURISTIC — Kaggle's 9h hard cap minus 5min cold-start
    #     overhead minus 5min safety.
    wall_clock_kill_seconds: int = 8 * 3600 + 50 * 60   # 8h50m
    # FORM: (a) — the agents.agent.Agent framework default is 80 actions.
    # SCALAR (80): admissible as the framework default we adopt; verified
    #              empirically that the server does not enforce a tighter cap.
    max_actions_per_game: int = 80
    # (b) BUDGET HEURISTIC — anticipated 110 private games per Kaggle eval batch.
    expected_total_games: int = 110


@dataclass(frozen=True)
class FingerprintConfig:
    # SCALAR (50): (c) DESIGNER CHOICE — total signature dimensionality.
    # Per-component bin counts (16/8/4) are also designer choices documented
    # in src/misfit_agent/fingerprint.py. Gemini audit: any data-driven bin
    # selection would move these to (a) DERIVED FROM PRIOR; we have not yet
    # implemented data-driven binning so they remain (c) for now.
    total_dim: int = 50


@dataclass(frozen=True)
class FrozenConfig:
    """Singleton view over all module configs.

    Constants-frozen tag policy:
      1. Tag the prior submission's git SHA before changing any (c) scalar
      2. Document the change in CHANGELOG.md with rationale
      3. Note which judge / reviewer finding justified it
      4. Re-run the full pytest suite before next push
    """
    tracker: TrackerConfig = TrackerConfig()
    world_model: WorldModelConfig = WorldModelConfig()
    pseudocount: PseudocountConfig = PseudocountConfig()
    abstain: AbstainConfig = AbstainConfig()
    mcts: MCTSConfig = MCTSConfig()
    budget: BudgetConfig = BudgetConfig()
    fingerprint: FingerprintConfig = FingerprintConfig()


CONFIG = FrozenConfig()
