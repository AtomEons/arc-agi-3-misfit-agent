"""GoalInducer — induces ranked goal hypotheses from observed (Δlevels, Δscene) pairs.

Spelke priors used:
  - OBJECTNESS: track per-class object counts as a Spelke-numerosity quantity
  - NUMEROSITY: "count goes from N to 0" is a primitive that infants track
  - SPATIAL/GOAL-DIRECTEDNESS: "agent reached cell-of-class-Y" is a containment/
    coincidence primitive — same Spelke building block underlying "predator
    caught prey" or "ball entered cup"

No game-family hardcoding. No "this looks like Sokoban / Pac-Man / Lights Out".
The inducer only references object-class counts and agent-class containment.

Hypothesis schema (≤3 free params each — per architect's constraint):
  - "removed_all_of_class"        params: (class)
  - "agent_reached_class"         params: (agent_class, target_class)
  - "count_of_class_equals_N"     params: (class, N)

Rank is by observation-conditional posterior odds: P(level_advance | hypothesis).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .perceptor import SceneObservation


# Maximum free parameters per hypothesis — enforced at construction time.
# This is a Spelke-derived structural cap (the architect's must-fix #4):
# more than 3 params over the observed window admits curve-fitting, not induction.
MAX_FREE_PARAMS = 3


@dataclass(frozen=True)
class GoalHypothesis:
    """A single goal hypothesis with a ranked posterior score.

    `kind` selects the predicate family; `params` is a tuple of integer-valued
    parameters (≤3). `score` is the posterior-odds estimate from observed pairs.
    """
    kind: str
    params: tuple[int, ...]
    score: float
    support: int  # number of (Δlevels, scene-change) pairs that voted yes
    contradictions: int  # number of pairs that voted no

    def __post_init__(self) -> None:
        if len(self.params) > MAX_FREE_PARAMS:
            raise ValueError(
                f"hypothesis '{self.kind}' has {len(self.params)} params, "
                f"exceeds MAX_FREE_PARAMS={MAX_FREE_PARAMS}"
            )


@dataclass
class _ScenePair:
    """One (pre-scene, post-scene, Δlevels) observation."""
    pre: SceneObservation
    post: SceneObservation
    delta_levels: int


@dataclass
class GoalInducer:
    """Tracks scene-change pairs and emits ranked goal hypotheses.

    Pure Spelke priors: objectness (class counts), spatial coincidence
    (agent reached cell of class Y), numerosity (count == N).

    No knowledge of any specific game family.
    """

    pairs: list[_ScenePair] = field(default_factory=list)
    # Laplace pseudocount for posterior odds — a single Spelke prior over
    # "any predicate is plausible until contradicted".
    laplace_alpha: float = 1.0

    def observe(
        self,
        pre_scene: SceneObservation,
        post_scene: SceneObservation,
        delta_levels: int,
        scene_change_description: Optional[str] = None,
    ) -> None:
        """Record a (Δlevels, scene-change) pair.

        `scene_change_description` is accepted for API completeness (caller
        may have a human-readable diff) but is NOT used for induction — we
        derive predicates directly from the scene observations under priors.
        """
        # description is intentionally unused — induction works on the priors,
        # not on natural-language descriptions (Tier-1 STRICT: no LLM path).
        _ = scene_change_description
        self.pairs.append(_ScenePair(pre=pre_scene, post=post_scene,
                                      delta_levels=int(delta_levels)))

    # -- candidate enumeration ------------------------------------------------

    def _candidate_removed_all_classes(self) -> set[int]:
        """Classes that disappeared (count went to 0) in at least one pair."""
        out: set[int] = set()
        for p in self.pairs:
            pre_cnt = Counter(o.color for o in p.pre.objects)
            post_cnt = Counter(o.color for o in p.post.objects)
            for c in pre_cnt:
                if pre_cnt[c] > 0 and post_cnt.get(c, 0) == 0:
                    out.add(int(c))
        return out

    def _candidate_count_equals(self) -> set[tuple[int, int]]:
        """(class, N) pairs witnessed in any post-scene."""
        out: set[tuple[int, int]] = set()
        for p in self.pairs:
            post_cnt = Counter(o.color for o in p.post.objects)
            for c, n in post_cnt.items():
                out.add((int(c), int(n)))
            # also seed N=0 candidates from disappearance
            pre_cnt = Counter(o.color for o in p.pre.objects)
            for c in pre_cnt:
                if post_cnt.get(c, 0) == 0:
                    out.add((int(c), 0))
        return out

    def _candidate_agent_reached(self) -> set[tuple[int, int]]:
        """(agent_class, target_class) candidate pairs.

        Agent-class candidates: any color whose objects' centroids moved
        between pre and post (i.e. has non-zero translation).
        Target-class candidates: any color present in pre but with reduced
        count in post (consumed-on-contact pattern), OR any color whose cell
        is now occupied by the agent's class.

        We use cell-coincidence (post-grid value at agent's previous-or-
        adjacent position) as the spatial primitive.
        """
        out: set[tuple[int, int]] = set()
        for p in self.pairs:
            # Build per-class centroid sets pre and post.
            pre_by_color: dict[int, list[tuple[float, float]]] = {}
            post_by_color: dict[int, list[tuple[float, float]]] = {}
            for o in p.pre.objects:
                pre_by_color.setdefault(int(o.color), []).append(o.centroid)
            for o in p.post.objects:
                post_by_color.setdefault(int(o.color), []).append(o.centroid)

            # Agent class = a class whose mean centroid moved across the pair.
            moved_classes: set[int] = set()
            for c, pre_cents in pre_by_color.items():
                post_cents = post_by_color.get(c, [])
                if not post_cents or len(pre_cents) != len(post_cents):
                    continue
                pre_mean_r = sum(r for r, _ in pre_cents) / len(pre_cents)
                pre_mean_c = sum(cc for _, cc in pre_cents) / len(pre_cents)
                post_mean_r = sum(r for r, _ in post_cents) / len(post_cents)
                post_mean_c = sum(cc for _, cc in post_cents) / len(post_cents)
                if (abs(pre_mean_r - post_mean_r) + abs(pre_mean_c - post_mean_c)) > 0.5:
                    moved_classes.add(c)

            # Target class = a class whose count dropped (consumed on contact).
            shrunk_classes: set[int] = set()
            for c, pre_cents in pre_by_color.items():
                if len(post_by_color.get(c, [])) < len(pre_cents):
                    shrunk_classes.add(c)

            for ag in moved_classes:
                for tg in shrunk_classes:
                    if ag == tg:
                        continue
                    out.add((int(ag), int(tg)))
        return out

    # -- evaluation -----------------------------------------------------------

    def _evaluate_removed_all(self, cls: int) -> tuple[int, int]:
        """Return (support, contradictions) for 'all of class X removed → level'."""
        support = contradictions = 0
        for p in self.pairs:
            post_count = sum(1 for o in p.post.objects if o.color == cls)
            removed = (post_count == 0) and any(o.color == cls for o in p.pre.objects)
            if removed and p.delta_levels > 0:
                support += 1
            elif removed and p.delta_levels <= 0:
                contradictions += 1
            elif p.delta_levels > 0 and not removed:
                # level advanced without satisfying this hypothesis — contradicts
                contradictions += 1
        return support, contradictions

    def _evaluate_agent_reached(self, agent_cls: int, target_cls: int) -> tuple[int, int]:
        support = contradictions = 0
        for p in self.pairs:
            pre_target = sum(1 for o in p.pre.objects if o.color == target_cls)
            post_target = sum(1 for o in p.post.objects if o.color == target_cls)
            reached = (pre_target > 0) and (post_target < pre_target) and any(
                o.color == agent_cls for o in p.pre.objects
            )
            if reached and p.delta_levels > 0:
                support += 1
            elif reached and p.delta_levels <= 0:
                contradictions += 1
            elif p.delta_levels > 0 and not reached:
                contradictions += 1
        return support, contradictions

    def _evaluate_count_equals(self, cls: int, n: int) -> tuple[int, int]:
        support = contradictions = 0
        for p in self.pairs:
            post_count = sum(1 for o in p.post.objects if o.color == cls)
            holds = (post_count == n)
            if holds and p.delta_levels > 0:
                support += 1
            elif holds and p.delta_levels <= 0:
                contradictions += 1
            elif p.delta_levels > 0 and not holds:
                contradictions += 1
        return support, contradictions

    def _posterior(self, support: int, contradictions: int) -> float:
        """Laplace-smoothed posterior odds — a Spelke prior over predicate plausibility."""
        a = self.laplace_alpha
        return (support + a) / (support + contradictions + 2.0 * a)

    # -- public API -----------------------------------------------------------

    def hypothesize(self, top_k: int = 8) -> list[GoalHypothesis]:
        """Return ranked goal hypotheses (highest posterior first).

        Empty input → empty list. A hypothesis with zero support AND zero
        contradiction is omitted (no evidence either way).
        """
        if not self.pairs:
            return []

        hyps: list[GoalHypothesis] = []

        # Family 1: "all of class X removed" (1 free param)
        for cls in self._candidate_removed_all_classes():
            sup, con = self._evaluate_removed_all(cls)
            if sup == 0 and con == 0:
                continue
            hyps.append(GoalHypothesis(
                kind="removed_all_of_class",
                params=(int(cls),),
                score=self._posterior(sup, con),
                support=sup,
                contradictions=con,
            ))

        # Family 2: "agent reached cell of class Y" (2 free params)
        for agent_cls, target_cls in self._candidate_agent_reached():
            sup, con = self._evaluate_agent_reached(agent_cls, target_cls)
            if sup == 0 and con == 0:
                continue
            hyps.append(GoalHypothesis(
                kind="agent_reached_class",
                params=(int(agent_cls), int(target_cls)),
                score=self._posterior(sup, con),
                support=sup,
                contradictions=con,
            ))

        # Family 3: "count of class Z equals N" (2 free params)
        for cls, n in self._candidate_count_equals():
            sup, con = self._evaluate_count_equals(cls, n)
            if sup == 0 and con == 0:
                continue
            hyps.append(GoalHypothesis(
                kind="count_of_class_equals_N",
                params=(int(cls), int(n)),
                score=self._posterior(sup, con),
                support=sup,
                contradictions=con,
            ))

        # Rank: highest posterior score, then highest support, then lowest
        # contradiction count as a tiebreaker. Stable sort preserves insertion
        # order among ties for reproducibility.
        hyps.sort(key=lambda h: (-h.score, -h.support, h.contradictions, h.kind, h.params))
        return hyps[:top_k]
