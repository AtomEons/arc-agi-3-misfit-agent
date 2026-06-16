"""HungarianTracker — object persistence across two scenes via assignment.

Spelke priors used:
  - COHESION: an object is a coherent region of cells; tracked as a unit.
  - CONTINUITY: an object that exists at t persists into t+1 unless destroyed.
  - SOLIDITY: two distinct objects do not occupy the same cell.

Cost function (≤3 weights per architect's constraint):
    cost(i, j) = alpha * centroid_distance(i, j)
               + beta  * shape_hamming(i, j)
               + gamma * color_mismatch(i, j)

Weights default to CONFIG.tracker (alpha=1.0, beta=0.5, gamma=2.0).

Solver:
  - Preferred: scipy.optimize.linear_sum_assignment (Kuhn-Munkres, O(n^3)).
  - Fallback:  pure-numpy greedy nearest-neighbor (O(n^2 log n)).
    The fallback is admitted because Kaggle's eval environment ships scipy,
    but local CI sometimes runs in a minimal venv. The greedy variant is
    near-optimal for sparse correspondences (most scenes have <20 objects)
    and is documented as a fallback in the test suite.

Birth/death:
  - Unmatched current-scene objects → spawned (born).
  - Unmatched previous-scene objects → destroyed.
  - A match whose cost exceeds the gating threshold is REJECTED — we'd
    rather call something destroyed-and-spawned than force a bad match.

Idempotence:
  - track(prev, curr) has no side effects on its arguments and no global
    state. Calling it twice on the same inputs yields identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CONFIG
from .perceptor import Obj, SceneObservation


# Try scipy; fall back to greedy nearest-neighbor.
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - environment-dependent
    _scipy_lsa = None
    _HAS_SCIPY = False


# Gating: a match with cost above this is rejected (treated as birth+death).
# (a) DERIVED FROM PRIOR — under cohesion, an object that traveled more than
# half the grid in one step with no color persistence is more likely two
# distinct objects than a single tracked one. The gating constant is large
# enough to admit reasonable motion but small enough to reject identity
# collisions between obviously different objects.
GATING_COST = 50.0


@dataclass(frozen=True)
class TrackingCosts:
    alpha: float
    beta: float
    gamma: float


def _shape_hamming(a: Obj, b: Obj) -> float:
    """Per-cell binary-mask hamming distance, normalized by max bbox area.

    Each object is rasterized to its bbox, padded to the union size, and
    XORed. Returns a value in [0, 1].
    """
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    ah, aw = ar1 - ar0 + 1, ac1 - ac0 + 1
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
    h = max(ah, bh)
    w = max(aw, bw)
    mask_a = np.zeros((h, w), dtype=np.int8)
    mask_b = np.zeros((h, w), dtype=np.int8)
    # Center each mask in the union box for translation-invariant shape compare.
    off_a_r = (h - ah) // 2
    off_a_c = (w - aw) // 2
    off_b_r = (h - bh) // 2
    off_b_c = (w - bw) // 2
    mask_a[off_a_r:off_a_r + ah, off_a_c:off_a_c + aw] = 1
    mask_b[off_b_r:off_b_r + bh, off_b_c:off_b_c + bw] = 1
    diff = int(np.sum(mask_a ^ mask_b))
    return diff / float(h * w) if h * w else 0.0


def _centroid_dist(a: Obj, b: Obj) -> float:
    ar, ac = a.centroid
    br, bc = b.centroid
    return float(np.hypot(ar - br, ac - bc))


def _color_mismatch(a: Obj, b: Obj) -> float:
    return 0.0 if a.color == b.color else 1.0


def _pairwise_cost(prev: list[Obj], curr: list[Obj], w: TrackingCosts) -> np.ndarray:
    """Build the (len(prev), len(curr)) cost matrix."""
    m, n = len(prev), len(curr)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float64)
    cost = np.zeros((m, n), dtype=np.float64)
    for i, p in enumerate(prev):
        for j, c in enumerate(curr):
            cost[i, j] = (
                w.alpha * _centroid_dist(p, c)
                + w.beta * _shape_hamming(p, c)
                + w.gamma * _color_mismatch(p, c)
            )
    return cost


def _greedy_assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy greedy nearest-neighbor fallback.

    Repeatedly pick the lowest-cost (i, j) pair, mark both used, repeat.
    Returns (row_ind, col_ind) in the same shape contract as scipy LSA
    (one array each, parallel indexing).

    Not optimal in general but tight for sparse correspondences with
    well-separated costs — which is the typical Spelke scene.
    """
    m, n = cost.shape
    if m == 0 or n == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    row_used = np.zeros(m, dtype=bool)
    col_used = np.zeros(n, dtype=bool)
    pairs: list[tuple[int, int]] = []
    # Sort all pairs by cost ascending, then greedy-fill.
    flat_idx = np.argsort(cost, axis=None, kind="stable")
    for fi in flat_idx:
        i, j = int(fi // n), int(fi % n)
        if row_used[i] or col_used[j]:
            continue
        pairs.append((i, j))
        row_used[i] = True
        col_used[j] = True
        if len(pairs) == min(m, n):
            break
    if not pairs:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    ri = np.array([p[0] for p in pairs], dtype=np.intp)
    ci = np.array([p[1] for p in pairs], dtype=np.intp)
    return ri, ci


@dataclass
class HungarianTracker:
    """Stateless object-correspondence tracker between consecutive scenes.

    `costs` is read once at construction; if not supplied, pulled from
    CONFIG.tracker (frozen weights with provenance in config.py).
    """

    costs: Optional[TrackingCosts] = None
    gating_cost: float = GATING_COST
    # When True, prefer the greedy fallback even if scipy is available.
    # Useful for tests that exercise the fallback path on machines with scipy.
    force_greedy: bool = False

    def __post_init__(self) -> None:
        if self.costs is None:
            self.costs = TrackingCosts(
                alpha=CONFIG.tracker.alpha_centroid_dist,
                beta=CONFIG.tracker.beta_shape_hamming,
                gamma=CONFIG.tracker.gamma_color_mismatch,
            )

    def track(
        self,
        prev_scene: SceneObservation,
        curr_scene: SceneObservation,
    ) -> dict[int, Optional[int]]:
        """Match objects from prev_scene to curr_scene.

        Returns:
          {prev_obj_idx -> curr_obj_idx | None}
          A None value means the object was destroyed (no successor).
          Spawned objects (in curr but unmatched) are NOT in the dict;
          callers can derive them as `set(range(len(curr.objects))) - set(values)`.

        Idempotent: no mutation of inputs, no global state, deterministic.
        """
        prev_objs = list(prev_scene.objects)
        curr_objs = list(curr_scene.objects)

        # Edge cases.
        if not prev_objs:
            return {}
        if not curr_objs:
            return {i: None for i in range(len(prev_objs))}

        cost = _pairwise_cost(prev_objs, curr_objs, self.costs)

        # Choose solver.
        if _HAS_SCIPY and not self.force_greedy:
            # scipy returns assignment over min(m, n) pairs.
            row_ind, col_ind = _scipy_lsa(cost)
        else:
            row_ind, col_ind = _greedy_assign(cost)

        # Apply gating: reject matches whose cost is above threshold.
        result: dict[int, Optional[int]] = {i: None for i in range(len(prev_objs))}
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] <= self.gating_cost:
                result[int(ri)] = int(ci)
            # else: leave as None (destroyed, even if scipy "matched" them)
        return result

    def spawned_indices(
        self,
        prev_scene: SceneObservation,
        curr_scene: SceneObservation,
        mapping: dict[int, Optional[int]],
    ) -> list[int]:
        """Indices in curr_scene.objects that were not matched (spawned)."""
        matched = {v for v in mapping.values() if v is not None}
        return [j for j in range(len(curr_scene.objects)) if j not in matched]

    def destroyed_indices(
        self,
        mapping: dict[int, Optional[int]],
    ) -> list[int]:
        """Indices in prev_scene.objects that have no successor."""
        return [i for i, v in mapping.items() if v is None]
