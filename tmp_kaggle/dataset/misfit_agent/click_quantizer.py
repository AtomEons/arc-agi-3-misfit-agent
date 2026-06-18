"""ClickQuantizer — collapse the ACTION6 (x,y) coordinate space.

The judges' architecture call-out: ACTION6 has a 64×64 = 4096-cell click
space. A random click has 1/4096 odds of landing on a meaningful target.
Quantizing to object-centroid + bbox-corner + edge-midpoint candidates
typically yields 5-20 candidates per frame — a **200-400× search-
efficiency win** with no Tier-2 contamination (all candidates derive from
the Spelke objectness prior applied to the in-context observation).

Per the priors audit, ALL candidate generation must be derived from
observed geometry — no hardcoded cell coordinates, no game-specific
"player is usually at row 32" priors. The 9-quadrant fallback is generic
geometric coverage when zero objects are detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .perceptor import SceneObservation


GRID_MAX_X = 63
GRID_MAX_Y = 63


@dataclass(frozen=True)
class ClickCandidate:
    x: int
    y: int
    rationale: str    # priors-only rationale string (no game-specific text)
    source: str       # "centroid" | "bbox_corner" | "edge_midpoint" | "quadrant_fallback"


def _clip(v: int, lo: int = 0, hi: int = GRID_MAX_X) -> int:
    return max(lo, min(hi, v))


def click_candidates(scene: SceneObservation, max_candidates: int = 20
                     ) -> list[ClickCandidate]:
    """Generate the priors-only ACTION6 candidate set for the given scene.

    Order: largest-object centroid first, then descending area for the
    remaining objects, then bbox corners of the top-3, then 9 quadrant
    fallbacks. Deduplicated by (x, y).
    """
    seen: set[tuple[int, int]] = set()
    out: list[ClickCandidate] = []

    # 1. Centroids of all detected objects (already area-sorted descending).
    for i, obj in enumerate(scene.objects):
        cr, cc = obj.centroid
        x, y = _clip(int(round(cc))), _clip(int(round(cr)))
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        out.append(ClickCandidate(
            x=x, y=y,
            rationale=f"centroid of object rank {i} (color={obj.color}, area={obj.area})",
            source="centroid",
        ))
        if len(out) >= max_candidates:
            return out

    # 2. BBox corners of the top-3 objects — captures grab handles and
    #    "destination" cells common in object-puzzle games.
    for i, obj in enumerate(scene.objects[:3]):
        r0, c0, r1, c1 = obj.bbox
        for (yy, xx, corner) in [(r0, c0, "TL"), (r0, c1, "TR"),
                                  (r1, c0, "BL"), (r1, c1, "BR")]:
            x, y = _clip(int(xx)), _clip(int(yy))
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            out.append(ClickCandidate(
                x=x, y=y,
                rationale=f"bbox {corner} of object rank {i}",
                source="bbox_corner",
            ))
            if len(out) >= max_candidates:
                return out

    # 3. Edge midpoints — captures "edge target" patterns the perceptor
    #    flagged via touches_edge but the centroid missed.
    for i, obj in enumerate(scene.objects[:3]):
        if not obj.touches_edge:
            continue
        r0, c0, r1, c1 = obj.bbox
        for (yy, xx, side) in [
            (r0, (c0 + c1) // 2, "top"),
            (r1, (c0 + c1) // 2, "bottom"),
            ((r0 + r1) // 2, c0, "left"),
            ((r0 + r1) // 2, c1, "right"),
        ]:
            x, y = _clip(int(xx)), _clip(int(yy))
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            out.append(ClickCandidate(
                x=x, y=y,
                rationale=f"edge midpoint ({side}) of edge-touching object rank {i}",
                source="edge_midpoint",
            ))
            if len(out) >= max_candidates:
                return out

    # 4. 9-quadrant geometric fallback — generic coverage when objectness
    #    is empty (e.g. blank-grid start state). Pure geometry prior.
    if scene.rows > 0 and scene.cols > 0:
        max_r = scene.rows - 1
        max_c = scene.cols - 1
        grid_pts = [
            (max_r * a // 2, max_c * b // 2)
            for a in (0, 1, 2)
            for b in (0, 1, 2)
        ]
        for (yy, xx) in grid_pts:
            x, y = _clip(int(xx)), _clip(int(yy))
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            out.append(ClickCandidate(
                x=x, y=y,
                rationale="9-quadrant geometric fallback",
                source="quadrant_fallback",
            ))
            if len(out) >= max_candidates:
                return out

    return out


def best_click_candidate(scene: SceneObservation,
                         policy_seeds_xy: Optional[list[tuple[int, int]]] = None
                         ) -> ClickCandidate:
    """Pick the single best candidate. Bias toward seed-aligned candidates
    if the resonance library provided any (x, y) hints from prior winning
    policies, else return the top of `click_candidates`.
    """
    candidates = click_candidates(scene)
    if not candidates:
        # Truly empty — center the grid.
        cx = (scene.cols - 1) // 2 if scene.cols else 0
        cy = (scene.rows - 1) // 2 if scene.rows else 0
        return ClickCandidate(
            x=_clip(cx), y=_clip(cy),
            rationale="empty scene; center-of-grid fallback",
            source="empty_fallback",
        )

    if policy_seeds_xy:
        # Pick the candidate closest to ANY seed coordinate.
        best = candidates[0]
        best_dist = float("inf")
        for c in candidates:
            for sx, sy in policy_seeds_xy:
                d = (c.x - sx) ** 2 + (c.y - sy) ** 2
                if d < best_dist:
                    best_dist = d
                    best = c
        return best

    return candidates[0]
