"""Episode fingerprint — 50-dim signature derived from in-context observations.

Each dimension must be computable from the Spelke priors alone, applied to the
agent's own observation history. No reference to ARC task families, no
hand-picked statistics tuned on the public eval.

Dimensions:
   0  mean rows / mean cols across observed scenes (aspect proxy)
   1  fraction of scenes with shape-preserving transitions
   2  mean object count per scene
   3  delta object count (first vs last scene observed)
   4  fraction of objects with vertical symmetry (mean)
   5  fraction of objects with horizontal symmetry (mean)
   6  fraction of objects touching edge (mean)
   7  mean largest-object area / total cells (mean)
   8  mean foreground-cell ratio
   9  total observed scenes (log-bucketed)
  10-19  palette density per color 0..9 (mean over scenes)
  20-29  palette delta (last - first) per color 0..9
  30-37  action-effect signature (cells_changed mean per ACTION1..ACTION7+RESET)
  38-45  action level-advance rate per ACTION1..ACTION7+RESET
  46-49  reserved / zeros (kept for forward compatibility)
"""

from __future__ import annotations

import math

import numpy as np

from .episode import EpisodeTracker
from .perceptor import SceneObservation


FINGERPRINT_DIM = 50


def fingerprint_episode(tracker: EpisodeTracker) -> np.ndarray:
    v = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    scenes = tracker.scenes
    if not scenes:
        return v

    rows = [s.rows for s in scenes]
    cols = [s.cols for s in scenes]
    v[0] = float(np.mean(rows)) / max(float(np.mean(cols)), 1.0)

    same_shape = sum(1 for i in range(1, len(scenes))
                     if scenes[i].grid.shape == scenes[i-1].grid.shape)
    v[1] = same_shape / max(len(scenes) - 1, 1)

    obj_counts = [len(s.objects) for s in scenes]
    v[2] = float(np.mean(obj_counts)) if obj_counts else 0.0
    v[3] = float(obj_counts[-1] - obj_counts[0]) if len(obj_counts) > 1 else 0.0

    def _mean_safe(arr): return float(np.mean(arr)) if arr else 0.0
    v[4] = _mean_safe([sum(o.v_symmetric for o in s.objects) / max(len(s.objects), 1)
                       for s in scenes])
    v[5] = _mean_safe([sum(o.h_symmetric for o in s.objects) / max(len(s.objects), 1)
                       for s in scenes])
    v[6] = _mean_safe([sum(o.touches_edge for o in s.objects) / max(len(s.objects), 1)
                       for s in scenes])
    v[7] = _mean_safe([(s.objects[0].area / max(s.rows*s.cols, 1)) if s.objects else 0.0
                       for s in scenes])
    v[8] = _mean_safe([s.foreground_cells / max(s.rows*s.cols, 1) for s in scenes])
    v[9] = float(math.log(1 + len(scenes)))

    # Palette density per color 0..9 (mean over scenes)
    palette_means = np.zeros(10, dtype=np.float32)
    for s in scenes:
        denom = max(s.rows * s.cols, 1)
        for c in range(10):
            palette_means[c] += s.color_histogram[c] / denom
    palette_means /= max(len(scenes), 1)
    v[10:20] = palette_means

    # Palette delta — last vs first scene
    first_p = np.array(scenes[0].color_histogram, dtype=np.float32) / max(
        scenes[0].rows * scenes[0].cols, 1)
    last_p = np.array(scenes[-1].color_histogram, dtype=np.float32) / max(
        scenes[-1].rows * scenes[-1].cols, 1)
    v[20:30] = (last_p - first_p)

    # Action-effect signature & level-advance rate per action enum slot.
    action_slots = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
                    "ACTION5", "ACTION6", "ACTION7"]
    for i, name in enumerate(action_slots):
        bucket = tracker.transition_signals.get(name)
        if not bucket or bucket["total"] == 0:
            continue
        v[30 + i] = bucket["cells_changed_sum"] / bucket["total"]
        v[38 + i] = bucket["level_advances"] / bucket["total"]

    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
