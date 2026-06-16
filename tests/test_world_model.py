"""World model + rule templates smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from misfit_agent.rules.translate import Translate
from misfit_agent.rules.no_op import NoOp
from misfit_agent.world_model import WorldModel


def _obs(action_name, pre_centroids_by_class, post_centroids_by_class):
    """Build a fake observation dict used by Rule.fit / WorldModel.fit."""
    classes = set(pre_centroids_by_class.keys()) | set(post_centroids_by_class.keys())
    return {
        "action_name": action_name,
        "classes_involved": list(classes),
        "pre_objects_of_class": [
            {"centroid": c, "area": 1} for c in pre_centroids_by_class.get(
                next(iter(pre_centroids_by_class), 0), []
            )
        ],
        "post_objects_of_class": [
            {"centroid": c, "area": 1} for c in post_centroids_by_class.get(
                next(iter(post_centroids_by_class), 0), []
            )
        ],
    }


def test_translate_rule_recovers_constant_shift_under_action1():
    rule = Translate(object_class=2)
    observations = []
    # Object class 2 always shifts down by 1 row under ACTION1
    for r in range(5):
        observations.append({
            "action_name": "ACTION1",
            "pre_objects_of_class": [{"centroid": (float(r), 3.0), "area": 1}],
            "post_objects_of_class": [{"centroid": (float(r + 1), 3.0), "area": 1}],
        })
    assert rule.fit(observations) is True
    assert rule.dy_per_action["ACTION1"] == 1
    assert rule.dx_per_action["ACTION1"] == 0


def test_translate_rule_rejects_inconsistent_shifts():
    rule = Translate(object_class=2)
    observations = [
        {"action_name": "ACTION1",
         "pre_objects_of_class": [{"centroid": (0.0, 0.0), "area": 1}],
         "post_objects_of_class": [{"centroid": (1.0, 0.0), "area": 1}]},
        {"action_name": "ACTION1",
         "pre_objects_of_class": [{"centroid": (0.0, 0.0), "area": 1}],
         "post_objects_of_class": [{"centroid": (5.0, 0.0), "area": 1}]},
        {"action_name": "ACTION1",
         "pre_objects_of_class": [{"centroid": (0.0, 0.0), "area": 1}],
         "post_objects_of_class": [{"centroid": (10.0, 0.0), "area": 1}]},
    ]
    fitted = rule.fit(observations)
    # All three deltas are wildly different — consistency below threshold
    assert rule.consistency_score < 0.8
    assert fitted is False


def test_translate_predict_shifts_grid_cells():
    rule = Translate(
        object_class=2,
        dx_per_action={"ACTION1": 0},
        dy_per_action={"ACTION1": 1},
    )
    grid = np.array([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    out = rule.predict(grid, "ACTION1")
    # The 2 should have moved from (1,1) to (2,1)
    assert out[2, 1] == 2
    assert out[1, 1] == 0


def test_noop_rule_fits_unchanged_observations():
    rule = NoOp(object_class=3)
    observations = [
        {"pre_objects_of_class": [{"centroid": (1.0, 1.0), "area": 4}],
         "post_objects_of_class": [{"centroid": (1.0, 1.0), "area": 4}]}
        for _ in range(3)
    ]
    assert rule.fit(observations) is True


def test_world_model_predicts_with_low_confidence_when_unseen():
    wm = WorldModel()
    grid = np.zeros((4, 4), dtype=np.int32)
    pred, conf = wm.predict(grid, "ACTION1")
    # No observations yet — predict returns original with confidence 0.0
    assert np.array_equal(pred, grid)
    assert conf == 0.0


def test_world_model_predict_reset_returns_low_confidence():
    wm = WorldModel()
    grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    pred, conf = wm.predict(grid, "RESET")
    assert conf == 0.0
    assert np.array_equal(pred, grid)


def test_world_model_coverage_zero_when_empty():
    wm = WorldModel()
    assert wm.coverage() == 0.0


# --- Outer refinement loop tests (HRM hidden-driver) ---


def _translate_obs(action_name, cls, pre_centroid, post_centroid):
    return {
        "action_name": action_name,
        "classes_involved": [cls],
        "pre_objects_of_class": [{"centroid": pre_centroid, "area": 1}],
        "post_objects_of_class": [{"centroid": post_centroid, "area": 1}],
    }


def test_fit_with_refinement_returns_scores_and_increments_counter():
    wm = WorldModel()
    obs = [
        _translate_obs("ACTION1", 2, (float(r), 3.0), (float(r + 1), 3.0))
        for r in range(5)
    ]
    scores = wm.fit_with_refinement(obs, max_iters=4)
    assert isinstance(scores, dict)
    assert wm.refinement_iterations_total >= 1
    # Score history populated, never decreasing in the converged case
    assert len(wm.last_fit_score_history) >= 1


def test_fit_with_refinement_early_stops_when_no_improvement():
    """A trivially-fittable observation set should stop refining quickly."""
    wm = WorldModel()
    obs = [
        _translate_obs("ACTION1", 2, (float(r), 0.0), (float(r + 1), 0.0))
        for r in range(8)
    ]
    wm.fit_with_refinement(obs, max_iters=10, improvement_threshold=0.001)
    # On a clean fit, refinement should not burn all 10 iterations
    assert wm.refinement_iterations_total < 10


def test_fit_with_refinement_max_iters_caps_at_8_per_hrm_analysis():
    """HRM analysis showed diminishing returns past ~8 iterations.
    Our default of 4 reflects the cheap-end of that curve.
    """
    wm = WorldModel()
    # Inconsistent observations — rules will fail to converge cleanly
    obs = [
        _translate_obs("ACTION1", 2, (0.0, 0.0), (1.0, 0.0)),
        _translate_obs("ACTION1", 2, (0.0, 0.0), (5.0, 0.0)),
        _translate_obs("ACTION1", 2, (0.0, 0.0), (9.0, 0.0)),
    ]
    wm.fit_with_refinement(obs, max_iters=8)
    assert wm.refinement_iterations_total <= 8


def test_prune_contradicting_rules_drops_bad_noop():
    """A NoOp rule should not survive when an observation shows the
    object moved. _prune_contradicting_rules is the feedback signal
    that makes refinement improve coverage instead of just re-fitting.
    """
    wm = WorldModel()
    wm.rules = [NoOp(object_class=2)]
    contradicting_obs = [{
        "action_name": "ACTION1",
        "classes_involved": [2],
        "pre_objects_of_class": [{"centroid": (0.0, 0.0), "area": 1}],
        "post_objects_of_class": [{"centroid": (1.0, 0.0), "area": 1}],
    }]
    wm._prune_contradicting_rules(contradicting_obs)
    assert wm.rules == []  # NoOp dropped: object actually moved
