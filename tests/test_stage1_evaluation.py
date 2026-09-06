import numpy as np
import pytest

from sfm_reconstruction.stage1_evaluation import evaluate_stage1_points


def test_stage1_point_metrics_match_supplied_chamfer_definition():
    ground_truth = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    estimated = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    metrics = evaluate_stage1_points(ground_truth, estimated)

    assert metrics.mean_ground_truth_to_estimate == pytest.approx(0.5)
    assert metrics.mean_estimate_to_ground_truth == pytest.approx(0.5)
    assert metrics.chamfer_distance == pytest.approx(1.0)
