import numpy as np
import pytest

from sfm_reconstruction.stage2_evaluation import evaluate_mesh_proxy


def test_evaluate_mesh_proxy_scales_crops_and_summarizes_distances():
    mesh = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    estimated = np.asarray([[0.2, 0.0, 0.0], [2.2, 0.0, 0.0], [8.0, 0.0, 0.0]])
    metrics = evaluate_mesh_proxy(
        mesh,
        estimated,
        scale=2.0,
        bounds=np.asarray([[-1.0, -1.0, -1.0], [2.0, 1.0, 1.0]]),
    )

    assert metrics.input_points == 3
    assert metrics.cropped_points == 2
    assert metrics.mean_nearest_vertex_distance == pytest.approx(3.2 / 3.0)
    assert metrics.median_nearest_vertex_distance == pytest.approx(0.1)
    assert metrics.fraction_within_005 == 0.0
    assert metrics.cropped_mean_nearest_vertex_distance == pytest.approx(0.1)
    assert metrics.cropped_median_nearest_vertex_distance == pytest.approx(0.1)


def test_evaluate_mesh_proxy_rejects_empty_crop():
    with pytest.raises(ValueError, match="no estimated points"):
        evaluate_mesh_proxy(
            np.zeros((1, 3)),
            np.ones((1, 3)),
            scale=1.0,
            bounds=np.asarray([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        )
