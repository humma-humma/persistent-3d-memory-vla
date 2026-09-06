import numpy as np

from sfm_reconstruction.dense_fusion import (
    DenseFusionConfig,
    DensePairCandidate,
    DensePairResult,
    _clamp_disparity_range,
    camera_to_camera_transform,
    scale_intrinsics,
    select_neighbor_pairs,
    select_useful_pair_candidates,
    score_neighbor_pairs,
    sparse_bounds,
    write_dense_diagnostics,
    write_dense_point_cloud,
)
from sfm_reconstruction.models import Pose


def test_select_neighbor_pairs_can_limit_evenly() -> None:
    all_pairs = select_neighbor_pairs(
        ["00000.jpg", "00001.jpg", "00002.jpg"],
        pair_step=1,
    )
    pairs = select_neighbor_pairs(
        ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg", "00004.jpg"],
        pair_step=1,
        max_pairs=3,
    )

    assert all_pairs == [
        ("00000.jpg", "00001.jpg"),
        ("00001.jpg", "00002.jpg"),
    ]
    assert pairs == [
        ("00000.jpg", "00001.jpg"),
        ("00001.jpg", "00002.jpg"),
        ("00003.jpg", "00004.jpg"),
    ]


def test_select_neighbor_pairs_can_include_multiple_steps() -> None:
    pairs = select_neighbor_pairs(
        ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"],
        pair_step=1,
        max_pair_step=2,
    )

    assert pairs == [
        ("00000.jpg", "00001.jpg"),
        ("00001.jpg", "00002.jpg"),
        ("00002.jpg", "00003.jpg"),
        ("00000.jpg", "00002.jpg"),
        ("00001.jpg", "00003.jpg"),
    ]


def test_select_neighbor_pairs_rejects_invalid_limits() -> None:
    try:
        select_neighbor_pairs(["00000.jpg", "00001.jpg"], max_pairs=0)
    except ValueError as error:
        assert "max_pairs" in str(error)
    else:
        raise AssertionError("Expected max_pairs validation error")


def test_score_and_select_useful_pair_candidates() -> None:
    names = ["00000.jpg", "00001.jpg", "00002.jpg"]
    poses = {
        "00000.jpg": Pose.identity(),
        "00001.jpg": Pose(np.eye(3), np.array([-0.2, 0.0, 0.0])),
        "00002.jpg": Pose(np.eye(3), np.array([-2.0, 0.0, 0.0])),
    }
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    sparse_points = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.1, 0.0, 5.0],
            [0.0, 0.1, 5.0],
            [10.0, 10.0, 5.0],
        ]
    )

    candidates = score_neighbor_pairs(
        names,
        [("00000.jpg", "00001.jpg"), ("00000.jpg", "00002.jpg")],
        poses,
        intrinsics,
        sparse_points,
        image_size=(100, 100),
    )
    useful = select_useful_pair_candidates(
        candidates,
        min_sparse_support=3,
        min_triangulation_angle=0.1,
        max_pairs=1,
    )

    assert candidates[0].sparse_support == 3
    assert candidates[1].score > candidates[0].score
    assert [(pair.first_name, pair.second_name) for pair in useful] == [
        ("00000.jpg", "00002.jpg")
    ]


def test_camera_to_camera_transform_matches_pose_convention() -> None:
    first = Pose.identity()
    second = Pose(np.eye(3), np.array([-1.0, 0.0, 0.0]))

    rotation, translation = camera_to_camera_transform(first, second)

    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation.ravel(), [-1.0, 0.0, 0.0])


def test_scale_intrinsics_scales_focal_lengths_and_principal_point() -> None:
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]]
    )

    scaled = scale_intrinsics(intrinsics, 0.5)

    assert np.allclose(
        scaled,
        [[50.0, 0.0, 25.0], [0.0, 60.0, 20.0], [0.0, 0.0, 1.0]],
    )


def test_sparse_bounds_expands_percentile_box() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [100.0, 100.0, 100.0],
        ]
    )

    lower, upper = sparse_bounds(points, percentile=0.0, margin_factor=0.1)

    assert np.allclose(lower, [-10.0, -10.0, -10.0])
    assert np.allclose(upper, [110.0, 110.0, 110.0])


def test_disparity_range_clamp_leaves_valid_image_columns() -> None:
    minimum, count = _clamp_disparity_range(256, 224, image_width=480)

    assert count % 16 == 0
    assert count >= 16
    assert minimum + count < 480


def test_write_dense_point_cloud_exports_rgb_and_pair_metadata(tmp_path) -> None:
    output = tmp_path / "dense.ply"
    count = write_dense_point_cloud(
        output,
        [
            DensePairResult(
                first_name="00000.jpg",
                second_name="00033.jpg",
                points=np.array([[1.0, 2.0, 3.0]]),
                colors=np.array([[10, 20, 30]], dtype=np.uint8),
                disparities=np.array([4.5]),
            )
        ],
    )

    lines = output.read_text(encoding="ascii").splitlines()
    assert count == 1
    assert "property uchar red" in lines
    assert "property int pair_index" in lines
    assert lines[-1] == "1 2 3 10 20 30 0 0 33 4.5"


def test_write_dense_diagnostics_exports_pair_camera_and_spatial_tables(tmp_path) -> None:
    output = tmp_path / "dense.ply"
    candidate = DensePairCandidate(
        first_name="00000.jpg",
        second_name="00001.jpg",
        step=1,
        sparse_support=12,
        baseline=0.5,
        median_triangulation_angle=3.0,
        score=36.0,
    )
    result = DensePairResult(
        first_name="00000.jpg",
        second_name="00001.jpg",
        points=np.array([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]]),
        colors=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        disparities=np.array([1.0, 2.0]),
    )

    paths = write_dense_diagnostics(
        output,
        ["00000.jpg", "00001.jpg"],
        [candidate],
        [candidate],
        [result],
        [result],
        (np.zeros(3), np.ones(3)),
        DenseFusionConfig(spatial_bins=2, min_dense_points_per_pair=1),
    )

    assert "status" in (tmp_path / "dense.pairs.csv").read_text(encoding="ascii")
    assert "00000.jpg,0,2" in (tmp_path / "dense.pair_heatmap.csv").read_text(
        encoding="ascii"
    )
    assert "total_points" in (tmp_path / "dense.cameras.csv").read_text(
        encoding="ascii"
    )
    assert "x_bin,y_bin,z_bin,count" in (
        tmp_path / "dense.spatial_density.csv"
    ).read_text(encoding="ascii")
    assert set(paths) == {"pairs", "pair_heatmap", "cameras", "spatial_density"}
