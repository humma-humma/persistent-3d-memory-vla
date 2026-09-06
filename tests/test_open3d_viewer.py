import numpy as np

from sfm_reconstruction.models import Pose
from sfm_reconstruction.open3d_viewer import (
    camera_geometry_arrays,
    choose_point_cloud_path,
    colors_for_mode,
    load_ascii_ply_table,
)


def test_choose_point_cloud_path_prefers_rich_export(tmp_path):
    basic = tmp_path / "estimated_points.ply"
    rich = tmp_path / "estimated_points_rich.ply"
    basic.write_text("", encoding="ascii")
    rich.write_text("", encoding="ascii")

    assert choose_point_cloud_path(tmp_path, None) == rich
    explicit = tmp_path / "custom.ply"
    assert choose_point_cloud_path(tmp_path, explicit) == explicit


def test_load_ascii_ply_table_and_color_modes(tmp_path):
    cloud_path = tmp_path / "points.ply"
    cloud_path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int observations",
                "property float mean_reprojection_error",
                "property float max_triangulation_angle",
                "end_header",
                "0 0 0 255 0 0 2 0.5 1.0",
                "0 1 2 0 0 255 5 3.0 4.0",
            )
        ),
        encoding="ascii",
    )

    table = load_ascii_ply_table(cloud_path)

    assert table.points.tolist() == [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
    assert colors_for_mode(table, "rgb").tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert colors_for_mode(table, "track_length").shape == (2, 3)
    assert colors_for_mode(table, "reprojection_error").shape == (2, 3)
    assert colors_for_mode(table, "triangulation_angle").shape == (2, 3)
    assert colors_for_mode(table, "height").shape == (2, 3)


def test_camera_geometry_arrays_include_trajectory_and_frustums():
    poses = {
        "00000.jpg": Pose.identity(),
        "00001.jpg": Pose(np.eye(3), np.array([-1.0, 0.0, 0.0])),
    }
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )

    vertices, lines, colors = camera_geometry_arrays(poses, intrinsics, 2.0)

    assert vertices.shape[1] == 3
    assert lines.shape[1] == 2
    assert colors.shape == (len(lines), 3)
    assert len(lines) == 17
