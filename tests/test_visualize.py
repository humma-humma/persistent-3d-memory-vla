import json

import numpy as np

from sfm_reconstruction.visualize import load_ascii_ply, load_camera_poses


def test_visualization_loaders(tmp_path):
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
                "end_header",
                "1 2 3",
                "4 5 6",
            )
        ),
        encoding="ascii",
    )
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(
        json.dumps(
            {
                "intrinsics": np.eye(3).tolist(),
                "extrinsics": {"00000.jpg": np.eye(4).tolist()},
            }
        ),
        encoding="utf-8",
    )

    points = load_ascii_ply(cloud_path)
    poses = load_camera_poses(camera_path)

    assert points.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert np.allclose(poses["00000.jpg"].camera_center, np.zeros(3))
