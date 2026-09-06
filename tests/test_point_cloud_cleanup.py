import numpy as np

from sfm_reconstruction.point_cloud_cleanup import write_rgb_point_cloud


def test_write_rgb_point_cloud_exports_points_and_colors(tmp_path):
    output = tmp_path / "cleaned.ply"

    count = write_rgb_point_cloud(
        output,
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[10, 20, 30]], dtype=np.uint8),
    )

    lines = output.read_text(encoding="ascii").splitlines()
    assert count == 1
    assert "property uchar red" in lines
    assert lines[-1] == "1 2 3 10 20 30"
