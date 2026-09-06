import numpy as np
import pytest

from sfm_reconstruction.demo_video import orbit_front, trim_point_table
from sfm_reconstruction.open3d_viewer import PlyTable


def test_trim_point_table_filters_percentile_outlier():
    table = PlyTable(
        ["x", "y", "z", "red"],
        np.array(
            [
                [0.0, 0.0, 0.0, 255.0],
                [1.0, 1.0, 1.0, 128.0],
                [2.0, 2.0, 2.0, 64.0],
                [100.0, 100.0, 100.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    trimmed = trim_point_table(table, 10.0)

    assert trimmed.values.shape[0] == 2
    assert trimmed.values[:, :3].tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]


def test_trim_point_table_rejects_invalid_percentile():
    table = PlyTable(["x", "y", "z"], np.zeros((1, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="trim-percentile"):
        trim_point_table(table, 50.0)


def test_orbit_front_keeps_unit_length_and_rotates():
    first = orbit_front(0, 4, np.array([1.0, 0.0, -0.5]))
    second = orbit_front(1, 4, np.array([1.0, 0.0, -0.5]))

    assert np.isclose(np.linalg.norm(first), 1.0)
    assert np.isclose(np.linalg.norm(second), 1.0)
    assert not np.allclose(first, second)
