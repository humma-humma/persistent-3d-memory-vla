import cv2
import numpy as np

from sfm_reconstruction.dataset import Stage1Dataset
from sfm_reconstruction.io import write_rich_point_cloud
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.reconstruction import ReconstructionResult


def test_write_rich_point_cloud_exports_color_and_track_metadata(tmp_path):
    image_path = tmp_path / "00000.png"
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[:, :] = [30, 20, 10]
    assert cv2.imwrite(str(image_path), image)

    dataset = Stage1Dataset(
        root=tmp_path,
        intrinsics=np.eye(3),
        image_paths={0: image_path},
        image_names={0: image_path.name},
        correspondence_paths={},
        ground_truth_extrinsics={},
    )
    result = ReconstructionResult(
        poses={0: Pose.identity()},
        points={0: np.array([1.0, 1.0, 1.0])},
        tracks=[Track(observations={0: np.array([1.0, 1.0])})],
        initial_pair=(0, 0),
        skipped_track_conflicts=0,
    )

    output = tmp_path / "points_rich.ply"
    write_rich_point_cloud(output, dataset, result, track_id_map={0: 17})

    lines = output.read_text(encoding="ascii").splitlines()
    assert "property uchar red" in lines
    assert "property int track_id" in lines
    assert "property float mean_reprojection_error" in lines
    values = lines[-1].split()
    assert values[:6] == ["1", "1", "1", "10", "20", "30"]
    assert values[6:9] == ["17", "1", "1"]
    assert values[9:12] == ["0", "0", "0"]
