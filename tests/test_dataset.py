import json

import numpy as np

from sfm_reconstruction.dataset import load_stage1_dataset


def test_load_stage1_dataset(tmp_path):
    images = tmp_path / "images"
    correspondences = tmp_path / "correspondences"
    images.mkdir()
    correspondences.mkdir()
    (images / "00000.jpg").touch()
    (images / "00001.jpg").touch()
    np.savetxt(
        correspondences / "0_1.txt",
        np.array([[10.0, 20.0, 11.0, 20.0]]),
    )
    (tmp_path / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0, 0, 1]],
                "extrinsics": {},
            }
        )
    )

    dataset = load_stage1_dataset(tmp_path)

    assert dataset.image_ids == [0, 1]
    assert dataset.load_correspondences((0, 1)).shape == (1, 4)
    assert dataset.intrinsics.shape == (3, 3)
