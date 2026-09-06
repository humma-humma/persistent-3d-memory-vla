import csv
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from sfm_reconstruction.dataset import Stage1Dataset
from sfm_reconstruction.matching import (
    FeatureSet,
    MatchingConfig,
    _feature_mask,
    _feature_methods,
    _match_feature_sets,
    circular_image_pairs,
    generate_correspondences,
    matching_image_pairs,
    match_descriptors,
)


def test_custom_matcher_applies_ratio_and_mutual_checks():
    first = np.array(
        [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    second = np.array(
        [[0.01, 0.0], [1.01, 1.0], [20.0, 20.0], [11.0, 11.0]],
        dtype=np.float32,
    )

    matches = match_descriptors(first, second, ratio_threshold=0.8)

    assert matches.tolist() == [[0, 0], [1, 1], [2, 3]]


def test_binary_matcher_applies_ratio_and_mutual_checks():
    first = np.array([[0], [255], [15]], dtype=np.uint8)
    second = np.array([[0], [240], [255], [128]], dtype=np.uint8)

    matches = match_descriptors(
        first,
        second,
        ratio_threshold=0.8,
        descriptor_type="binary",
    )

    assert matches.tolist() == [[0, 0], [1, 2]]


def test_feature_mode_can_combine_sift_and_akaze_sources():
    assert _feature_methods("sift+akaze") == ("sift", "akaze")

    first = (
        FeatureSet(
            points=np.array([[1.0, 2.0]]),
            descriptors=np.array([[0.0, 0.0]], dtype=np.float32),
            descriptor_type="float",
            source="sift",
        ),
        FeatureSet(
            points=np.array([[3.0, 4.0]]),
            descriptors=np.array([[0]], dtype=np.uint8),
            descriptor_type="binary",
            source="akaze",
        ),
    )
    second = (
        FeatureSet(
            points=np.array([[1.5, 2.5], [9.0, 9.0]]),
            descriptors=np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
            descriptor_type="float",
            source="sift",
        ),
        FeatureSet(
            points=np.array([[3.5, 4.5], [8.0, 8.0]]),
            descriptors=np.array([[0], [255]], dtype=np.uint8),
            descriptor_type="binary",
            source="akaze",
        ),
    )

    correspondences = _match_feature_sets(first, second, ratio_threshold=0.8)

    assert correspondences.tolist() == [
        [1.0, 2.0, 1.5, 2.5],
        [3.0, 4.0, 3.5, 4.5],
    ]


def test_generate_correspondences_supports_learned_frontend(tmp_path, monkeypatch):
    image_paths = {}
    for image_id in (0, 1):
        path = tmp_path / f"{image_id}.png"
        cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))
        image_paths[image_id] = path
    dataset = Stage1Dataset(
        root=tmp_path,
        intrinsics=np.eye(3),
        image_paths=image_paths,
        image_names={0: "0.png", 1: "1.png"},
        correspondence_paths={},
        ground_truth_extrinsics={},
    )

    class FakeMatcher:
        def extract(self, path, mask=None):
            return path

        def match(self, first, second):
            points = np.asarray(
                [[5, 5], [10, 8], [15, 12], [20, 18], [25, 24], [30, 30]],
                dtype=np.float64,
            )
            return points, points + np.asarray([1.0, 0.0])

    monkeypatch.setattr(
        "sfm_reconstruction.matching.create_superpoint_lightglue_matcher",
        lambda **kwargs: FakeMatcher(),
    )
    monkeypatch.setattr(
        "sfm_reconstruction.matching.estimate_relative_pose",
        lambda *args, **kwargs: SimpleNamespace(inliers=np.ones(6, dtype=bool)),
    )

    correspondence_dir, summary = generate_correspondences(
        dataset,
        tmp_path / "cache",
        MatchingConfig(
            feature_mode="superpoint-lightglue",
            pair_window=1,
            min_inliers=5,
        ),
    )

    assert summary.accepted_pairs == 1
    assert summary.correspondences == 6
    assert (correspondence_dir / "0_1.txt").is_file()


def test_wide_learned_pairs_require_cycle_support_and_write_diagnostics(
    tmp_path, monkeypatch
):
    image_paths = {}
    for image_id in range(4):
        path = tmp_path / f"{image_id}.png"
        cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))
        image_paths[image_id] = path
    dataset = Stage1Dataset(
        root=tmp_path,
        intrinsics=np.eye(3),
        image_paths=image_paths,
        image_names={image_id: f"{image_id}.png" for image_id in range(4)},
        correspondence_paths={},
        ground_truth_extrinsics={},
    )

    class FakeMatcher:
        def extract(self, path, mask=None):
            return int(path.stem)

        def match_with_indices(self, first, second):
            indices = np.column_stack((np.arange(6), np.arange(6))).astype(np.int32)
            points = np.asarray(
                [[5, 5], [20, 5], [35, 5], [5, 20], [20, 20], [35, 20]],
                dtype=np.float64,
            )
            return indices, points + first, points + second

    class FakeRetriever:
        def describe(self, paths):
            return np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)

    monkeypatch.setattr(
        "sfm_reconstruction.matching.create_superpoint_lightglue_matcher",
        lambda **kwargs: FakeMatcher(),
    )
    monkeypatch.setattr(
        "sfm_reconstruction.matching.create_dinov2_retriever",
        lambda **kwargs: FakeRetriever(),
    )
    monkeypatch.setattr(
        "sfm_reconstruction.matching.estimate_relative_pose",
        lambda *args, **kwargs: SimpleNamespace(inliers=np.ones(6, dtype=bool)),
    )

    cache_root = tmp_path / "cache"
    correspondence_dir, summary = generate_correspondences(
        dataset,
        cache_root,
        MatchingConfig(
            feature_mode="superpoint-lightglue",
            pair_window=1,
            min_inliers=5,
            wide_baseline=True,
            wide_pose_only=True,
            wide_retrieval_max_pairs=2,
            wide_min_frame_gap=2,
            wide_min_similarity=0.9,
            wide_min_spatial_coverage=0.2,
            wide_min_cycle_matches=5,
            wide_max_pairs_per_image=2,
        ),
    )

    assert summary.accepted_pairs == 6
    assert not (correspondence_dir / "0_2.txt").exists()
    assert (cache_root / "wide_correspondences" / "0_2.txt").is_file()
    with (cache_root / "pair_diagnostics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wide = [row for row in rows if row["pair_type"] == "retrieved"]
    assert len(wide) == 2
    assert all(row["cycle_matches"] == "6" for row in wide)
    assert all(row["accepted"] == "1" for row in wide)

    supplied_pairs = {
        (first, second): tmp_path / f"{first}_{second}.txt"
        for first in range(4)
        for second in range(first + 1, 4)
    }
    _, cycle_summary = generate_correspondences(
        replace(dataset, correspondence_paths=supplied_pairs),
        tmp_path / "cycle_cache",
        MatchingConfig(
            feature_mode="superpoint-lightglue",
            pair_source="supplied",
            min_inliers=5,
            learned_cycle_filter=True,
            learned_min_cycle_matches=5,
        ),
    )
    assert cycle_summary.accepted_pairs == 6
    assert cycle_summary.correspondences == 36

    for path in supplied_pairs.values():
        np.savetxt(path, np.zeros((2, 4)))
    _, augmented_summary = generate_correspondences(
        replace(dataset, correspondence_paths=supplied_pairs),
        tmp_path / "augmented_cache",
        MatchingConfig(
            feature_mode="superpoint-lightglue",
            pair_source="supplied",
            min_inliers=5,
            learned_cycle_filter=True,
            learned_min_cycle_matches=5,
            learned_augment_supplied=True,
        ),
    )
    assert augmented_summary.accepted_pairs == 6
    assert augmented_summary.correspondences == 30


def test_circular_pairs_include_loop_closure_neighbors():
    pairs = circular_image_pairs([0, 10, 20, 30, 40], pair_window=1)

    assert pairs == [(0, 10), (0, 40), (10, 20), (20, 30), (30, 40)]


def test_matching_pairs_can_reuse_supplied_pair_graph(tmp_path):
    dataset = Stage1Dataset(
        root=tmp_path,
        intrinsics=np.eye(3),
        image_paths={image_id: tmp_path / f"{image_id}.jpg" for image_id in range(4)},
        image_names={image_id: f"{image_id}.jpg" for image_id in range(4)},
        correspondence_paths={
            (0, 2): tmp_path / "0_2.txt",
            (1, 3): tmp_path / "1_3.txt",
        },
        ground_truth_extrinsics={},
    )

    assert matching_image_pairs(
        dataset, MatchingConfig(pair_source="supplied")
    ) == [(0, 2), (1, 3)]


def test_feature_mask_is_disabled_by_default():
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    assert _feature_mask(image, False, 0.12) is None
