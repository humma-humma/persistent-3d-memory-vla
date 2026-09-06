from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Iterator

import numpy as np


PAIR_PATTERN = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)\.txt$")


@dataclass(frozen=True)
class Stage1Dataset:
    root: Path
    intrinsics: np.ndarray
    image_paths: dict[int, Path]
    image_names: dict[int, str]
    correspondence_paths: dict[tuple[int, int], Path]
    ground_truth_extrinsics: dict[int, np.ndarray]

    @property
    def image_ids(self) -> list[int]:
        return sorted(self.image_paths)

    def load_correspondences(self, pair: tuple[int, int]) -> np.ndarray:
        path = self.correspondence_paths[pair]
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)
        if values.shape[1] != 4:
            raise ValueError(f"{path} must contain four columns, got {values.shape[1]}")
        return values

    def iter_correspondences(
        self,
    ) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
        for pair in sorted(self.correspondence_paths):
            yield pair, self.load_correspondences(pair)

    def subset(self, max_images: int | None) -> "Stage1Dataset":
        if max_images is None or max_images >= len(self.image_paths):
            return self
        if max_images < 2:
            raise ValueError("max_images must be at least 2")
        selected = set(self.image_ids[:max_images])
        return replace(
            self,
            image_paths={key: value for key, value in self.image_paths.items() if key in selected},
            image_names={key: value for key, value in self.image_names.items() if key in selected},
            correspondence_paths={
                pair: path
                for pair, path in self.correspondence_paths.items()
                if pair[0] in selected and pair[1] in selected
            },
            ground_truth_extrinsics={
                key: value
                for key, value in self.ground_truth_extrinsics.items()
                if key in selected
            },
        )


def load_dataset(
    root: str | Path,
    correspondence_dir: str | Path | None = None,
    require_correspondences: bool = True,
) -> Stage1Dataset:
    root = Path(root).resolve()
    images_dir = root / "images"
    correspondences_dir = (
        Path(correspondence_dir).resolve()
        if correspondence_dir is not None
        else root / "correspondences"
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if require_correspondences and not correspondences_dir.is_dir():
        raise FileNotFoundError(
            f"Missing correspondences directory: {correspondences_dir}"
        )

    camera_path = next(
        (
            root / name
            for name in (
                "gt_camera_parameters.json",
                "camera_parameters.json",
                "poses.json",
            )
            if (root / name).is_file()
        ),
        None,
    )
    if camera_path is None:
        raise FileNotFoundError(f"No camera parameter JSON found under {root}")

    camera_data = json.loads(camera_path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(camera_data["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("Camera intrinsics must have shape (3, 3)")

    image_paths: dict[int, Path] = {}
    image_names: dict[int, str] = {}
    for path in sorted(images_dir.glob("*.jpg")):
        image_id = int(path.stem)
        image_paths[image_id] = path
        image_names[image_id] = path.name
    if len(image_paths) < 2:
        raise ValueError(f"Expected at least two JPEG images under {images_dir}")

    correspondence_paths: dict[tuple[int, int], Path] = {}
    if correspondences_dir.is_dir():
        for path in correspondences_dir.glob("*.txt"):
            match = PAIR_PATTERN.match(path.name)
            if match is None:
                continue
            pair = (int(match["first"]), int(match["second"]))
            if pair[0] in image_paths and pair[1] in image_paths:
                correspondence_paths[pair] = path
    if require_correspondences and not correspondence_paths:
        raise ValueError(f"No valid correspondence files found under {correspondences_dir}")

    ground_truth_extrinsics = {
        int(Path(name).stem): np.asarray(matrix, dtype=np.float64)
        for name, matrix in camera_data.get("extrinsics", {}).items()
        if int(Path(name).stem) in image_paths
    }
    return Stage1Dataset(
        root=root,
        intrinsics=intrinsics,
        image_paths=image_paths,
        image_names=image_names,
        correspondence_paths=correspondence_paths,
        ground_truth_extrinsics=ground_truth_extrinsics,
    )


def load_stage1_dataset(root: str | Path) -> Stage1Dataset:
    return load_dataset(root)


def load_image_dataset(root: str | Path) -> Stage1Dataset:
    return load_dataset(root, require_correspondences=False)


def load_stage2_dataset(
    root: str | Path, correspondence_dir: str | Path
) -> Stage1Dataset:
    return load_dataset(root, correspondence_dir=correspondence_dir)
