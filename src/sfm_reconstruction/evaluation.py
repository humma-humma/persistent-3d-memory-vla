from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .models import Pose


@dataclass(frozen=True)
class PoseMetrics:
    cameras: int
    scale: float
    mean_rotation_radians: float
    mean_rotation_degrees: float
    median_rotation_degrees: float
    mean_translation: float
    median_translation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_poses(
    estimated: dict[int, Pose],
    ground_truth: dict[int, np.ndarray],
) -> PoseMetrics | None:
    camera_ids = sorted(set(estimated) & set(ground_truth))
    if not camera_ids:
        return None

    scale_ratios = []
    for camera_id in camera_ids:
        estimated_norm = np.linalg.norm(estimated[camera_id].translation)
        ground_truth_norm = np.linalg.norm(ground_truth[camera_id][:3, 3])
        if ground_truth_norm > 1e-8:
            scale_ratios.append(estimated_norm / ground_truth_norm)
    if not scale_ratios:
        return None
    scale = float(np.mean(scale_ratios))

    rotation_errors = []
    translation_errors = []
    for camera_id in camera_ids:
        predicted = estimated[camera_id]
        target = ground_truth[camera_id]
        cosine = (
            np.trace(target[:3, :3] @ predicted.rotation.T) - 1.0
        ) / 2.0
        rotation_errors.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        translation_errors.append(
            np.linalg.norm(
                target[:3, 3] - predicted.translation.ravel() / scale
            )
        )

    rotations = np.asarray(rotation_errors)
    translations = np.asarray(translation_errors)
    return PoseMetrics(
        cameras=len(camera_ids),
        scale=scale,
        mean_rotation_radians=float(np.mean(rotations)),
        mean_rotation_degrees=float(np.degrees(np.mean(rotations))),
        median_rotation_degrees=float(np.degrees(np.median(rotations))),
        mean_translation=float(np.mean(translations)),
        median_translation=float(np.median(translations)),
    )
