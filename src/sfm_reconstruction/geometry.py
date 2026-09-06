from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Pose


@dataclass(frozen=True)
class RelativePoseEstimate:
    pose: Pose
    inliers: np.ndarray


@dataclass(frozen=True)
class PnPEstimate:
    pose: Pose
    inliers: np.ndarray


def project_points(
    points_3d: np.ndarray, pose: Pose, intrinsics: np.ndarray
) -> np.ndarray:
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    camera_points = (pose.rotation @ points_3d.T + pose.translation).T
    pixels_h = (np.asarray(intrinsics) @ camera_points.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def estimate_relative_pose(
    points_first: np.ndarray,
    points_second: np.ndarray,
    intrinsics: np.ndarray,
    ransac_threshold: float = 1.0,
) -> RelativePoseEstimate:
    points_first = np.asarray(points_first, dtype=np.float64).reshape(-1, 2)
    points_second = np.asarray(points_second, dtype=np.float64).reshape(-1, 2)
    if len(points_first) < 5:
        raise ValueError("At least five correspondences are required")

    essential, mask = cv2.findEssentialMat(
        points_first,
        points_second,
        np.asarray(intrinsics, dtype=np.float64),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )
    if essential is None:
        raise RuntimeError("Essential-matrix estimation failed")

    candidates = np.asarray(essential).reshape(-1, 3, 3)
    best: RelativePoseEstimate | None = None
    best_count = -1
    for candidate in candidates:
        count, rotation, translation, pose_mask = cv2.recoverPose(
            candidate,
            points_first,
            points_second,
            np.asarray(intrinsics, dtype=np.float64),
            mask=mask.copy() if mask is not None else None,
        )
        if count > best_count:
            best_count = int(count)
            best = RelativePoseEstimate(
                pose=Pose(rotation, translation),
                inliers=pose_mask.ravel().astype(bool),
            )
    if best is None:
        raise RuntimeError("Relative-pose recovery failed")
    return best


def triangulate_points(
    points_first: np.ndarray,
    points_second: np.ndarray,
    first_pose: Pose,
    second_pose: Pose,
    intrinsics: np.ndarray,
) -> np.ndarray:
    projection_first = np.asarray(intrinsics) @ np.hstack(
        (first_pose.rotation, first_pose.translation)
    )
    projection_second = np.asarray(intrinsics) @ np.hstack(
        (second_pose.rotation, second_pose.translation)
    )
    homogeneous = cv2.triangulatePoints(
        projection_first,
        projection_second,
        np.asarray(points_first, dtype=np.float64).reshape(-1, 2).T,
        np.asarray(points_second, dtype=np.float64).reshape(-1, 2).T,
    )
    return (homogeneous[:3] / homogeneous[3]).T


def points_in_front(points_3d: np.ndarray, pose: Pose) -> np.ndarray:
    camera_points = (
        pose.rotation @ np.asarray(points_3d, dtype=np.float64).reshape(-1, 3).T
        + pose.translation
    )
    return camera_points[2] > 0


def reprojection_errors(
    points_3d: np.ndarray,
    observations: np.ndarray,
    pose: Pose,
    intrinsics: np.ndarray,
) -> np.ndarray:
    projections = project_points(points_3d, pose, intrinsics)
    return np.linalg.norm(
        projections - np.asarray(observations, dtype=np.float64).reshape(-1, 2),
        axis=1,
    )


def triangulation_angles(
    points_3d: np.ndarray, first_pose: Pose, second_pose: Pose
) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    first_rays = points - first_pose.camera_center
    second_rays = points - second_pose.camera_center
    first_rays /= np.linalg.norm(first_rays, axis=1, keepdims=True)
    second_rays /= np.linalg.norm(second_rays, axis=1, keepdims=True)
    cosines = np.sum(first_rays * second_rays, axis=1)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: np.ndarray,
    reprojection_error: float = 4.0,
) -> PnPEstimate:
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(points_3d) < 6:
        raise ValueError("At least six 2D-3D correspondences are required")

    success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        np.asarray(intrinsics, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=1000,
        reprojectionError=reprojection_error,
        confidence=0.999,
    )
    if not success or inlier_indices is None:
        raise RuntimeError("PnP failed")

    inlier_indices = inlier_indices.ravel()
    cv2.solvePnP(
        points_3d[inlier_indices],
        points_2d[inlier_indices],
        np.asarray(intrinsics, dtype=np.float64),
        None,
        rotation_vector,
        translation,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    mask = np.zeros(len(points_3d), dtype=bool)
    mask[inlier_indices] = True
    return PnPEstimate(Pose(rotation, translation), mask)
