from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Stage1Dataset
from .geometry import project_points
from .reconstruction import ReconstructionResult


def reprojection_observation_rows(
    dataset: Stage1Dataset,
    result: ReconstructionResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for track_id in sorted(result.points):
        point = result.points[track_id].reshape(1, 3)
        track = result.tracks[track_id]
        for image_id in sorted(set(track.observations) & set(result.poses)):
            observed = track.observations[image_id].reshape(1, 2)
            projected = project_points(
                point,
                result.poses[image_id],
                dataset.intrinsics,
            )[0]
            error = float(np.linalg.norm(projected - observed[0]))
            rows.append(
                {
                    "image_id": image_id,
                    "image_name": dataset.image_names.get(image_id, str(image_id)),
                    "track_id": track_id,
                    "observed_x": float(observed[0, 0]),
                    "observed_y": float(observed[0, 1]),
                    "projected_x": float(projected[0]),
                    "projected_y": float(projected[1]),
                    "error": error,
                }
            )
    return rows


def per_camera_reprojection_rows(
    observation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    errors_by_camera: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in observation_rows:
        errors_by_camera[
            (int(row["image_id"]), str(row["image_name"]))
        ].append(float(row["error"]))
    camera_rows = []
    for (image_id, image_name), errors in sorted(errors_by_camera.items()):
        values = np.asarray(errors, dtype=np.float64)
        camera_rows.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "observations": len(values),
                "mean_error": float(np.mean(values)),
                "median_error": float(np.median(values)),
                "p90_error": float(np.percentile(values, 90)),
                "max_error": float(np.max(values)),
            }
        )
    return camera_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _draw_cross(
    image: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = point
    cv2.line(image, (x - 4, y), (x + 4, y), color, 1, cv2.LINE_AA)
    cv2.line(image, (x, y - 4), (x, y + 4), color, 1, cv2.LINE_AA)


def write_reprojection_overlays(
    output_dir: Path,
    dataset: Stage1Dataset,
    observation_rows: list[dict[str, object]],
    camera_rows: list[dict[str, object]],
    max_points_per_image: int = 300,
) -> int:
    if max_points_per_image < 1:
        raise ValueError("max_points_per_image must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in observation_rows:
        rows_by_image[int(row["image_id"])].append(row)
    camera_by_id = {int(row["image_id"]): row for row in camera_rows}

    written = 0
    for image_id, rows in sorted(rows_by_image.items()):
        image_path = dataset.image_paths.get(image_id)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        finite_rows = [
            row
            for row in rows
            if np.isfinite(
                [
                    row["observed_x"],
                    row["observed_y"],
                    row["projected_x"],
                    row["projected_y"],
                    row["error"],
                ]
            ).all()
        ]
        finite_rows = sorted(finite_rows, key=lambda row: float(row["error"]), reverse=True)
        for row in finite_rows[:max_points_per_image]:
            observed = (
                int(round(float(row["observed_x"]))),
                int(round(float(row["observed_y"]))),
            )
            projected = (
                int(round(float(row["projected_x"]))),
                int(round(float(row["projected_y"]))),
            )
            cv2.circle(image, observed, 3, (0, 220, 0), -1, cv2.LINE_AA)
            _draw_cross(image, projected, (0, 0, 255))
            cv2.line(image, observed, projected, (0, 220, 220), 1, cv2.LINE_AA)

        metrics = camera_by_id.get(image_id)
        if metrics is not None:
            label = (
                f"{dataset.image_names.get(image_id, image_id)} "
                f"n={metrics['observations']} "
                f"med={float(metrics['median_error']):.2f}px "
                f"p90={float(metrics['p90_error']):.2f}px"
            )
            cv2.putText(
                image,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        output_path = output_dir / f"{image_id:05d}_reprojection.png"
        if cv2.imwrite(str(output_path), image):
            written += 1
    return written


def write_reprojection_diagnostics(
    output_dir: str | Path,
    dataset: Stage1Dataset,
    result: ReconstructionResult,
    max_overlay_points: int = 300,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_rows = reprojection_observation_rows(dataset, result)
    camera_rows = per_camera_reprojection_rows(observation_rows)

    observations_path = output_dir / "observations.csv"
    cameras_path = output_dir / "per_camera.csv"
    overlays_dir = output_dir / "overlays"
    _write_csv(
        observations_path,
        observation_rows,
        [
            "image_id",
            "image_name",
            "track_id",
            "observed_x",
            "observed_y",
            "projected_x",
            "projected_y",
            "error",
        ],
    )
    _write_csv(
        cameras_path,
        camera_rows,
        [
            "image_id",
            "image_name",
            "observations",
            "mean_error",
            "median_error",
            "p90_error",
            "max_error",
        ],
    )
    overlays_written = write_reprojection_overlays(
        overlays_dir,
        dataset,
        observation_rows,
        camera_rows,
        max_overlay_points,
    )
    all_errors = np.asarray(
        [float(row["error"]) for row in observation_rows],
        dtype=np.float64,
    )
    summary = {
        "observations": len(observation_rows),
        "cameras": len(camera_rows),
        "overlays_written": overlays_written,
        "mean_error": float(np.mean(all_errors)) if len(all_errors) else None,
        "median_error": float(np.median(all_errors)) if len(all_errors) else None,
        "p90_error": float(np.percentile(all_errors, 90)) if len(all_errors) else None,
        "max_error": float(np.max(all_errors)) if len(all_errors) else None,
        "observations_csv": str(observations_path),
        "per_camera_csv": str(cameras_path),
        "overlays_dir": str(overlays_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
