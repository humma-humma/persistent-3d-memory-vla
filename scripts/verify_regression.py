from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_SUMMARIES: dict[str, dict[str, Any]] = {
    "outputs/stage1_box_filter_regression": {
        "images": 46,
        "registered_cameras": 46,
        "points": 5746,
        "tracks": 5856,
        "initial_pair": [0, 1],
        "pose_evaluation.mean_rotation_degrees": 0.8751328865568679,
        "pose_evaluation.median_rotation_degrees": 0.9240908961273638,
        "pose_evaluation.mean_translation": 0.06662300991196492,
        "pose_evaluation.median_translation": 0.06462088261555078,
    },
    "outputs/stage2_milk_masked_track3_ba": {
        "images": 50,
        "registered_cameras": 50,
        "points": 1739,
        "tracks": 2688,
        "initial_pair": [0, 56],
        "pose_evaluation.mean_rotation_degrees": 6.778941878778285,
        "pose_evaluation.median_rotation_degrees": 5.631800008571144,
        "pose_evaluation.mean_translation": 0.06253726404733964,
        "pose_evaluation.median_translation": 0.052623883256953205,
    },
    "outputs/stage2_milk_masked_track4_ba": {
        "images": 50,
        "registered_cameras": 50,
        "points": 961,
        "tracks": 1556,
        "initial_pair": [0, 56],
        "pose_evaluation.mean_rotation_degrees": 6.507478560960005,
        "pose_evaluation.median_rotation_degrees": 5.035375426746094,
        "pose_evaluation.mean_translation": 0.061025079179172846,
        "pose_evaluation.median_translation": 0.04402158761451644,
    },
    "outputs/stage2_boot_improved": {
        "images": 51,
        "registered_cameras": 51,
        "points": 1254,
        "tracks": 2911,
        "initial_pair": [0, 67],
    },
    "outputs/stage2_milk_sift_akaze_probe": {
        "images": 50,
        "registered_cameras": 50,
        "points": 8004,
        "tracks": 16465,
        "initial_pair": [0, 56],
        "pose_evaluation.mean_rotation_degrees": 7.562833873314472,
        "pose_evaluation.median_rotation_degrees": 5.550360376941608,
        "pose_evaluation.mean_translation": 0.06772543612120346,
        "pose_evaluation.median_translation": 0.04837975637702165,
    },
    "outputs/stage2_boot_sift_akaze_probe": {
        "images": 51,
        "registered_cameras": 51,
        "points": 5291,
        "tracks": 17003,
        "initial_pair": [0, 67],
    },
}

EXPECTED_PLY_COUNTS = {
    "outputs/stage1_box_filter_regression/estimated_points.ply": 5746,
    "outputs/stage2_milk_masked_track3_ba/estimated_points.ply": 1739,
    "outputs/stage2_milk_masked_track4_ba/estimated_points.ply": 961,
    "outputs/stage2_boot_improved/estimated_points.ply": 1254,
    "outputs/stage2_milk_sift_akaze_probe/estimated_points.ply": 8004,
    "outputs/stage2_boot_sift_akaze_probe/estimated_points.ply": 5291,
    "outputs/stage2_milk_sift_akaze_probe/dense_points_allpairs_cleaned.ply": 63938,
    "outputs/stage2_boot_sift_akaze_probe/dense_points_allpairs_cleaned.ply": 79872,
}

EXPECTED_VIDEOS = {
    "assets/demo/boot_dense_open3d.mp4": {
        "frames": 120,
        "width": 960,
        "height": 540,
        "fps": 20.0,
    },
    "assets/demo/milk_dense_open3d.mp4": {
        "frames": 120,
        "width": 960,
        "height": 540,
        "fps": 20.0,
    },
}

DELIVERABLE_ARCHIVE = Path("deliverables/reconstruction_deliverables_2026-06-18.zip")
DELIVERABLE_SHA256 = "D5C1BBD635358D5FB7B9BFA1B533CCCD039F864E46927BC0F86AB600D9E359EA"


def nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def read_ply_vertex_count(path: Path) -> int:
    with path.open("r", encoding="ascii") as file:
        if file.readline().strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        for line in file:
            fields = line.strip().split()
            if fields[:2] == ["element", "vertex"]:
                return int(fields[2])
            if fields == ["end_header"]:
                break
    raise ValueError(f"{path} has no vertex count")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check_summaries(root: Path, errors: list[str]) -> None:
    for relative, expected in EXPECTED_SUMMARIES.items():
        summary_path = root / relative / "summary.json"
        if not summary_path.is_file():
            errors.append(f"Missing summary: {summary_path}")
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            try:
                actual = nested_get(data, key)
            except KeyError:
                errors.append(f"{summary_path}: missing key {key}")
                continue
            if isinstance(expected_value, float):
                if abs(float(actual) - expected_value) > 1e-9:
                    errors.append(
                        f"{summary_path}: {key}={actual!r}, expected {expected_value!r}"
                    )
            elif actual != expected_value:
                errors.append(
                    f"{summary_path}: {key}={actual!r}, expected {expected_value!r}"
                )
        print(f"summary ok: {relative}")


def check_ply_counts(root: Path, errors: list[str]) -> None:
    for relative, expected_count in EXPECTED_PLY_COUNTS.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing PLY: {path}")
            continue
        try:
            actual = read_ply_vertex_count(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        if actual != expected_count:
            errors.append(f"{path}: vertices={actual}, expected {expected_count}")
        print(f"ply ok: {relative}")


def check_videos(root: Path, errors: list[str]) -> None:
    try:
        import cv2
    except ImportError:
        errors.append("OpenCV is required to verify demo MP4 metadata")
        return

    for relative, expected in EXPECTED_VIDEOS.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing video: {path}")
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                errors.append(f"Could not open video: {path}")
                continue
            actual = {
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            }
        finally:
            capture.release()
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if isinstance(expected_value, float):
                if abs(actual_value - expected_value) > 1e-6:
                    errors.append(
                        f"{path}: {key}={actual_value!r}, expected {expected_value!r}"
                    )
            elif actual_value != expected_value:
                errors.append(
                    f"{path}: {key}={actual_value!r}, expected {expected_value!r}"
                )
        print(f"video ok: {relative}")


def check_deliverable_archive(root: Path, errors: list[str]) -> None:
    archive = root / DELIVERABLE_ARCHIVE
    if not archive.is_file():
        print(f"archive skipped: {DELIVERABLE_ARCHIVE} is not present")
        return
    actual = file_sha256(archive)
    if actual != DELIVERABLE_SHA256:
        errors.append(
            f"{archive}: sha256={actual}, expected {DELIVERABLE_SHA256}"
        )
    print(f"archive ok: {DELIVERABLE_ARCHIVE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify checked-in demos and archived reconstruction outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip MP4 metadata checks.",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Skip deliverables zip checksum check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    errors: list[str] = []

    check_summaries(root, errors)
    check_ply_counts(root, errors)
    if not args.skip_videos:
        check_videos(root, errors)
    if not args.skip_archive:
        check_deliverable_archive(root, errors)

    if errors:
        print("\nVerification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nRegression artifacts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
