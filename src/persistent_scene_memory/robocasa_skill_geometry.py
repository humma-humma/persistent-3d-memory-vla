"""Geometry-provider features for the hierarchical RoboCasa skill head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class RelativeWorldGeometryProvider:
    """Adapt privileged, reconstructed, or remembered world geometry to one contract."""

    def __init__(self, source: str):
        if not source:
            raise ValueError("geometry source must be nonempty")
        self.source = source
        self.initial_info = None

    def reset(self, initial_info: dict) -> None:
        self.initial_info = dict(initial_info)

    def features(self, current_info: dict) -> np.ndarray:
        if self.initial_info is None:
            raise RuntimeError("geometry provider must be reset before use")
        return geometry_feature_vector(current_info, self.initial_info)


def geometry_feature_vector(info: dict, initial_info: dict) -> np.ndarray:
    """Build geometry features available from odometry plus object memory."""
    required = ("eef_world", "object_world", "base_world")
    if any(key not in info or key not in initial_info for key in required):
        raise ValueError("geometry metadata requires eef, object, and base world positions")
    eef = np.asarray(info["eef_world"], dtype=np.float32)
    obj = np.asarray(info["object_world"], dtype=np.float32)
    base = np.asarray(info["base_world"], dtype=np.float32)
    initial_eef = np.asarray(initial_info["eef_world"], dtype=np.float32)
    initial_obj = np.asarray(initial_info["object_world"], dtype=np.float32)
    initial_base = np.asarray(initial_info["base_world"], dtype=np.float32)
    vectors = (eef - obj, obj - initial_obj, base - initial_base, eef - initial_eef)
    if any(value.shape != (3,) for value in vectors):
        raise ValueError("world positions must be 3D")
    return np.concatenate(
        [
            *vectors,
            np.asarray([np.linalg.norm(value) for value in vectors[:3]], dtype=np.float32),
            np.asarray([float(bool(info.get("grasped", False)))], dtype=np.float32),
        ]
    )


def augment_skill_features(source: str | Path, output: str | Path) -> dict:
    source = Path(source)
    with np.load(source) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    roots = arrays["episode_roots"].tolist()
    frame_indices = arrays["frame_indices"].astype(np.int64)
    manifests = {}
    initial_info = {}
    geometry = []
    for root, frame_index in zip(roots, frame_indices):
        if root not in manifests:
            manifest = json.loads(
                (Path(root) / "manifest.json").read_text(encoding="utf-8")
            )
            manifests[root] = manifest
            initial_info[root] = manifest["frames"][0]["action_source"]
        info = manifests[root]["frames"][int(frame_index)]["action_source"]
        geometry.append(geometry_feature_vector(info, initial_info[root]))
    geometry_array = np.stack(geometry).astype(np.float32)
    arrays["context_features"] = arrays["features"]
    arrays["geometry_features"] = geometry_array
    arrays["features"] = np.concatenate(
        [arrays["context_features"], geometry_array], axis=1
    ).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "status": "skill_geometry_complete",
        "samples": len(geometry_array),
        "context_dimension": int(arrays["context_features"].shape[1]),
        "geometry_dimension": int(geometry_array.shape[1]),
        "combined_dimension": int(arrays["features"].shape[1]),
        "output": str(output.resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(augment_skill_features(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
