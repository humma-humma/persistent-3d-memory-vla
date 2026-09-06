"""Interactive Open3D timeline viewer for persistent scene-memory snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


COLOR_MODES = ("status", "confidence", "rgb")


@dataclass(frozen=True)
class MemoryFrame:
    frame_index: int
    keys: tuple[str, ...]
    positions: np.ndarray
    features: np.ndarray
    confidence: np.ndarray
    ages: np.ndarray


def load_memory_frame(path: str | Path) -> MemoryFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frame_index = int(payload["frame_index"])
    tokens = payload["tokens"]
    feature_size = int(payload["feature_size"])
    positions = np.asarray([token["position"] for token in tokens], dtype=np.float64)
    features = np.asarray([token["feature"] for token in tokens], dtype=np.float64)
    confidence = np.asarray([token["confidence"] for token in tokens], dtype=np.float64)
    ages = np.asarray(
        [frame_index - int(token["last_seen"]) for token in tokens],
        dtype=np.int64,
    )
    if tokens:
        positions = positions.reshape(-1, 3)
        features = features.reshape(-1, feature_size)
    else:
        positions = np.empty((0, 3), dtype=np.float64)
        features = np.empty((0, feature_size), dtype=np.float64)
    if np.any(ages < 0):
        raise ValueError(f"{path} contains a token last seen in the future")
    return MemoryFrame(
        frame_index=frame_index,
        keys=tuple(str(token["key"]) for token in tokens),
        positions=positions,
        features=features,
        confidence=confidence,
        ages=ages,
    )


def load_memory_timeline(memory_dir: str | Path) -> list[MemoryFrame]:
    paths = sorted(Path(memory_dir).glob("memory_frame_*.json"))
    if not paths:
        raise FileNotFoundError(f"no memory_frame_*.json files found in {memory_dir}")
    frames = [load_memory_frame(path) for path in paths]
    if any(second.frame_index <= first.frame_index for first, second in zip(frames, frames[1:])):
        raise ValueError("memory frame indices must be strictly increasing")
    return frames


def colors_for_memory_frame(frame: MemoryFrame, mode: str) -> np.ndarray:
    if mode not in COLOR_MODES:
        raise ValueError(f"unknown memory color mode: {mode}")
    if mode == "rgb":
        if frame.features.shape[1] < 3:
            return np.ones((len(frame.keys), 3), dtype=np.float64)
        return np.clip(frame.features[:, :3], 0.0, 1.0)
    if mode == "confidence":
        confidence = np.clip(frame.confidence, 0.0, 1.0)[:, None]
        low = np.asarray([0.9, 0.12, 0.08])
        high = np.asarray([0.1, 0.9, 0.25])
        return low + confidence * (high - low)

    current = np.asarray([0.1, 0.9, 0.3])
    persisted = np.asarray([1.0, 0.5, 0.08])
    colors = np.where((frame.ages == 0)[:, None], current, persisted)
    brightness = 0.35 + 0.65 * np.clip(frame.confidence, 0.0, 1.0)
    return colors * brightness[:, None]


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is not installed. Install the project with the open3d extra."
        ) from exc
    return o3d


def open_memory_viewer(
    memory_dir: str | Path,
    *,
    point_size: float = 6.0,
    initial_mode: str = "status",
) -> None:
    if point_size <= 0.0:
        raise ValueError("point_size must be positive")
    if initial_mode not in COLOR_MODES:
        raise ValueError(f"unknown memory color mode: {initial_mode}")
    frames = load_memory_timeline(memory_dir)
    o3d = _require_open3d()
    state = {"frame": 0, "mode": COLOR_MODES.index(initial_mode)}
    first_frame = frames[0]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(first_frame.positions)
    cloud.colors = o3d.utility.Vector3dVector(
        colors_for_memory_frame(first_frame, initial_mode)
    )

    def update(vis) -> bool:
        frame = frames[state["frame"]]
        mode = COLOR_MODES[state["mode"]]
        cloud.points = o3d.utility.Vector3dVector(frame.positions)
        cloud.colors = o3d.utility.Vector3dVector(colors_for_memory_frame(frame, mode))
        vis.update_geometry(cloud)
        current = int(np.sum(frame.ages == 0))
        persisted = len(frame.keys) - current
        print(
            f"\rFrame {frame.frame_index} ({state['frame'] + 1}/{len(frames)}) | "
            f"current {current} | persisted {persisted} | mode {mode}      ",
            end="",
            flush=True,
        )
        return False

    def next_frame(vis) -> bool:
        state["frame"] = min(state["frame"] + 1, len(frames) - 1)
        return update(vis)

    def previous_frame(vis) -> bool:
        state["frame"] = max(state["frame"] - 1, 0)
        return update(vis)

    def next_mode(vis) -> bool:
        state["mode"] = (state["mode"] + 1) % len(COLOR_MODES)
        return update(vis)

    def reset_view(vis) -> bool:
        vis.reset_view_point(True)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Persistent Scene Memory - {Path(memory_dir).name}")
    vis.add_geometry(cloud, reset_bounding_box=True)
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), previous_frame)
    vis.register_key_callback(ord("M"), next_mode)
    vis.register_key_callback(ord("R"), reset_view)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0.02, 0.02, 0.025])
    print("Controls: N next | P previous | M color mode | R reset view | Q close")
    update(vis)
    vis.run()
    vis.destroy_window()
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--color-mode", choices=COLOR_MODES, default="status")
    args = parser.parse_args(argv)
    open_memory_viewer(
        args.memory_dir.resolve(),
        point_size=args.point_size,
        initial_mode=args.color_mode,
    )


if __name__ == "__main__":
    main()
