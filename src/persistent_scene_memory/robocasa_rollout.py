"""Record RoboCasa RGB-D rollouts and convert them to persistent geometry tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Callable, Mapping

import numpy as np

from .adapter import GeometryTokenAdapter
from .memory import GeometryObservation, PersistentSceneMemory


CAMERA_PREFIX = "video."


def _slug(name: str) -> str:
    return name.replace(".", "__")


def _zero_action(space) -> dict[str, np.ndarray | int]:
    if hasattr(space, "spaces"):
        return {key: _zero_action(value) for key, value in space.spaces.items()}
    if hasattr(space, "n"):
        return 0
    return np.zeros(space.shape, dtype=space.dtype)


def _simulator(env):
    wrapper = env.unwrapped
    return wrapper.env.sim


def _camera_pairs(observation: Mapping[str, object]) -> list[tuple[str, str]]:
    pairs = []
    for key in observation:
        if not key.startswith(CAMERA_PREFIX) or not key.endswith("_depth"):
            continue
        mapped_name = key[: -len("_depth")]
        sim_name = mapped_name[len(CAMERA_PREFIX) :]
        pairs.append((mapped_name, sim_name))
    return sorted(pairs)


def _calibration(env, camera_pairs, height: int, width: int) -> dict[str, dict[str, np.ndarray]]:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    sim = _simulator(env)
    return {
        mapped: {
            "intrinsics": get_camera_intrinsic_matrix(sim, name, height, width),
            "camera_to_world": get_camera_extrinsic_matrix(sim, name),
        }
        for mapped, name in camera_pairs
    }


def _metric_depth(env, depth: np.ndarray) -> np.ndarray:
    from robosuite.utils.camera_utils import get_real_depth_map

    return get_real_depth_map(_simulator(env), depth)


def record_episode(
    env,
    output_dir: str | Path,
    *,
    steps: int,
    seed: int = 0,
    action_fn: Callable[[object, int], Mapping[str, object]] | None = None,
    calibration_fn: Callable[..., dict] = _calibration,
    depth_fn: Callable[[object, np.ndarray], np.ndarray] = _metric_depth,
    stop_on_success: bool = False,
) -> dict:
    """Record synchronized observations and applied actions from one environment."""
    if steps < 1:
        raise ValueError("steps must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    observation, _ = env.reset(seed=seed)
    camera_pairs = _camera_pairs(observation)
    if not camera_pairs:
        raise ValueError("RoboCasa observation contains no depth cameras")

    frame_rows = []
    for frame_index in range(steps):
        first_rgb = np.asarray(observation[camera_pairs[0][0] + "_image"])
        height, width = first_rgb.shape[:2]
        calibration = calibration_fn(env, camera_pairs, height, width)
        if action_fn is None:
            action = dict(_zero_action(env.action_space))
        elif hasattr(action_fn, "act"):
            action = dict(action_fn.act(env, frame_index, observation))
        else:
            action = dict(action_fn(env, frame_index))
        next_observation, reward, terminated, truncated, info = env.step(action)

        arrays = {}
        for mapped_name, _ in camera_pairs:
            slug = _slug(mapped_name)
            arrays[f"rgb__{slug}"] = np.asarray(observation[mapped_name + "_image"], dtype=np.uint8)
            raw_depth = np.asarray(observation[mapped_name + "_depth"], dtype=np.float32).squeeze(-1)
            arrays[f"depth__{slug}"] = np.asarray(depth_fn(env, raw_depth), dtype=np.float32)
            arrays[f"intrinsics__{slug}"] = np.asarray(calibration[mapped_name]["intrinsics"], dtype=np.float64)
            arrays[f"camera_to_world__{slug}"] = np.asarray(calibration[mapped_name]["camera_to_world"], dtype=np.float64)
        state_keys = sorted(key for key in observation if key.startswith("state."))
        for key in state_keys:
            arrays[f"state__{_slug(key)}"] = np.asarray(observation[key])
        for key, value in sorted(action.items()):
            arrays[f"action__{_slug(key)}"] = np.asarray(value)

        filename = f"frame_{frame_index:06d}.npz"
        np.savez_compressed(destination / filename, **arrays)
        frame_row = {
                "frame_index": frame_index,
                "file": filename,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(info.get("success", False)),
            }
        if action_fn is not None and hasattr(action_fn, "last_info"):
            frame_row["action_source"] = dict(action_fn.last_info)
        frame_rows.append(frame_row)
        observation = next_observation
        if terminated or truncated or (stop_on_success and frame_row["success"]):
            break

    manifest = {
        "format": "robocasa-rgbd-rollout-v1",
        "seed": seed,
        "task": str(observation.get("annotation.human.task_description", "")),
        "cameras": [mapped for mapped, _ in camera_pairs],
        "state_keys": state_keys,
        "action_keys": sorted(action),
        "frames": frame_rows,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class RoboCasaEpisode:
    """Sequential reader for the portable rollout format."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != "robocasa-rgbd-rollout-v1":
            raise ValueError("unsupported RoboCasa rollout format")

    def __len__(self) -> int:
        return len(self.manifest["frames"])

    def frame(self, index: int) -> dict:
        row = self.manifest["frames"][index]
        with np.load(self.root / row["file"]) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        return {**row, "task": self.manifest["task"], "arrays": arrays}

    def world_points(self, index: int, *, stride: int = 8, max_depth: float = 5.0):
        """Backproject all cameras into MuJoCo world coordinates."""
        if stride < 1 or max_depth <= 0:
            raise ValueError("stride and max_depth must be positive")
        arrays = self.frame(index)["arrays"]
        point_parts, color_parts, camera_parts = [], [], []
        for camera_index, camera in enumerate(self.manifest["cameras"]):
            slug = _slug(camera)
            rgb = arrays[f"rgb__{slug}"]
            depth = arrays[f"depth__{slug}"]
            intrinsics = arrays[f"intrinsics__{slug}"]
            camera_to_world = arrays[f"camera_to_world__{slug}"]
            rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
            sampled_depth = depth[::stride, ::stride]
            valid = np.isfinite(sampled_depth) & (sampled_depth > 0) & (sampled_depth < max_depth)
            z = sampled_depth[valid]
            x = (cols[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
            y = (rows[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
            camera_points = np.column_stack([x, y, z, np.ones_like(z)])
            world = (camera_to_world @ camera_points.T).T[:, :3]
            point_parts.append(world)
            color_parts.append(rgb[::stride, ::stride][valid].astype(np.float64) / 255.0)
            camera_parts.append(np.full(len(world), camera_index, dtype=np.int32))
        return (
            np.concatenate(point_parts),
            np.concatenate(color_parts),
            np.concatenate(camera_parts),
        )


def episode_to_memory(
    episode: RoboCasaEpisode,
    output_dir: str | Path,
    *,
    stride: int = 8,
    voxel_size: float = 0.05,
    max_tokens: int = 128,
) -> dict:
    """Fuse world RGB-D voxels into memory and write policy-token snapshots."""
    if voxel_size <= 0 or max_tokens < 1:
        raise ValueError("voxel_size and max_tokens must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    memory = PersistentSceneMemory()
    adapter = GeometryTokenAdapter(max_tokens=max_tokens)
    rows = []
    camera_count = len(episode.manifest["cameras"])

    for frame_index in range(len(episode)):
        points, colors, camera_ids = episode.world_points(frame_index, stride=stride)
        voxels = np.floor(points / voxel_size).astype(np.int64)
        grouped = {}
        for voxel, point, color, camera_id in zip(voxels, points, colors, camera_ids):
            grouped.setdefault(tuple(voxel), []).append((point, color, int(camera_id)))
        for voxel, samples in grouped.items():
            sample_points = np.stack([item[0] for item in samples])
            sample_colors = np.stack([item[1] for item in samples])
            support = len({item[2] for item in samples})
            memory.update(
                GeometryObservation(
                    key="voxel:" + ":".join(map(str, voxel)),
                    position=sample_points.mean(axis=0),
                    feature=sample_colors.mean(axis=0),
                    confidence=support / camera_count,
                    frame_index=frame_index,
                )
            )
        memory.save(destination / f"memory_frame_{frame_index:06d}.json", frame_index)
        encoded = adapter.encode(memory.tokens(frame_index), frame_index, feature_size=3)
        token_path = destination / f"adapter_frame_{frame_index:06d}.npz"
        np.savez_compressed(token_path, values=encoded.values, mask=encoded.mask, keys=np.asarray(encoded.keys))
        rows.append({"frame_index": frame_index, "observed_voxels": len(grouped), "memory_tokens": len(memory.tokens(frame_index)), "policy_tokens": int(encoded.mask.sum())})

    summary = {"episode": str(episode.root), "voxel_size": voxel_size, "stride": stride, "frames": rows}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--object-group")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--action-source", choices=("zero", "oracle"), default="zero")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--oracle-grasp-orientation-tolerance", type=float, default=0.15)
    args = parser.parse_args(argv)

    import gymnasium as gym
    import robocasa  # noqa: F401

    random.seed(args.seed)
    np.random.seed(args.seed)
    environment_kwargs = {
        "split": "pretrain",
        "obj_registries": ("lightwheel",),
        "camera_depths": True,
        "camera_widths": args.camera_size,
        "camera_heights": args.camera_size,
    }
    if args.object_group:
        environment_kwargs["obj_groups"] = args.object_group
    env = gym.make(f"robocasa/{args.task}", **environment_kwargs)
    try:
        action_source = None
        if args.action_source == "oracle":
            from .robocasa_actions import OraclePickPlaceCounterToCabinet

            if args.task != "PickPlaceCounterToCabinet":
                raise ValueError("the oracle action source currently supports PickPlaceCounterToCabinet only")
            action_source = OraclePickPlaceCounterToCabinet(
                grasp_orientation_tolerance=args.oracle_grasp_orientation_tolerance
            )
        manifest = record_episode(
            env, args.output_dir / "episode", steps=args.steps, seed=args.seed,
            action_fn=action_source,
            stop_on_success=args.stop_on_success,
        )
    finally:
        env.close()
    summary = None
    if not args.skip_memory:
        summary = episode_to_memory(
            RoboCasaEpisode(args.output_dir / "episode"),
            args.output_dir / "memory",
            stride=args.stride,
            voxel_size=args.voxel_size,
            max_tokens=args.max_tokens,
        )
    if args.viewer:
        if args.skip_memory:
            raise ValueError("--viewer requires memory export; remove --skip-memory")
        from .robocasa_rollout_viewer import build_rollout_viewer

        (args.output_dir / "viewer.html").write_text(
            build_rollout_viewer(args.output_dir / "episode", args.output_dir / "memory"),
            encoding="utf-8",
        )
    print(json.dumps({
        "frames": len(manifest["frames"]),
        "task": manifest["task"],
        "success": bool(any(frame["success"] for frame in manifest["frames"])),
        "memory_exported": summary is not None,
        "final": summary["frames"][-1] if summary is not None else manifest["frames"][-1],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
