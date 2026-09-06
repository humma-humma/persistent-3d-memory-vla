"""Offline SmolVLA replay on recorded RoboCasa observations.

This module intentionally does not expose a simulator action source.  The local
checkpoint uses SO-100 state/action semantics, whereas the recorded RoboCasa
task uses Cartesian arm, gripper, and mobile-base controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .robocasa_rollout import RoboCasaEpisode, _slug


ACTION_FIELDS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


class EmbodimentMismatchError(RuntimeError):
    """Raised when an SO-100 checkpoint output is requested as RoboCasa control."""


def quaternion_xyzw_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a stable three-dimensional rotation vector."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("quaternion must be nonzero")
    value = value / norm
    if value[3] < 0:
        value = -value
    vector_norm = np.linalg.norm(value[:3])
    if vector_norm < 1e-8:
        return (2.0 * value[:3]).astype(np.float32)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(value[3], -1.0, 1.0))
    return (value[:3] * angle / vector_norm).astype(np.float32)


def robocasa_state6(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Build a compact Cartesian pose state for interface testing.

    It has the checkpoint's expected width, but it is not claimed to have the
    SO-100 checkpoint's joint-state semantics or normalization distribution.
    """
    position = np.asarray(
        arrays["state__state__end_effector_position_relative"], dtype=np.float32
    )
    rotation = quaternion_xyzw_to_rotation_vector(
        arrays["state__state__end_effector_rotation_relative"]
    )
    if position.shape != (3,):
        raise ValueError("end-effector position must have shape (3,)")
    return np.concatenate((position, rotation)).astype(np.float32)


def robocasa_state16(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the complete recorded RoboCasa proprioceptive state."""
    fields = (
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    )
    parts = []
    for field in fields:
        key = f"state__{_slug(field)}"
        if key not in arrays:
            raise KeyError(f"recorded frame is missing {field!r}")
        parts.append(np.asarray(arrays[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(parts)
    if state.shape != (16,):
        raise ValueError(f"expected a 16D RoboCasa state, received {state.shape}")
    return state


def robocasa_action12(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a recorded RoboCasa action in a documented, stable field order."""
    parts = []
    for field in ACTION_FIELDS:
        key = f"action__{_slug(field)}"
        if key not in arrays:
            raise KeyError(f"recorded frame is missing {field!r}")
        parts.append(np.asarray(arrays[key], dtype=np.float32).reshape(-1))
    action = np.concatenate(parts)
    if action.shape != (12,):
        raise ValueError(f"expected a 12D RoboCasa action, received {action.shape}")
    return action


def load_replay_sample(
    episode: RoboCasaEpisode,
    memory_dir: str | Path,
    frame_index: int,
) -> dict:
    """Load aligned RGB, state, oracle action, and persistent geometry tokens."""
    frame = episode.frame(frame_index)
    arrays = frame["arrays"]
    cameras = episode.manifest["cameras"]
    if len(cameras) != 3:
        raise ValueError(f"SmolVLA checkpoint expects exactly 3 cameras, received {len(cameras)}")
    images = [arrays[f"rgb__{_slug(camera)}"] for camera in cameras]
    with np.load(Path(memory_dir) / f"adapter_frame_{frame_index:06d}.npz") as archive:
        geometry_values = archive["values"].astype(np.float32, copy=True)
        geometry_mask = archive["mask"].astype(bool, copy=True)
    if geometry_values.ndim != 2 or geometry_mask.shape != geometry_values.shape[:1]:
        raise ValueError("geometry values and mask are not aligned")
    return {
        "frame_index": frame_index,
        "task": frame["task"],
        "images": images,
        "camera_names": list(cameras),
        "state6": robocasa_state6(arrays),
        "oracle_action12": robocasa_action12(arrays),
        "geometry_values": geometry_values,
        "geometry_mask": geometry_mask,
        "phase": frame.get("action_source", {}).get("phase"),
    }


def policy_action_to_robocasa(_action: np.ndarray) -> np.ndarray:
    """Reject unsafe direct execution of the robot-specific pretrained output."""
    raise EmbodimentMismatchError(
        "Direct control is disabled: the checkpoint predicts normalized SO-100 "
        "actions, not RoboCasa's 12D Cartesian/mobile-base action. Fine-tune a "
        "RoboCasa-native state/action head before closed-loop execution."
    )


def _policy_frame(sample: dict, torch) -> dict:
    result = {
        "observation.state": torch.from_numpy(sample["state6"]),
        "task": sample["task"],
    }
    for index, image in enumerate(sample["images"], start=1):
        result[f"observation.images.camera{index}"] = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
    return result


def run_offline_replay(
    episode_dir: str | Path,
    memory_dir: str | Path,
    checkpoint: str | Path,
    frame_indices: list[int],
    *,
    device: str = "cuda",
) -> dict:
    """Run base and untrained geometry-conditioned policies without environment control."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from .geometry_fusion import (
        GEOMETRY_MASK,
        GEOMETRY_VALUES,
        GeometryConditionedSmolVLA,
    )

    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"SmolVLA checkpoint not found: {checkpoint}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    episode = RoboCasaEpisode(episode_dir)
    if not frame_indices:
        raise ValueError("at least one frame index is required")
    if any(index < 0 or index >= len(episode) for index in frame_indices):
        raise IndexError("frame index is outside the recorded episode")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    results = []
    geometry_policy = None
    for frame_index in frame_indices:
        sample = load_replay_sample(episode, memory_dir, frame_index)
        batch = preprocess(_policy_frame(sample, torch))
        torch.manual_seed(0)
        if device == "cuda":
            torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            base = postprocess(policy.predict_action_chunk(batch)).detach().cpu().numpy()

        if geometry_policy is None:
            torch.manual_seed(1)
            geometry_policy = GeometryConditionedSmolVLA(
                policy, input_dim=sample["geometry_values"].shape[1]
            ).to(device).eval()
        geometry_batch = dict(batch)
        geometry_batch[GEOMETRY_VALUES] = torch.from_numpy(
            sample["geometry_values"]
        ).unsqueeze(0).to(device)
        geometry_batch[GEOMETRY_MASK] = torch.from_numpy(
            sample["geometry_mask"]
        ).unsqueeze(0).to(device)
        torch.manual_seed(0)
        if device == "cuda":
            torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            geometry = postprocess(
                geometry_policy.predict_action_chunk(geometry_batch)
            ).detach().cpu().numpy()

        base = np.squeeze(base, axis=0)
        geometry = np.squeeze(geometry, axis=0)
        if not np.isfinite(base).all() or not np.isfinite(geometry).all():
            raise RuntimeError("SmolVLA returned a non-finite action")
        results.append(
            {
                "frame_index": frame_index,
                "phase": sample["phase"],
                "camera_names": sample["camera_names"],
                "image_shapes": [list(image.shape) for image in sample["images"]],
                "state6": sample["state6"].tolist(),
                "valid_geometry_tokens": int(sample["geometry_mask"].sum()),
                "oracle_action12": sample["oracle_action12"].tolist(),
                "base_chunk_shape": list(base.shape),
                "base_first_action6": base[0].tolist(),
                "geometry_first_action6": geometry[0].tolist(),
                "base_geometry_first_action_l2": float(np.linalg.norm(base[0] - geometry[0])),
            }
        )

    return {
        "status": "offline_interface_verified",
        "episode": str(Path(episode_dir).resolve()),
        "checkpoint": str(checkpoint),
        "device": device,
        "task": episode.manifest["task"],
        "frames": results,
        "geometry_connector_trained": False,
        "closed_loop_control_enabled": False,
        "embodiment_warning": (
            "Predicted 6D values retain SO-100 semantics and cannot be compared "
            "dimension-wise with or executed as the recorded RoboCasa action12."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default=Path("models/smolvla_base"), type=Path)
    parser.add_argument("--frames", nargs="+", type=int, default=[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_offline_replay(
        args.episode_dir, args.memory_dir, args.checkpoint, args.frames, device=args.device
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
