"""Bounded SmolVLA adaptation to recorded RoboCasa state and action semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .fine_tune_pilot import select_adaptation_parameters
from .robocasa_rollout import RoboCasaEpisode, _slug
from .robocasa_smolvla_replay import robocasa_action12, robocasa_state16


ACTION_FIELD_INDICES = {
    "end_effector_position": (0, 1, 2),
    "end_effector_rotation": (3, 4, 5),
    "gripper_close": (6,),
    "base_motion": (7, 8, 9, 10),
    "control_mode": (11,),
}


def selected_action_indices(fields: list[str]) -> list[int]:
    """Return sorted native action indices for a unique list of field names."""
    unknown = sorted(set(fields) - set(ACTION_FIELD_INDICES))
    if unknown:
        raise ValueError(f"unknown action fields: {unknown}")
    if len(set(fields)) != len(fields):
        raise ValueError("action fields must be unique")
    return sorted(index for field in fields for index in ACTION_FIELD_INDICES[field])


def action_event_label(actions: np.ndarray) -> str:
    """Label an action chunk by the switching behavior targeted in training."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) == 0:
        raise ValueError("actions must have shape [steps, 12]")
    gripper_closed = actions[:, 6] >= 0.5
    control_active = actions[:, 11] >= 0.5
    if np.any(np.diff(gripper_closed)):
        return "gripper_transition"
    if not np.any(gripper_closed):
        return "gripper_open"
    if np.any(np.diff(control_active)):
        return "control_transition"
    return "steady_closed"


def episode_anchors(frame_count: int, horizon: int = 50, stride: int = 10) -> list[int]:
    """Return causal action-chunk anchors that stay within an episode."""
    if frame_count < 1 or horizon < 1 or stride < 1:
        raise ValueError("frame_count, horizon, and stride must be positive")
    last = frame_count - horizon
    if last < 0:
        return []
    anchors = list(range(0, last + 1, stride))
    if anchors[-1] != last:
        anchors.append(last)
    return anchors


def available_episode_anchors(
    frame_count: int,
    horizon: int,
    stride: int,
    training_cache: dict | None = None,
) -> list[int]:
    """Restrict requested anchors to RGB frames declared by a sparse cache."""
    requested = episode_anchors(frame_count, horizon, stride)
    if not training_cache:
        return requested
    cached = set(
        episode_anchors(
            frame_count,
            horizon=int(training_cache["horizon"]),
            stride=int(training_cache["stride"]),
        )
    )
    return [anchor for anchor in requested if anchor in cached]


def phase_balanced_example_order(phases: list[str], steps: int, seed: int) -> list[int]:
    """Sample phases uniformly while shuffling examples within each phase."""
    if not phases or steps < 1 or any(not phase for phase in phases):
        raise ValueError("phases must be nonempty strings and steps must be positive")
    groups = {
        phase: np.flatnonzero(np.asarray(phases) == phase).tolist()
        for phase in sorted(set(phases))
    }
    rng = np.random.default_rng(seed)
    queues = {phase: [] for phase in groups}
    order = []
    while len(order) < steps:
        for phase in rng.permutation(list(groups)):
            if not queues[phase]:
                queues[phase] = rng.permutation(groups[phase]).tolist()
            order.append(queues[phase].pop())
            if len(order) == steps:
                break
    return order


def load_episode_vectors(episode: RoboCasaEpisode) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned full-state and applied-action arrays from one episode."""
    states, actions = [], []
    for index in range(len(episode)):
        arrays = episode.frame(index)["arrays"]
        states.append(robocasa_state16(arrays))
        actions.append(robocasa_action12(arrays))
    return np.stack(states), np.stack(actions)


def validate_demonstrations(episodes: list[RoboCasaEpisode]) -> list[dict]:
    """Reject failed or duplicate-seed rollouts before policy training."""
    if not episodes:
        raise ValueError("at least one episode is required")
    records = []
    identifiers = []
    for episode in episodes:
        successful_frames = [
            frame["frame_index"] for frame in episode.manifest["frames"] if frame.get("success")
        ]
        if not successful_frames:
            raise ValueError(f"training episode has no successful frame: {episode.root}")
        seed = episode.manifest.get("seed")
        demonstration_id = episode.manifest.get("demonstration_id")
        if seed is None and demonstration_id is None:
            raise ValueError(
                f"training episode has no seed or demonstration id: {episode.root}"
            )
        identifier = ("seed", seed) if seed is not None else ("id", demonstration_id)
        identifiers.append(identifier)
        records.append(
            {
                "episode": str(episode.root.resolve()),
                "seed": seed,
                "demonstration_id": demonstration_id,
                "first_success_frame": successful_frames[0],
            }
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("training episodes must use distinct identifiers")
    return records


def dataset_statistics(states: np.ndarray, actions: np.ndarray) -> dict:
    """Compute LeRobot mean/std/min/max statistics for native features."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError("states must have shape [frames, 16]")
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError("actions must have shape [frames, 12]")

    def feature_stats(values: np.ndarray) -> dict:
        return {
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
        }

    return {
        "observation.state": feature_stats(states),
        "action": feature_stats(actions),
    }


def _image(arrays: dict[str, np.ndarray], camera: str) -> np.ndarray:
    return np.ascontiguousarray(arrays[f"rgb__{_slug(camera)}"])


def run_finetune(
    checkpoint: str | Path,
    episode_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    steps: int,
    stride: int = 10,
    horizon: int = 50,
    image_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    seed: int = 17,
    expert_layers: int = 0,
    target_fields: list[str] | None = None,
    preserve_checkpoint_stats: bool = False,
    sampling_balance: str = "phase",
) -> dict:
    """Fine-tune projections for a native 16D-state/12D-action RoboCasa contract."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if steps < 1 or image_size < 64:
        raise ValueError("steps must be positive and image_size must be at least 64")
    if not episode_dirs:
        raise ValueError("at least one episode directory is required")
    if sampling_balance not in {"phase", "action_event"}:
        raise ValueError("sampling_balance must be 'phase' or 'action_event'")
    target_fields = list(target_fields or [])
    target_indices = selected_action_indices(target_fields)
    if target_fields and not preserve_checkpoint_stats:
        raise ValueError("targeted output training requires preserved checkpoint statistics")
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    episodes = [RoboCasaEpisode(path) for path in episode_dirs]
    demonstration_records = validate_demonstrations(episodes)
    tasks = {episode.manifest["task"] for episode in episodes}
    if not tasks or any(not task for task in tasks):
        raise ValueError("all training episodes must have a language task")
    if any(len(episode.manifest["cameras"]) != 3 for episode in episodes):
        raise ValueError("the current SmolVLA configuration requires exactly three cameras")

    episode_data = []
    all_states, all_actions = [], []
    for episode in episodes:
        states, actions = load_episode_vectors(episode)
        anchors = available_episode_anchors(
            len(episode), horizon, stride, episode.manifest.get("training_cache")
        )
        if not anchors:
            raise ValueError(f"episode {episode.root} is shorter than the action horizon")
        episode_data.append((episode, states, actions, anchors))
        all_states.append(states)
        all_actions.append(actions)
    stats_numpy = dataset_statistics(np.concatenate(all_states), np.concatenate(all_actions))
    stats = {
        key: {name: torch.from_numpy(value) for name, value in values.items()}
        for key, values in stats_numpy.items()
    }

    examples = []
    for episode, states, actions, anchors in episode_data:
        for anchor in anchors:
            arrays = episode.frame(anchor)["arrays"]
            examples.append(
                {
                    "observation.state": torch.from_numpy(states[anchor].copy()),
                    "observation.images.camera1": _image(arrays, episode.manifest["cameras"][0]),
                    "observation.images.camera2": _image(arrays, episode.manifest["cameras"][1]),
                    "observation.images.camera3": _image(arrays, episode.manifest["cameras"][2]),
                    "action": torch.from_numpy(actions[anchor : anchor + horizon].copy()),
                    "task": episode.manifest["task"],
                    "episode": str(episode.root.resolve()),
                    "anchor": anchor,
                    "phase": episode.manifest["frames"][anchor]
                    .get("action_source", {})
                    .get("phase", "unknown"),
                    "action_event": action_event_label(actions[anchor : anchor + horizon]),
                }
            )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    config.push_to_hub = False
    config.resize_imgs_with_padding = (image_size, image_size)
    config.chunk_size = horizon
    config.n_action_steps = horizon
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (16,)),
        "observation.images.camera1": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "observation.images.camera2": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "observation.images.camera3": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (12,))}
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device)
    if target_fields:
        for parameter in policy.parameters():
            parameter.requires_grad = False
        output_projection = policy.model.action_out_proj
        output_projection.weight.requires_grad = True
        output_projection.bias.requires_grad = True
        row_mask = torch.zeros(
            output_projection.out_features, dtype=output_projection.weight.dtype, device=device
        )
        row_mask[target_indices] = 1
        output_projection.weight.register_hook(lambda grad: grad * row_mask[:, None])
        output_projection.bias.register_hook(lambda grad: grad * row_mask)
        parameters = [output_projection.weight, output_projection.bias]
        parameter_names = ["model.action_out_proj.weight", "model.action_out_proj.bias"]
    else:
        parameters, parameter_names = select_adaptation_parameters(policy, expert_layers)
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=0.0 if target_fields else 1e-10
    )
    if preserve_checkpoint_stats:
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    else:
        preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=stats)

    def materialize(example: dict) -> dict:
        batch = {
            key: value
            for key, value in example.items()
            if key not in {"episode", "anchor", "phase", "action_event"}
        }
        for camera_index in range(1, 4):
            key = f"observation.images.camera{camera_index}"
            batch[key] = torch.from_numpy(batch[key]).permute(2, 0, 1).float() / 255.0
        return batch

    probe = preprocess(materialize(examples[0]))
    if probe["action"].ndim == 2:
        probe["action"] = probe["action"].unsqueeze(0)
    fixed_noise = torch.zeros(
        (1, horizon, policy.config.max_action_dim), device=device
    )
    fixed_time = torch.full((1,), 0.5, device=device)
    policy.eval()
    with torch.inference_mode():
        probe_loss_before = float(
            policy(probe, noise=fixed_noise, time=fixed_time)[0].detach().cpu()
        )

    policy.train()
    losses = []
    started = time.perf_counter()
    training_order = phase_balanced_example_order(
        [example[sampling_balance] for example in examples], steps, seed
    )
    for example_index in training_order:
        optimizer.zero_grad(set_to_none=True)
        batch = preprocess(materialize(examples[example_index]))
        if batch["action"].ndim == 2:
            batch["action"] = batch["action"].unsqueeze(0)
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite RoboCasa fine-tuning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    policy.eval()
    with torch.inference_mode():
        probe_loss_after = float(
            policy(probe, noise=fixed_noise, time=fixed_time)[0].detach().cpu()
        )
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        prediction = postprocess(policy.predict_action_chunk(probe)).detach().cpu().numpy()
    prediction = np.squeeze(prediction, axis=0)
    target = examples[0]["action"].numpy()
    if prediction.shape != (horizon, 12) or not np.isfinite(prediction).all():
        raise RuntimeError(f"native checkpoint returned invalid action chunk {prediction.shape}")
    prediction_rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    zero_action_rmse = float(np.sqrt(np.mean(target**2)))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocess.save_pretrained(output_dir)
    postprocess.save_pretrained(output_dir)
    report = {
        "status": "training_smoke_complete",
        "base_checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "training_episodes": [str(episode.root.resolve()) for episode in episodes],
        "training_seeds": [
            episode.manifest.get("seed")
            for episode in episodes
            if episode.manifest.get("seed") is not None
        ],
        "training_demonstration_ids": [
            episode.manifest.get("demonstration_id")
            for episode in episodes
            if episode.manifest.get("demonstration_id") is not None
        ],
        "training_tasks": sorted(tasks),
        "demonstrations": demonstration_records,
        "training_examples": len(examples),
        "training_examples_by_phase": dict(
            sorted(Counter(example["phase"] for example in examples).items())
        ),
        "training_examples_by_action_event": dict(
            sorted(Counter(example["action_event"] for example in examples).items())
        ),
        "training_updates_by_phase": dict(
            sorted(Counter(examples[index]["phase"] for index in training_order).items())
        ),
        "training_updates_by_sampling_group": dict(
            sorted(
                Counter(examples[index][sampling_balance] for index in training_order).items()
            )
        ),
        "sampling_balance": sampling_balance,
        "horizon": horizon,
        "stride": stride,
        "steps": steps,
        "state_dimension": 16,
        "action_dimension": 12,
        "action_fields": [
            "end_effector_position[3]",
            "end_effector_rotation[3]",
            "gripper_close[1]",
            "base_motion[4]",
            "control_mode[1]",
        ],
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_tensor_names": parameter_names,
        "target_fields": target_fields,
        "target_action_indices": target_indices,
        "preserved_checkpoint_statistics": preserve_checkpoint_stats,
        "initial_training_loss": losses[0],
        "final_training_loss": losses[-1],
        "probe_loss_before": probe_loss_before,
        "probe_loss_after": probe_loss_after,
        "prediction_shape": list(prediction.shape),
        "first_predicted_action": prediction[0].tolist(),
        "first_recorded_action": target[0].tolist(),
        "training_probe_prediction_rmse": prediction_rmse,
        "training_probe_zero_action_rmse": zero_action_rmse,
        "elapsed_seconds": elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0,
        "closed_loop_validated": False,
        "scientific_interpretation": (
            f"Training-set diagnostic on {len(episodes)} distinct demonstrations. "
            "Held-out-seed task success and hold-action comparisons are required "
            "before policy promotion."
        ),
    }
    (output_dir / "robocasa_finetune_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/smolvla_base"))
    parser.add_argument("--episode-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expert-layers", type=int, default=0)
    parser.add_argument(
        "--target-field", action="append", choices=tuple(ACTION_FIELD_INDICES)
    )
    parser.add_argument("--preserve-checkpoint-stats", action="store_true")
    parser.add_argument(
        "--sampling-balance", choices=("phase", "action_event"), default="phase"
    )
    args = parser.parse_args(argv)
    result = run_finetune(
        args.checkpoint,
        args.episode_dir,
        args.output_dir,
        steps=args.steps,
        stride=args.stride,
        horizon=args.horizon,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        expert_layers=args.expert_layers,
        target_fields=args.target_field,
        preserve_checkpoint_stats=args.preserve_checkpoint_stats,
        sampling_balance=args.sampling_balance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
