"""Held-out open-loop evaluation for a RoboCasa-native SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .robocasa_finetune import episode_anchors, load_episode_vectors
from .robocasa_rollout import RoboCasaEpisode, _slug
from .robocasa_smolvla_replay import ACTION_FIELDS, robocasa_state16


ACTION_SLICES = {
    "end_effector_position": slice(0, 3),
    "end_effector_rotation": slice(3, 6),
    "gripper_close": slice(6, 7),
    "base_motion": slice(7, 11),
    "control_mode": slice(11, 12),
}


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Compare predicted actions against recorded actions and a zero-action baseline."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 2 or target.shape[1] != 12:
        raise ValueError("prediction and target must have matching shape [steps, 12]")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values**2)))

    residual = prediction - target
    result = {
        "policy_rmse": rmse(residual),
        "zero_action_rmse": rmse(target),
        "beats_zero_action": rmse(residual) < rmse(target),
        "per_field": {},
    }
    for name, field_slice in ACTION_SLICES.items():
        result["per_field"][name] = {
            "policy_rmse": rmse(residual[:, field_slice]),
            "zero_action_rmse": rmse(target[:, field_slice]),
        }
    return result


def blend_predictions(
    primary: np.ndarray, secondary: np.ndarray, secondary_fields: list[str]
) -> np.ndarray:
    """Replace selected action fields in primary with secondary predictions."""
    primary = np.asarray(primary, dtype=np.float32)
    secondary = np.asarray(secondary, dtype=np.float32)
    if primary.shape != secondary.shape or primary.ndim != 2 or primary.shape[1] != 12:
        raise ValueError("primary and secondary predictions must match shape [steps, 12]")
    unknown = sorted(set(secondary_fields) - set(ACTION_SLICES))
    if unknown:
        raise ValueError(f"unknown action fields: {unknown}")
    blended = primary.copy()
    for field in secondary_fields:
        blended[:, ACTION_SLICES[field]] = secondary[:, ACTION_SLICES[field]]
    return blended


def _training_seeds(checkpoint: Path) -> list[int]:
    report_path = checkpoint / "robocasa_finetune_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"checkpoint has no training report: {report_path}")
    return list(json.loads(report_path.read_text(encoding="utf-8"))["training_seeds"])


def evaluate_checkpoint(
    checkpoint: str | Path,
    episode_dir: str | Path,
    *,
    stride: int = 25,
    horizon: int = 50,
    device: str = "cuda",
    seed: int = 23,
    predictions_output: str | Path | None = None,
) -> dict:
    """Evaluate action chunks on a successful simulator seed excluded from training."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint).resolve()
    episode = RoboCasaEpisode(episode_dir)
    heldout_seed = episode.manifest.get("seed")
    training_seeds = _training_seeds(checkpoint)
    if heldout_seed in training_seeds:
        raise ValueError(f"episode seed {heldout_seed} was used for training")
    if not any(frame.get("success") for frame in episode.manifest["frames"]):
        raise ValueError("held-out episode must be a successful demonstration")
    anchors = episode_anchors(len(episode), horizon=horizon, stride=stride)
    if not anchors:
        raise ValueError("held-out episode is shorter than the action horizon")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    states, actions = load_episode_vectors(episode)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    config.chunk_size = horizon
    config.n_action_steps = horizon
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    predictions, targets = [], []
    for anchor in anchors:
        arrays = episode.frame(anchor)["arrays"]
        batch = {
            "observation.state": torch.from_numpy(robocasa_state16(arrays)),
            "task": episode.manifest["task"],
        }
        for camera_index, camera in enumerate(episode.manifest["cameras"], start=1):
            image = np.ascontiguousarray(arrays[f"rgb__{_slug(camera)}"])
            batch[f"observation.images.camera{camera_index}"] = (
                torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            )
        batch = preprocess(batch)
        torch.manual_seed(seed + anchor)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed + anchor)
        with torch.inference_mode():
            chunk = postprocess(policy.predict_action_chunk(batch)).detach().cpu().numpy()
        chunk = np.squeeze(chunk, axis=0)
        predictions.append(chunk[:horizon])
        targets.append(actions[anchor : anchor + horizon])

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    if predictions_output is not None:
        predictions_output = Path(predictions_output)
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_output, prediction=prediction, target=target)
    metrics = prediction_metrics(prediction, target)
    return {
        "status": "heldout_open_loop_complete",
        "checkpoint": str(checkpoint),
        "episode": str(Path(episode_dir).resolve()),
        "training_seeds": training_seeds,
        "heldout_seed": heldout_seed,
        "task": episode.manifest["task"],
        "anchors": anchors,
        "evaluated_actions": int(target.shape[0]),
        "action_fields": list(ACTION_FIELDS),
        **metrics,
        "closed_loop_validated": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint,
        args.episode_dir,
        stride=args.stride,
        horizon=args.horizon,
        device=args.device,
        seed=args.seed,
        predictions_output=args.predictions_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
