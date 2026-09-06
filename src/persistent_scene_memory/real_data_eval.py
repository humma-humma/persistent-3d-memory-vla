"""Evaluate a local SmolVLA checkpoint against recorded SO-100 actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def action_metrics(prediction: np.ndarray, target: np.ndarray, action_std: np.ndarray) -> dict:
    """Compute raw and scale-normalized open-loop action errors."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    action_std = np.asarray(action_std, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have the same [time, action] shape")
    if action_std.shape != (prediction.shape[1],) or np.any(action_std <= 0):
        raise ValueError("action_std must be positive and match the action dimension")
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.sqrt(np.mean((error / action_std) ** 2))),
        "per_dimension_mae": np.mean(np.abs(error), axis=0).tolist(),
    }


def blend_with_hold(
    prediction: np.ndarray, hold: np.ndarray, alpha: float | list[float] | np.ndarray
) -> np.ndarray:
    """Interpret SmolVLA output as a bounded correction to hold-state."""
    prediction = np.asarray(prediction, dtype=np.float64)
    hold = np.asarray(hold, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("hold blend alpha must be between 0 and 1")
    if prediction.shape != hold.shape:
        raise ValueError("prediction and hold must have identical shapes")
    if alpha.ndim > 1 or (alpha.ndim == 1 and alpha.shape != (prediction.shape[-1],)):
        raise ValueError("hold blend alpha must be scalar or match the action dimension")
    return hold + alpha * (prediction - hold)


def _decode_frames(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    import av

    wanted = set(indices)
    result = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                result[index] = frame.to_ndarray(format="rgb24")
            if index >= max(wanted):
                break
    missing = wanted - result.keys()
    if missing:
        raise RuntimeError(f"video ended before frames {sorted(missing)}")
    return result


def run_evaluation(
    checkpoint: Path,
    dataset_dir: Path,
    anchors: list[int],
    device: str = "cuda",
    seed: int = 0,
    image_size: int | None = None,
    episodes: list[int] | None = None,
    hold_blend_alpha: float | list[float] | None = None,
    include_actions: bool = False,
) -> dict:
    import pandas as pd
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = checkpoint.resolve()
    dataset_dir = dataset_dir.resolve()
    data = pd.read_parquet(dataset_dir / "data/chunk-000/file-000.parquet")
    episodes = episodes or [0]
    episode_tables = {
        episode_index: data[data["episode_index"] == episode_index]
        .sort_values("frame_index")
        .reset_index(drop=True)
        for episode_index in episodes
    }
    if any(table.empty for table in episode_tables.values()):
        raise ValueError("every requested evaluation episode must exist")
    task_table = pd.read_parquet(dataset_dir / "meta/tasks.parquet")
    tasks = {
        episode_index: str(task_table.index[int(table.iloc[0]["task_index"])])
        for episode_index, table in episode_tables.items()
    }
    stats = json.loads((dataset_dir / "meta/stats.json").read_text(encoding="utf-8"))
    action_std = np.asarray(stats["action"]["std"], dtype=np.float64)

    horizon = 50
    if any(
        anchor < 0 or anchor + horizon > len(table)
        for table in episode_tables.values()
        for anchor in anchors
    ):
        raise ValueError(
            f"anchors must leave a {horizon}-frame horizon inside every requested episode"
        )
    specs = [
        (episode_index, anchor, int(table.iloc[anchor]["index"]))
        for episode_index, table in episode_tables.items()
        for anchor in anchors
    ]
    video_indices = [global_index for _, _, global_index in specs]
    top_frames = _decode_frames(
        dataset_dir / "videos/observation.images.top/chunk-000/file-000.mp4", video_indices
    )
    wrist_frames = _decode_frames(
        dataset_dir / "videos/observation.images.wrist/chunk-000/file-000.mp4", video_indices
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    if image_size is not None:
        config.resize_imgs_with_padding = (image_size, image_size)
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=stats)

    results = []
    for episode_index, anchor, global_index in specs:
        episode = episode_tables[episode_index]
        sample_seed = seed + episode_index * 10_000 + anchor
        torch.manual_seed(sample_seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(sample_seed)
        row = episode.iloc[anchor]
        frame = {
            "observation.state": torch.from_numpy(
                np.array(row["observation.state"], dtype=np.float32, copy=True)
            ),
            "observation.images.camera1": torch.from_numpy(top_frames[global_index].copy()).permute(2, 0, 1).float() / 255.0,
            "observation.images.camera2": torch.from_numpy(wrist_frames[global_index].copy()).permute(2, 0, 1).float() / 255.0,
            "task": tasks[episode_index],
        }
        batch = preprocess(frame)
        started = time.perf_counter()
        with torch.inference_mode():
            predicted = postprocess(policy.predict_action_chunk(batch))
        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        prediction = predicted[0].detach().cpu().numpy()
        target = np.stack(episode.iloc[anchor : anchor + horizon]["action"].to_numpy())
        hold = np.repeat(np.asarray(row["observation.state"])[None, :], horizon, axis=0)
        item = {
                "episode_index": episode_index,
                "anchor_frame": anchor,
                "inference_seconds": inference_seconds,
                "smolvla": action_metrics(prediction, target, action_std),
                "hold_state_baseline": action_metrics(hold, target, action_std),
                "first_predicted_action": prediction[0].tolist(),
                "first_recorded_action": target[0].tolist(),
            }
        if hold_blend_alpha is not None:
            blended = blend_with_hold(prediction, hold, hold_blend_alpha)
            item["residual_blend"] = action_metrics(blended, target, action_std)
        if include_actions:
            item["predicted_actions"] = prediction.tolist()
            item["recorded_actions"] = target.tolist()
            item["hold_actions"] = hold.tolist()
        results.append(item)

    aggregate = {}
    policy_names = ["smolvla", "hold_state_baseline"]
    if hold_blend_alpha is not None:
        policy_names.append("residual_blend")
    for name in policy_names:
        aggregate[name] = {
            metric: float(np.mean([item[name][metric] for item in results]))
            for metric in ("mae", "rmse", "normalized_rmse")
        }
    per_episode = {}
    for episode_index in episodes:
        episode_results = [item for item in results if item["episode_index"] == episode_index]
        per_episode[str(episode_index)] = {
            name: {
                metric: float(np.mean([item[name][metric] for item in episode_results]))
                for metric in ("mae", "rmse", "normalized_rmse")
            }
            for name in policy_names
        }
    return {
        "checkpoint": str(checkpoint),
        "dataset": "lerobot/svla_so100_pickplace",
        "dataset_dir": str(dataset_dir),
        "episodes": episodes,
        "tasks": tasks,
        "anchors": anchors,
        "horizon": horizon,
        "image_size": list(config.resize_imgs_with_padding),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0,
        "aggregate": aggregate,
        "per_episode": per_episode,
        "results": results,
        "hold_blend_alpha": hold_blend_alpha,
        "status": "ok",
        "interpretation": "Open-loop recorded-action diagnostic; not closed-loop task performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/smolvla_base"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/svla_so100_pickplace"))
    parser.add_argument("--anchors", type=int, nargs="+", default=[0, 50, 100])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--hold-blend-alpha", type=float, nargs="+")
    parser.add_argument("--include-actions", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/smolvla_real_data_baseline.json"))
    args = parser.parse_args()
    result = run_evaluation(
        args.checkpoint,
        args.dataset_dir,
        args.anchors,
        args.device,
        args.seed,
        args.image_size,
        args.episodes,
        args.hold_blend_alpha[0]
        if args.hold_blend_alpha is not None and len(args.hold_blend_alpha) == 1
        else args.hold_blend_alpha,
        args.include_actions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
