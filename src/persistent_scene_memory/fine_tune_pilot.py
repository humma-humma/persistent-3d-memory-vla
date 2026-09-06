"""Low-memory SmolVLA projection-layer adaptation pilot on recorded SO-100 data."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from .real_data_eval import _decode_frames

TRAINABLE_PARTS = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)
HELD_OUT_EPISODES = (0, 11, 12, 13, 14, 15)
DEFAULT_TRAINING_EPISODES = tuple(range(1, 11)) + tuple(range(16, 50))


def select_projection_parameters(policy) -> tuple[list, list[str]]:
    """Freeze the policy and return the small task-specific projection parameter set."""
    selected = []
    names = []
    for name, parameter in policy.named_parameters():
        parameter.requires_grad = any(part in name for part in TRAINABLE_PARTS)
        if parameter.requires_grad:
            selected.append(parameter)
            names.append(name)
    if not selected:
        raise RuntimeError("no SmolVLA projection parameters matched")
    return selected, names


def select_adaptation_parameters(
    policy, expert_layers: int = 0
) -> tuple[list, list[str]]:
    """Select projections and, optionally, the final action-expert layers."""
    if expert_layers < 0:
        raise ValueError("expert_layers must be non-negative")
    named_parameters = list(policy.named_parameters())
    layer_indices = [
        int(match.group(1))
        for name, _ in named_parameters
        if (match := re.search(r"\.lm_expert\.layers\.(\d+)\.", name))
    ]
    if expert_layers and not layer_indices:
        raise RuntimeError("no SmolVLA action-expert layers matched")
    first_trainable_layer = (
        max(layer_indices) - expert_layers + 1 if expert_layers else None
    )
    selected = []
    names = []
    for name, parameter in named_parameters:
        match = re.search(r"\.lm_expert\.layers\.(\d+)\.", name)
        train_expert = (
            first_trainable_layer is not None
            and match is not None
            and int(match.group(1)) >= first_trainable_layer
        )
        train_norm = expert_layers > 0 and ".lm_expert.norm." in name
        parameter.requires_grad = (
            any(part in name for part in TRAINABLE_PARTS)
            or train_expert
            or train_norm
        )
        if parameter.requires_grad:
            selected.append(parameter)
            names.append(name)
    if not selected:
        raise RuntimeError("no SmolVLA adaptation parameters matched")
    return selected, names


def balanced_example_order(example_count: int, steps: int, seed: int) -> list[int]:
    """Return deterministic shuffled epochs without favoring early episodes."""
    if example_count < 1 or steps < 1:
        raise ValueError("example_count and steps must be positive")
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(rng.permutation(example_count).tolist())
    return order[:steps]


def run_pilot(
    checkpoint: Path,
    dataset_dir: Path,
    output_dir: Path,
    anchors: list[int],
    episodes: list[int],
    steps: int,
    image_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    seed: int = 17,
    expert_layers: int = 0,
) -> dict:
    import pandas as pd
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = checkpoint.resolve()
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if steps < 1 or image_size < 64:
        raise ValueError("steps must be positive and image_size must be at least 64")
    overlap = sorted(set(episodes) & set(HELD_OUT_EPISODES))
    if overlap:
        raise ValueError(f"training episodes overlap the frozen evaluation set: {overlap}")
    data = pd.read_parquet(dataset_dir / "data/chunk-000/file-000.parquet")
    episode_tables = {
        episode_index: data[data["episode_index"] == episode_index]
        .sort_values("frame_index")
        .reset_index(drop=True)
        for episode_index in episodes
    }
    if any(table.empty for table in episode_tables.values()):
        raise ValueError("every requested training episode must exist")
    if any(anchor < 0 or anchor + 50 > len(table) for table in episode_tables.values() for anchor in anchors):
        raise ValueError("every training anchor must leave a 50-action horizon")
    task_table = pd.read_parquet(dataset_dir / "meta/tasks.parquet")
    task = str(task_table.index[int(next(iter(episode_tables.values())).iloc[0]["task_index"])])
    stats = json.loads((dataset_dir / "meta/stats.json").read_text(encoding="utf-8"))
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
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    config.resize_imgs_with_padding = (image_size, image_size)
    config.push_to_hub = False
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device)
    parameters, parameter_names = select_adaptation_parameters(policy, expert_layers)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-10)
    preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=stats)

    examples = []
    for episode_index, anchor, global_index in specs:
        episode = episode_tables[episode_index]
        row = episode.iloc[anchor]
        actions = np.stack(episode.iloc[anchor : anchor + 50]["action"].to_numpy())
        examples.append(
            {
                "observation.state": torch.from_numpy(
                    np.array(row["observation.state"], dtype=np.float32, copy=True)
                ),
                "observation.images.camera1": top_frames[global_index],
                "observation.images.camera2": wrist_frames[global_index],
                "action": torch.from_numpy(actions.astype(np.float32, copy=True)),
                "task": task,
            }
        )

    def materialize(example: dict) -> dict:
        return {
            **example,
            "observation.images.camera1": torch.from_numpy(
                example["observation.images.camera1"]
            ).permute(2, 0, 1).float()
            / 255.0,
            "observation.images.camera2": torch.from_numpy(
                example["observation.images.camera2"]
            ).permute(2, 0, 1).float()
            / 255.0,
        }

    policy.train()
    losses = []
    started = time.perf_counter()
    training_order = balanced_example_order(len(examples), steps, seed)
    for step, example_index in enumerate(training_order):
        optimizer.zero_grad(set_to_none=True)
        batch = preprocess(materialize(examples[example_index]))
        if batch["action"].ndim == 2:
            batch["action"] = batch["action"].unsqueeze(0)
        loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocess.save_pretrained(output_dir)
    postprocess.save_pretrained(output_dir)
    report = {
        "base_checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "dataset": "lerobot/svla_so100_pickplace",
        "training_episodes": episodes,
        "training_anchors": anchors,
        "training_examples": len(examples),
        "frozen_evaluation_episodes": list(HELD_OUT_EPISODES),
        "sampling": "deterministic shuffled epochs",
        "steps": steps,
        "image_size": image_size,
        "learning_rate": learning_rate,
        "expert_layers": expert_layers,
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_tensor_names": parameter_names,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "mean_first_50_loss": float(np.mean(losses[:50])),
        "mean_last_50_loss": float(np.mean(losses[-50:])),
        "elapsed_seconds": elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0,
        "status": "ok",
        "scope": (
            "Projection layers plus bounded final action-expert layers; "
            "VLM remains frozen."
        ),
    }
    (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/smolvla_base"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/svla_so100_pickplace"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smolvla_projection_pilot"))
    parser.add_argument(
        "--anchors", type=int, nargs="+", default=[0, 50, 100, 150, 200, 250]
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=list(DEFAULT_TRAINING_EPISODES)
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expert-layers", type=int, default=0)
    args = parser.parse_args()
    report = run_pilot(
        args.checkpoint,
        args.dataset_dir,
        args.output_dir,
        args.anchors,
        args.episodes,
        args.steps,
        args.image_size,
        args.learning_rate,
        args.device,
        args.seed,
        args.expert_layers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
