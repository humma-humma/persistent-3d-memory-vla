"""Extract frozen SmolVLA context features for hierarchical skill learning."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .robocasa_hierarchical import SKILLS, phase_to_skill
from .robocasa_rollout import RoboCasaEpisode, _slug
from .robocasa_smolvla_replay import robocasa_state16


def extract_skill_features(
    checkpoint: str | Path,
    episode_dirs: list[str | Path],
    output: str | Path,
    *,
    stride: int = 4,
    device: str = "cuda",
) -> dict:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        make_att_2d_masks,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    if stride < 1 or not episode_dirs:
        raise ValueError("stride must be positive and episodes must be nonempty")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    checkpoint = Path(checkpoint).resolve()
    episodes = [RoboCasaEpisode(path) for path in episode_dirs]
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device).eval()
    preprocess, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    features, labels, frames, sources = [], [], [], []
    for episode in episodes:
        frame_rows = episode.manifest["frames"]
        if any("rgb_available" in row for row in frame_rows):
            candidates = [
                int(row["frame_index"])
                for row in frame_rows
                if row.get("rgb_available", False)
            ][::stride]
        else:
            candidates = range(0, len(episode), stride)
        for frame_index in candidates:
            frame = episode.frame(frame_index)
            phase = frame.get("action_source", {}).get("phase")
            try:
                skill = phase_to_skill(phase)
            except ValueError:
                continue
            arrays = frame["arrays"]
            camera_keys = [f"rgb__{_slug(camera)}" for camera in episode.manifest["cameras"]]
            if not all(key in arrays for key in camera_keys):
                continue
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
            with torch.inference_mode():
                images, image_masks = policy.prepare_images(batch)
                state = policy.prepare_state(batch)
                language_tokens = batch[OBS_LANGUAGE_TOKENS]
                language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
                prefix, padding, attention = policy.model.embed_prefix(
                    images,
                    image_masks,
                    language_tokens,
                    language_masks,
                    state=state,
                )
                attention_2d = make_att_2d_masks(padding, attention)
                position_ids = torch.cumsum(padding, dim=1) - 1
                outputs, _ = policy.model.vlm_with_expert.forward(
                    attention_mask=attention_2d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix, None],
                    use_cache=False,
                    fill_kv_cache=True,
                )
                valid = padding.unsqueeze(-1).to(outputs[0].dtype)
                pooled = (outputs[0] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            features.append(pooled.squeeze(0).float().cpu().numpy())
            labels.append(SKILLS.index(skill))
            frames.append(frame_index)
            sources.append(str(episode.root.resolve()))

    if not features:
        raise ValueError("episodes contain no supported oracle skill labels")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        frame_indices=np.asarray(frames, dtype=np.int64),
        episode_roots=np.asarray(sources),
        skills=np.asarray(SKILLS),
    )
    counts = Counter(SKILLS[label] for label in labels)
    return {
        "status": "skill_features_complete",
        "checkpoint": str(checkpoint),
        "episodes": [str(episode.root.resolve()) for episode in episodes],
        "samples": len(features),
        "feature_dimension": int(features[0].shape[0]),
        "stride": stride,
        "skill_counts": dict(sorted(counts.items())),
        "output": str(output.resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episode-dir", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    result = extract_skill_features(
        args.checkpoint,
        args.episode_dir,
        args.output,
        stride=args.stride,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
