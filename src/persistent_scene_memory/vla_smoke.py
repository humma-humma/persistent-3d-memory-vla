"""Offline contract check and optional pretrained VLA inference smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .adapter import GeometryTokenAdapter
from .memory import GeometryObservation, PersistentSceneMemory


def build_smoke_input() -> tuple[np.ndarray, str, np.ndarray]:
    memory = PersistentSceneMemory()
    memory.update(GeometryObservation("red block", [0.42, -0.08, 0.03], [1.0, 0.0], 0.9, 0))
    tokens = memory.tokens(0)
    adapter = GeometryTokenAdapter(max_tokens=8)
    encoded = adapter.encode(tokens, 0)
    prompt = adapter.augment_instruction("Move the gripper to the red block.", tokens)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[80:145, 90:155, 0] = 255
    return image, prompt, encoded.values


def run_mock() -> dict:
    image, prompt, geometry = build_smoke_input()
    action = np.array([0.42, -0.08, 0.03, 0.0, 0.0, 0.0, 1.0])
    return {
        "backend": "mock",
        "image_shape": list(image.shape),
        "geometry_shape": list(geometry.shape),
        "prompt": prompt,
        "action": action.tolist(),
        "status": "ok",
    }


def run_openvla(model_id: str) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    image, prompt, geometry = build_smoke_input()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    inputs = processor(prompt, Image.fromarray(image)).to("cuda", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return {
        "backend": "openvla",
        "model_id": model_id,
        "geometry_shape": list(geometry.shape),
        "action": np.asarray(action).tolist(),
        "status": "ok",
    }


def run_smolvla(
    checkpoint: str | Path, device: str = "cuda", geometry_fusion: bool = False
) -> dict:
    """Run one real SmolVLA action prediction from a synthetic three-camera observation."""
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
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    image, prompt, geometry = build_smoke_input()
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    frame = {
        "observation.state": torch.zeros(6),
        "observation.images.camera1": image_tensor,
        "observation.images.camera2": image_tensor.clone(),
        "observation.images.camera3": image_tensor.clone(),
        "task": prompt,
    }

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    # The policy checkpoint already contains the VLM weights. Avoid downloading
    # and loading a redundant copy before applying the policy state dict.
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    ).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    batch = preprocess(frame)
    inference_policy = policy
    connector_parameters = 0
    if geometry_fusion:
        inference_policy = GeometryConditionedSmolVLA(
            policy, input_dim=geometry.shape[1]
        ).to(device).eval()
        batch[GEOMETRY_VALUES] = torch.from_numpy(geometry).unsqueeze(0).to(device)
        geometry_mask = torch.zeros((1, geometry.shape[0]), dtype=torch.bool, device=device)
        geometry_mask[:, 0] = True
        batch[GEOMETRY_MASK] = geometry_mask
        connector_parameters = sum(
            parameter.numel() for parameter in inference_policy.connector.parameters()
            if parameter.requires_grad
        )
    with torch.inference_mode():
        action = postprocess(inference_policy.select_action(batch))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    action_array = action.detach().cpu().numpy()
    if not np.all(np.isfinite(action_array)):
        raise RuntimeError("SmolVLA returned a non-finite action")
    return {
        "backend": "smolvla_geometry" if geometry_fusion else "smolvla",
        "checkpoint": str(checkpoint),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "geometry_shape": list(geometry.shape),
        "geometry_fusion": geometry_fusion,
        "connector_trainable_parameters": connector_parameters,
        "action_shape": list(action_array.shape),
        "action": action_array.tolist(),
        "elapsed_seconds_including_load": elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("mock", "smolvla", "smolvla_geometry", "openvla"),
        default="mock",
    )
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/smolvla_base"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend == "mock":
        result = run_mock()
    elif args.backend in ("smolvla", "smolvla_geometry"):
        result = run_smolvla(
            args.checkpoint, args.device, geometry_fusion=args.backend == "smolvla_geometry"
        )
    else:
        result = run_openvla(args.model_id)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
