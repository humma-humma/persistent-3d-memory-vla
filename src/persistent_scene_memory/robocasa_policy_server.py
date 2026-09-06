"""Serve RoboCasa-native SmolVLA actions from the CUDA Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket

import numpy as np

from .policy_rpc import receive_message, send_message


def serve(checkpoint: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint).resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda"
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to("cuda").eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        print(f"READY {host}:{port}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                try:
                    request = receive_message(connection)
                except ConnectionError:
                    continue
                while True:
                    command = request.get("command")
                    if command == "shutdown":
                        send_message(connection, {"ok": True})
                        return
                    if command == "reset":
                        if hasattr(policy, "reset"):
                            policy.reset()
                        send_message(connection, {"ok": True})
                    elif command != "act":
                        send_message(connection, {"error": f"unknown command {command!r}"})
                    else:
                        try:
                            batch = {
                                "observation.state": torch.from_numpy(
                                    np.asarray(request["state"], dtype=np.float32)
                                ),
                                "task": str(request["task"]),
                            }
                            for index, image in enumerate(request["images"], start=1):
                                value = np.ascontiguousarray(image)
                                batch[f"observation.images.camera{index}"] = (
                                    torch.from_numpy(value).permute(2, 0, 1).float() / 255.0
                                )
                            with torch.inference_mode():
                                action = postprocess(policy.select_action(preprocess(batch)))
                            value = action.detach().cpu().numpy().reshape(-1)
                            if value.shape != (12,) or not np.isfinite(value).all():
                                raise RuntimeError(f"invalid policy action {value.shape}")
                            send_message(connection, {"action": value})
                        except Exception as error:
                            send_message(connection, {"error": repr(error)})
                    try:
                        request = receive_message(connection)
                    except ConnectionError:
                        break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.checkpoint, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
