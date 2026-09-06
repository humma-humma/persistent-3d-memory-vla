"""Optional learned local matching for loop-closure experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SuperPointLightGlueMatcher:
    """Extract SuperPoint features and match them with LightGlue."""

    def __init__(
        self,
        *,
        max_keypoints: int = 2500,
        device: str = "auto",
        filter_threshold: float = 0.1,
        image_scale: float = 1.0,
    ) -> None:
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import load_image, rbd
        except ImportError as exc:
            raise RuntimeError(
                "SuperPoint+LightGlue requires PyTorch and the official LightGlue "
                "package. Install the project's learned extra, then install "
                "LightGlue from https://github.com/cvg/LightGlue."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--learned-device cuda requested, but CUDA is unavailable")
        if device not in {"cpu", "cuda"}:
            raise ValueError("learned device must be one of: auto, cpu, cuda")
        if max_keypoints <= 0:
            raise ValueError("max_keypoints must be positive")
        if not 0.0 <= filter_threshold <= 1.0:
            raise ValueError("filter_threshold must be between 0 and 1")
        if image_scale <= 0.0:
            raise ValueError("image_scale must be positive")

        self._torch = torch
        self._load_image = load_image
        self._rbd = rbd
        self._device = device
        self._image_scale = image_scale
        self._extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self._matcher = (
            LightGlue(features="superpoint", filter_threshold=filter_threshold)
            .eval()
            .to(device)
        )

    @property
    def device(self) -> str:
        return self._device

    def extract(self, image_path: str | Path, mask: np.ndarray | None = None) -> dict:
        image = self._load_image(Path(image_path)).to(self._device)
        resize = None
        if self._image_scale != 1.0:
            resize = max(1, int(round(max(image.shape[-2:]) * self._image_scale)))
        with self._torch.inference_mode():
            features = self._extractor.extract(image, resize=resize)
        if mask is not None:
            points = features["keypoints"][0]
            pixels = points.round().long()
            pixels[:, 0].clamp_(0, mask.shape[1] - 1)
            pixels[:, 1].clamp_(0, mask.shape[0] - 1)
            keep = self._torch.as_tensor(
                mask[pixels[:, 1].cpu().numpy(), pixels[:, 0].cpu().numpy()] > 0,
                device=self._device,
            )
            for key in ("keypoints", "keypoint_scores", "descriptors"):
                features[key] = features[key][:, keep]
        return features

    def match(self, features0: dict, features1: dict) -> tuple[np.ndarray, np.ndarray]:
        matches, points0, points1 = self.match_with_indices(features0, features1)
        del matches
        return points0, points1

    def match_with_indices(
        self, features0: dict, features1: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._torch.inference_mode():
            result = self._rbd(
                self._matcher({"image0": features0, "image1": features1})
            )
            features0_unbatched = self._rbd(features0)
            features1_unbatched = self._rbd(features1)

        matches = result["matches"]
        points0 = features0_unbatched["keypoints"][matches[..., 0]]
        points1 = features1_unbatched["keypoints"][matches[..., 1]]
        return (
            matches.detach().cpu().numpy().astype(np.int32, copy=False),
            points0.detach().cpu().numpy().astype(np.float64, copy=False),
            points1.detach().cpu().numpy().astype(np.float64, copy=False),
        )


def create_superpoint_lightglue_matcher(
    *,
    max_keypoints: int,
    device: str,
    filter_threshold: float,
    image_scale: float,
) -> SuperPointLightGlueMatcher:
    return SuperPointLightGlueMatcher(
        max_keypoints=max_keypoints,
        device=device,
        filter_threshold=filter_threshold,
        image_scale=image_scale,
    )
