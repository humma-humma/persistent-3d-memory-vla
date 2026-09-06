"""Optional DINOv2 global descriptors for Stage 3 loop retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LoopProposal:
    source_index: int
    target_index: int
    similarity: float
    rank: int


def select_global_loop_candidates(
    descriptors: np.ndarray,
    frame_indices: list[int],
    *,
    min_separation: int,
    max_candidates: int,
    endpoint_spacing: int,
    min_similarity: float,
) -> list[LoopProposal]:
    """Rank and diversify non-adjacent frame pairs by cosine similarity."""
    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2 or len(descriptors) != len(frame_indices):
        raise ValueError("descriptors must have one row per frame index")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    normalized = descriptors / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T

    ranked = []
    for source_position in range(len(frame_indices)):
        source_index = frame_indices[source_position]
        for target_position in range(source_position):
            target_index = frame_indices[target_position]
            if source_index - target_index < min_separation:
                continue
            similarity = float(similarities[source_position, target_position])
            if similarity >= min_similarity:
                ranked.append((similarity, source_index, target_index))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected: list[LoopProposal] = []
    spacing = max(endpoint_spacing, 0)
    for similarity, source_index, target_index in ranked:
        if any(
            abs(source_index - proposal.source_index) < spacing
            or abs(target_index - proposal.target_index) < spacing
            for proposal in selected
        ):
            continue
        selected.append(
            LoopProposal(source_index, target_index, similarity, len(selected) + 1)
        )
        if len(selected) >= max_candidates:
            break
    return selected


def refine_loop_proposals(
    proposals: list[LoopProposal],
    *,
    frame_count: int,
    radius: int,
    top_k: int,
    min_separation: int,
) -> list[LoopProposal]:
    """Add small one-endpoint perturbations around top coarse proposals."""
    if radius <= 0 or top_k <= 0:
        return proposals
    refined = list(proposals)
    seen = {(proposal.source_index, proposal.target_index) for proposal in refined}
    for proposal in proposals[:top_k]:
        for offset in range(1, radius + 1):
            for source_index, target_index in (
                (proposal.source_index - offset, proposal.target_index),
                (proposal.source_index + offset, proposal.target_index),
                (proposal.source_index, proposal.target_index - offset),
                (proposal.source_index, proposal.target_index + offset),
            ):
                pair = (source_index, target_index)
                if (
                    source_index <= target_index
                    or source_index - target_index < min_separation
                    or target_index < 0
                    or source_index >= frame_count
                    or pair in seen
                ):
                    continue
                seen.add(pair)
                refined.append(
                    LoopProposal(
                        source_index,
                        target_index,
                        proposal.similarity,
                        proposal.rank,
                    )
                )
    return refined


class DinoV2Retriever:
    """Extract normalized DINOv2 ViT-S/14 image descriptors."""

    def __init__(self, *, device: str = "auto", batch_size: int = 16) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 retrieval requires the project's learned dependencies."
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DINOv2 CUDA requested, but CUDA is unavailable")
        if device not in {"cpu", "cuda"}:
            raise ValueError("retrieval device must be one of: auto, cpu, cuda")
        if batch_size < 1:
            raise ValueError("retrieval batch size must be positive")

        self._torch = torch
        self._image = Image
        self._device = device
        self._batch_size = batch_size
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        ).eval().to(device)

    def describe(self, image_paths: list[str | Path]) -> np.ndarray:
        batches = []
        for start in range(0, len(image_paths), self._batch_size):
            tensors = []
            for path in image_paths[start : start + self._batch_size]:
                with self._image.open(path) as image:
                    tensors.append(self._transform(image.convert("RGB")))
            batch = self._torch.stack(tensors).to(self._device)
            with self._torch.inference_mode():
                descriptors = self._model(batch)
                descriptors = self._torch.nn.functional.normalize(
                    descriptors, dim=1
                )
            batches.append(descriptors.cpu().numpy())
        if not batches:
            return np.empty((0, 384), dtype=np.float32)
        return np.vstack(batches).astype(np.float32, copy=False)


def create_dinov2_retriever(*, device: str, batch_size: int) -> DinoV2Retriever:
    return DinoV2Retriever(device=device, batch_size=batch_size)
