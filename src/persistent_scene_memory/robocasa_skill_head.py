"""Train and evaluate a small hierarchical skill head on frozen SmolVLA features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .robocasa_hierarchical import SKILLS


def monotonic_decode(logits: np.ndarray) -> np.ndarray:
    """Decode a sequence while allowing only hold or one-stage advancement."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != len(SKILLS) or len(logits) == 0:
        raise ValueError(f"logits must have shape [steps, {len(SKILLS)}]")
    current = 0
    decoded = []
    for row in logits:
        candidates = [current]
        if current + 1 < len(SKILLS):
            candidates.append(current + 1)
        current = max(candidates, key=lambda index: float(row[index]))
        decoded.append(current)
    return np.asarray(decoded, dtype=np.int64)


def classification_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 1 or len(target) == 0:
        raise ValueError("prediction and target must be matching nonempty vectors")
    confusion = np.zeros((len(SKILLS), len(SKILLS)), dtype=np.int64)
    for expected, actual in zip(target, prediction):
        confusion[expected, actual] += 1
    per_skill = {}
    recalls = []
    for index, skill in enumerate(SKILLS):
        count = int(confusion[index].sum())
        recall = float(confusion[index, index] / count) if count else None
        per_skill[skill] = {"samples": count, "recall": recall}
        if recall is not None:
            recalls.append(recall)
    return {
        "accuracy": float(np.mean(prediction == target)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_skill": per_skill,
        "confusion": confusion.tolist(),
    }


def train_skill_head(
    train_features: str | Path,
    heldout_features: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 1000,
    learning_rate: float = 1e-2,
    seed: int = 17,
) -> dict:
    import torch
    from safetensors.torch import save_file

    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    with np.load(train_features) as archive:
        train_x = archive["features"].astype(np.float32)
        train_y = archive["labels"].astype(np.int64)
        train_skills = tuple(archive["skills"].tolist())
    with np.load(heldout_features) as archive:
        heldout_x = archive["features"].astype(np.float32)
        heldout_y = archive["labels"].astype(np.int64)
        heldout_skills = tuple(archive["skills"].tolist())
    if train_skills != SKILLS or heldout_skills != SKILLS:
        raise ValueError("feature files do not use the hierarchical skill contract")
    if train_x.ndim != 2 or heldout_x.shape[1] != train_x.shape[1]:
        raise ValueError("train and held-out features must share a feature dimension")

    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_normalized = (train_x - mean) / scale
    heldout_normalized = (heldout_x - mean) / scale
    counts = np.bincount(train_y, minlength=len(SKILLS)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("every skill must be represented in training")
    class_weights = len(train_y) / (len(SKILLS) * counts)

    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], len(SKILLS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-3)
    x_tensor = torch.from_numpy(train_normalized)
    y_tensor = torch.from_numpy(train_y)
    weight_tensor = torch.from_numpy(class_weights)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            head(x_tensor), y_tensor, weight=weight_tensor
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    head.eval()
    with torch.inference_mode():
        train_logits = head(x_tensor).numpy()
        heldout_logits = head(torch.from_numpy(heldout_normalized)).numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "weight": head.weight.detach(),
            "bias": head.bias.detach(),
            "feature_mean": torch.from_numpy(mean),
            "feature_scale": torch.from_numpy(scale),
        },
        output_dir / "skill_head.safetensors",
    )
    raw_prediction = heldout_logits.argmax(axis=1)
    monotonic_prediction = monotonic_decode(heldout_logits)
    report = {
        "status": "skill_head_complete",
        "skills": list(SKILLS),
        "training_samples": len(train_y),
        "heldout_samples": len(heldout_y),
        "feature_dimension": train_x.shape[1],
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "training": classification_metrics(train_logits.argmax(axis=1), train_y),
        "heldout_raw": classification_metrics(raw_prediction, heldout_y),
        "heldout_monotonic": classification_metrics(monotonic_prediction, heldout_y),
        "majority_baseline_accuracy": float(
            np.max(np.bincount(heldout_y, minlength=len(SKILLS))) / len(heldout_y)
        ),
        "head": str((output_dir / "skill_head.safetensors").resolve()),
    }
    (output_dir / "skill_head_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--heldout-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    result = train_skill_head(
        args.train_features,
        args.heldout_features,
        args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
