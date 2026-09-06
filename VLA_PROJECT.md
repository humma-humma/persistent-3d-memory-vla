# Confidence-Aware Persistent 3D Scene Memory for VLA Policies

## Research question

Can persistent, world-aligned geometric tokens improve a VLA policy when the
current view is incomplete, and can geometric confidence prevent stale or
noisy memory from degrading its actions?

## Minimal system

```text
RGB-D / multi-view video
  -> camera pose, points, tracks, geometric diagnostics
  -> confidence-aware persistent scene memory
  -> compact world-aligned tokens
  -> lightweight adapter for a pretrained VLA
  -> language-conditioned action
```

Each token contains a stable identifier, 3D position, feature vector,
confidence, and last-seen frame. Confidence decays while a token is unseen.
Consistent observations are fused; a confident spatially inconsistent
observation replaces stale geometry so that the map can represent change.

## Required comparison

1. Current RGB observation only.
2. Short RGB/video history.
3. Persistent geometry without confidence gating.
4. Confidence-aware persistent geometry.

The controlled evaluation will vary occlusion duration, camera-pose noise,
depth noise, and object movement. Initial metrics are action-prediction error,
memory position error, confidence calibration, and eventually closed-loop task
success.

## Friday milestone

- Reproducible environment and pretrained VLA inference smoke test.
- Tested persistent-memory implementation and token export.
- One controlled occlusion/change demonstration.
- Architecture figure, short demo, and small baseline table.
- Application-ready one-page research description and honest CV bullet.

Full VLA training, publication-scale benchmarking, dynamic SLAM, and physical
robot deployment are explicitly outside this milestone.

## Current milestone status (2026-08-11)

- Environment lock and setup guide: complete (`requirements-vla.txt`,
  `docs/VLA_SETUP.md`).
- Persistent memory, confidence decay/change handling, and JSON token export:
  implemented and tested.
- Numeric/prompt VLA adapter and inference contract: implemented and tested.
  Real SmolVLA inference is verified locally on the RTX 2080; OpenVLA-7B remains
  wired but is too large for the 8 GB GPU in its BF16 loading path.
- Controlled occlusion/change benchmark and four-baseline table: complete.
- Real-data pre-fine-tuning baseline on official SO-100 pick-place recordings:
  complete; zero-shot SmolVLA is worse than holding the current motor state.
- Low-memory task-adaptation pilot: complete. A 1.64M-parameter projection
  adapter trained on episodes 1–10 reduces held-out normalized RMSE 30.9% versus
  the base model, but does not consistently beat hold-state in raw metrics.
- Architecture figure, one-page research description, and honest CV bullet:
  complete under `deliverables/`.
- Six-episode adapter promotion gate: complete across 900 target actions. The
  adapter improves aggregate RMSE 40.2% versus base and beats base on every
  episode, but fails promotion because RMSE remains 18.7% worse than hold-state.

The current evidence is synthetic and open loop. It validates the mechanism and
software contract, not pretrained-VLA improvement or robot-task success.
