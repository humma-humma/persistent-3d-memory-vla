# Confidence-Aware Persistent 3D Scene Memory for VLA Policies

## Motivation and question

Vision-language-action policies are typically driven by the current camera
view or a short temporal window. Objects that become occluded can therefore
disappear from the policy input, while a naive long-lived map can retain stale
geometry after objects move. This project asks whether compact, world-aligned
geometric tokens improve action prediction under incomplete views, and whether
calibrated confidence prevents persistent memory from becoming harmful.

## Proposed system

RGB-D or multi-view observations provide camera poses, 3D points, tracks, and
geometric diagnostics. A persistent memory stores one token per stable entity:
identifier, world position, visual feature, confidence, and last-seen frame.
Confidence decays while an entity is unseen. Spatially consistent observations
are confidence-weighted; a confident inconsistent observation replaces stale
geometry. A small adapter exports padded numeric tokens and an attention mask,
or augments the instruction for an unmodified text-conditioned VLA.

## Controlled evaluation

Four inputs are compared: current RGB only, three-frame RGB history,
persistent geometry without confidence gating, and confidence-aware persistent
geometry. The deterministic initial harness sweeps occlusion length, pose noise,
depth noise, and object movement. Metrics are action-position error, memory
position error, confidence Brier score, and memory availability. Closed-loop
task success is reserved for the next stage.

Across 480 seeded synthetic episodes, confidence-aware memory reduces mean
action error from 0.6230 (current view), 0.4269 (short history), and 0.2054
(ungated persistence) to 0.1748. Against ungated persistence, it reduces action
error by 14.9% and confidence Brier score by 51.7%. The result supports the
mechanism but does not yet establish improvement for a pretrained VLA.

## Milestone and next experiment

The repository now contains a tested memory, stable JSON token export,
fixed-size VLA adapter, reproducible controlled benchmark, architecture figure,
and offline inference-contract smoke test. The official SmolVLA base checkpoint
also runs locally on an 8 GB RTX 2080. A 1.64M-parameter projection adapter was
then trained for 500 steps on ten SO-100 episodes, using 1,277 MiB peak PyTorch
VRAM. Across 900 target actions from six held-out episodes, it reduced aggregate
RMSE 40.2% versus the matched base model and beat base on every episode. It
failed the promotion gate because aggregate RMSE remained 18.7% worse than
motor-state persistence. Geometry should be added only after a task-adapted
policy passes that control robustly.

## Honest CV bullet

Built and tested a confidence-aware world-frame 3D memory and lightweight VLA
adapter; in a 480-episode controlled synthetic occlusion/change study, reduced
action-position error 14.9% versus ungated persistent geometry (pretrained-VLA
and closed-loop validation pending).
