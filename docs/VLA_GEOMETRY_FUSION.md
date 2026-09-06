# Multi-view memory and SmolVLA geometry fusion

## Implemented end-to-end slice

The proposal's update rule and the VLA connector now share one executable
interface:

```text
causal rich-PLY observations
  -> stable world-frame IDs and geometric confidence
  -> retain / uncertain / update memory
  -> [xyz, confidence, age, RGB, uncertainty] tokens + mask
  -> geometry encoder and cross-attention
  -> SmolVLA state prefix token
  -> pretrained action expert
```

`ConsistencyGatedMemory` keeps the committed token after one inconsistent
observation. A candidate is marked uncertain and exposed separately to the
policy. A second spatially agreeing observation from a different frame commits
the candidate. Repeating the same frame cannot confirm a change.

`GeometryConditionedSmolVLA` replaces SmolVLA's ordinary state projection with
a residual geometry cross-attention projection. The original projected motor
state is the query; masked geometry tokens are keys and values. The resulting
state prefix is consumed by SmolVLA's existing VLM/action-expert path. This is
numeric token fusion, not prompt augmentation.

## Measured checks

The local `smolvla_base` checkpoint successfully consumed an `8 x 7` geometry
tensor and returned a finite six-dimensional action on an RTX 2080:

- connector trainable parameters: `4,654,095`;
- peak allocated VRAM: `1,232.20 MiB`;
- artifact: `artifacts/smolvla_geometry_fusion_smoke.json`.

The ambiguity variant of the controlled benchmark adds isolated, single-view
geometric outliers and the two-view gate. Over the same 40-trial condition
sweep:

| Memory rule | Action error | Confidence Brier |
|---|---:|---:|
| Ungated persistence | 0.1977 | 0.2936 |
| Confidence-aware single-view update | **0.1810** | 0.1976 |
| Two-view consistency | 0.1997 | **0.1231** |

Repeated confirmation improves calibration but its one-frame delay worsens the
current hand-authored action proxy. This motivates exposing both the committed
state and uncertain candidate to the VLA, which the 9D adapter now does.

## Real causal replay

The seven existing Stage 1 box prefixes were replayed with
`--confirmation-views 2` into
`outputs/stage1_box_causal_memory_8/causal_memory_consistency`.

The final frame contains 711 committed tokens. The replay produced no uncertain
events: the scene is static, shared point drift is below the 0.25 threshold,
and new points receive new identities. It validates the interface but cannot
evaluate scene-change confirmation.

## Commands

```powershell
$env:PYTHONPATH = "src"

& $python -m persistent_scene_memory.benchmark `
  --include-consistency `
  --single-view-outliers `
  --output-dir artifacts\multi_view_gate_demo

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& .\.venv-smolvla\Scripts\python.exe -m persistent_scene_memory.vla_smoke `
  --backend smolvla_geometry `
  --checkpoint models\smolvla_base `
  --device cuda `
  --output artifacts\smolvla_geometry_fusion_smoke.json
```

## Remaining scientific milestone

The connector is untrained. The next required data artifact must contain, in
the same episodes, language instructions, robot state/action chunks, calibrated
sequential observations, controlled occlusion or displacement, and object-level
identity across the change. On that dataset, train only the connector first and
compare RGB-only, short-history, current geometry, ungated memory,
confidence-weighted memory, and multi-view consistency memory.

## RoboCasa offline policy bridge

The recorded RoboCasa episode can now be replayed through both the base and
geometry-conditioned SmolVLA paths:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& .\.venv-smolvla\Scripts\python.exe -m persistent_scene_memory.robocasa_smolvla_replay `
  --episode-dir outputs\robocasa_oracle_pick_place_success\episode `
  --memory-dir outputs\robocasa_oracle_pick_place_success\memory `
  --checkpoint models\smolvla_base `
  --frames 0 64 128 `
  --output artifacts\robocasa_smolvla_offline_replay.json
```

This is an interface verification, not a policy result. The local pretrained
checkpoint consumes six SO-100 joint-state values and predicts six normalized
SO-100 controls. RoboCasa records a 12D Cartesian arm/gripper/mobile-base
action. The bridge uses end-effector pose as a six-value observation surrogate,
retains the oracle action as an explicit 12D label, and refuses to convert the
pretrained output into simulator control. Closed-loop evaluation requires a
RoboCasa-native state/action head and fine-tuning. The geometry connector is
also still untrained.

## RoboCasa-native action head

`persistent_scene_memory.robocasa_finetune` changes the checkpoint contract to
the simulator's complete 16D proprioceptive state and 12D structured action,
computes training-only normalization statistics, and fine-tunes the existing
SmolVLA state/action/time projections while keeping the VLM frozen:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& .\.venv-smolvla\Scripts\python.exe -m persistent_scene_memory.robocasa_finetune `
  --checkpoint models\smolvla_base `
  --episode-dir outputs\robocasa_oracle_pick_place_success\episode `
  --output-dir outputs\smolvla_robocasa_native_smoke `
  --steps 2 --stride 20
```

The 100-step seed-0 overfit run trains 1,635,152 parameters at 1.29 GiB peak
allocated VRAM and reduces a fixed-noise probe loss from `1.2548` to `0.5509`.
Its training-probe action RMSE is `0.4403`, versus `0.4591` for zero control.
It verifies that the saved processor and checkpoint produce finite `50 x 12`
native action chunks and can learn the recorded controls. It is deliberately
labelled a same-seed overfit smoke.
Policy promotion requires frozen held-out seeds, comparison against zero/hold
control, and closed-loop task success.

The rollout CLI supports `--stop-on-success --skip-memory` for inexpensive RGB-D
label collection and `--object-group` for controlled object-family studies.
Scene generation is seeded before environment construction, and the improved
oracle uses object-axis-aligned grasp frames, long-axis grasp-point selection,
object-sized cabinet clearance, slow carrying motion, grasp-loss hysteresis,
verified terminal success, and slow base staging before insertion.

The data checkpoint uses successful seed-0 and seed-2 episodes (30 causal
chunks). Its 100-step native-head training run reduces fixed-noise probe loss
from `2.6339` to `1.2883` and beats zero control on the training probe (`0.4035`
versus `0.4591` RMSE). The fine-tuning entry point rejects unsuccessful
demonstrations and duplicate seeds before loading SmolVLA.

Seed 1 is now a successful frozen held-out demonstration: cabinet-width
alignment for safe in-hand placement and the task's own inside-cabinet predicate
produce native success at frame 271. `robocasa_policy_eval` evaluates six
causal chunks (300 actions) without updating weights. The seed-0/2 checkpoint
scores `0.2689` RMSE versus `0.4014` for zero action. Per-field results show the
gain is concentrated in gripper and control mode; arm translation, rotation,
and base motion remain worse than zero.

`robocasa_policy_server` and `robocasa_closed_loop` bridge the incompatible
CUDA/SmolVLA and MuJoCo/RoboCasa Python environments over loopback. The first
seed-1 run executes 300 bounded native actions from three current RGB views and
16D proprioception, with no oracle. It fails the task: the gripper is commanded
closed for all frames, the end effector moves `0.576 m` in relative coordinates,
and the base moves only `0.017 m`. This verifies closed-loop execution, not
closed-loop competence. More successful demonstrations and then the matched
RGB-only versus persistent-memory ablation are still required.

## Phase-balanced RGB-only follow-up

The follow-up run samples the seven represented controller phases uniformly
instead of sampling the 30 causal anchors uniformly. A 500-step run assigns
`71–72` updates per phase and improves training-probe RMSE to `0.3541`.
Frozen seed-1 RMSE nevertheless regresses from `0.2689` to `0.3066`, and a
matched 300-step closed-loop run remains unsuccessful. It does show less
degenerate behavior: gripper-close frequency falls from `1.00` to `0.817` and
base displacement rises from `0.017 m` to `0.067 m`.

Corrected reach/orientation handling lets oracle candidates 5–7 acquire and
transport the object, but none of seeds 5–10 passes native task success within
1,000 steps. They remain excluded. The comparison is recorded in
`outputs/smolvla_phase_balanced_comparison.json`; the 100-step checkpoint is
retained as the stronger frozen-held-out RGB baseline.

## Official RoboCasa human demonstrations

The official pretraining-human `PickPlaceCounterToCabinet` bundle is imported
through `robocasa_official_import`. All 108 released episodes terminate with
positive reward. The released modalities exactly match the 16D state and 12D
action fields, but use a different field order and `-1/1` binary encoding; the
adapter reads those mappings from metadata and exports the live `0/1` contract.

The targeted cache contains 32 unique successful language/object tasks and 341
causal examples. Combined with two mobile-base oracle episodes, a 1,000-step
run trains the projections and final action-expert layer (7.65M parameters,
1.32 GiB peak VRAM). Frozen seed-1 RMSE is `0.3391`, still above the retained
baseline's `0.2689`, and closed-loop success is false. Official episodes have
zero base motion, so the trained policy uses base control on only 5.3% of frames
and moves `0.005 m`. The next data requirement is therefore successful
mobile-base demonstrations matched to the evaluation scene distribution.

Nine successful keyboard demonstrations were subsequently collected across two
RoboCasa layouts and converted with `robocasa_teleop_import`. Exact-model replay
exports 42,624 native actions and 2,122 sparse RGB anchors; 1,778 raw frames
contain nonzero base commands. The portable cache occupies 186.44 MiB.

Combining those demonstrations with the two oracle episodes produces 2,152
causal examples. A 1,000-step RGB-only run adapts 1.64M parameters and assigns
exactly 100 updates to each of ten represented behavior phases. Frozen seed-1
RMSE is `0.3043` versus `0.4014` for zero action. This marginally improves on
the earlier two-teleoperation motion-balanced result (`0.3077`) but remains
worse than the retained two-oracle checkpoint (`0.2689`). Base-field RMSE
improves from `0.1482` to `0.1239`, but it too remains worse than the retained
checkpoint (`0.1082`) and zero action (`0.0909`). Under the fixed promotion
rule, the new checkpoint is retained as ablation evidence but is not promoted
or advanced to closed-loop evaluation.

## Targeted anti-forgetting follow-up

Per-field diagnostics show that the nine-teleoperation model improves only
end-effector position. A field-routed upper-bound experiment therefore keeps
the retained checkpoint for all other action fields and substitutes the new
position prediction. On the same 300 frozen seed-1 actions it reaches `0.2584`
RMSE, below the retained `0.2689`. Because field selection uses this evaluation
seed, this result is explicitly diagnostic rather than independent evidence.

The deployable follow-up starts from the retained checkpoint, reuses its state
and action statistics, freezes every model parameter, and masks gradients to
rows 0–2 of `action_out_proj`. After 200 phase-balanced updates, only those
three weight and bias rows differ from the parent checkpoint. Frozen seed-1
RMSE reaches `0.2662`; every reported field improves slightly and the result
passes the predeclared open-loop comparison.

The required closed-loop run nevertheless remains unsuccessful after 300
steps. End-effector displacement is `0.664 m`, base displacement is `0.047 m`,
and gripper-close fraction is `0.993`. The checkpoint is retained as the best
open-loop RGB model, not promoted as a task-competent policy.

## Short-horizon temporal intervention

The exported temporal diagnostic compares the held-out target, the unchanged
10-step parent, the gripper-targeted candidate, and its native rollout. On the
150 matched held-out actions, the target contains two gripper and nine
control-mode transitions. The parent instead predicts zero gripper and 72
control-mode transitions. The training cache contains 2,147 valid 10-step
examples, but only 30 gripper-transition and 29 control-transition windows.

`action_event` sampling assigns 50 updates each to gripper-transition,
gripper-open, control-transition, and steady-closed windows. Joint gripper and
control training overcorrects control mode (`0.3262` versus the matched
parent's `0.3013` RMSE) and is rejected before rollout. Preserving control and
updating only the gripper row reaches `0.3010`, reduces held-out always-closed
behavior from 100% to 95.3%, and passes the matched comparison by 0.09%.

Its 300-step closed-loop rollout still fails, although gripper-close frequency
falls from 99.3% to 94.3%. Base displacement falls to `0.014 m`, confirming
that the intervention does not recover navigation. Privileged takeover after
30 frames and after 5 frames both fail to recover seed 0 within 1,000 steps;
neither failed episode is admitted to training. Successful human corrections
from policy-visited states are therefore required before another behavior-
cloning iteration.

## Hierarchical VLA scaffold

The monolithic failure motivates a six-skill interface: approach object, grasp,
lift, approach cabinet, insert, and release. `robocasa_skill_features` extracts
frozen contextual SmolVLA features from three RGB views, language, and 16D
proprioception without action diffusion. `robocasa_skill_geometry` appends a
relative-world feature contract built from end-effector, object, base, and
grasp state. The provider is source-agnostic: privileged simulation geometry,
current reconstruction, and persistent memory must all supply the same fields.

The low-level decomposition is viable. A fresh seed-1 privileged rollout passes
native task success in 356 frames, while the monolithic SmolVLA fails after 300
frames. The learned high-level gate is not yet viable: 159 seed-0/2 samples
produce 48.5% monotonic accuracy from context alone and 55.9% after privileged
geometry is appended on 68 frozen seed-1 samples. Insert and release recall
remain zero. The selector is therefore not connected to the successful
executor, and no learned hierarchical success is claimed. Failed seed-3/4
oracle probes are also excluded from training.

### Bounded official-data follow-up

One hundred successful episodes from the already-local official
`PickPlaceCounterToCabinet` bundle are exported with event-aligned six-skill
labels. Gripper close/open transitions anchor grasp and release; the carrying
interval supplies lift, cabinet approach, and insertion segments. Start,
midpoint, and end RGB anchors for each segment prevent sparse sampling from
dropping short skills. The resulting cache has 22,336 frames, 2,631 RGB
anchors, 68 unique task instructions, and occupies 658.41 MiB.

The frozen 960D SmolVLA table contains 589 approach-object, 328 grasp, 383
lift, 549 approach-cabinet, 441 insert, and 341 release samples. An
official-only linear head transfers poorly to frozen seed 1 (14.7% raw, 4.4%
monotonic), exposing the official-to-oracle domain gap. Capping the auxiliary
data at 48 samples per skill and mixing it with the original in-domain features
raises context-only monotonic accuracy from 48.5% to 52.9%. It remains below
the 55.9% privileged-geometry result and still has zero insert/release recall,
so no closed-loop promotion is made.
