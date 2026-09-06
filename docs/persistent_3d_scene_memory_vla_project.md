# Persistent 3D Scene Memory for Vision-Language-Action Policies

## Project proposal, implementation record, results, and revised plan

**Status date:** 13 August 2026
**Current stage:** Initial milestone complete; real-policy promotion gate failed

## Executive summary

This project extends an existing Structure-from-Motion (SfM) and RGB-D
reconstruction stack into embodied AI by using a persistent, world-aligned 3D
scene representation as memory for a Vision-Language-Action (VLA) policy.

The central research question is:

> Does a temporally persistent geometric scene representation improve VLA
> robustness and long-horizon action prediction compared with RGB-only,
> short-history, or per-frame 3D representations?

The project has completed its initial engineering milestone and one additional
real-data adaptation study. A confidence-aware memory, token adapter,
controlled benchmark, local SmolVLA environment, real-data evaluator, and
low-memory fine-tuning path are implemented and tested.

The controlled synthetic study supports the memory mechanism: confidence-aware
persistence achieved the lowest action error under occlusion and scene change.
The real SmolVLA study also demonstrated learning relative to the pretrained
base model. However, the adapted policy remained worse than a trivial
hold-state baseline. It therefore failed the predeclared promotion gate.

This does not reject the project hypothesis. It means the real RGB-only task
policy is not yet strong enough to support an interpretable geometry-fusion
experiment. The next prerequisite is to train a task-adapted baseline that
reliably beats hold-state. Only then should persistent geometry be introduced
and evaluated.

## Motivation and hypothesis

Many VLA policies operate primarily on the current RGB observation or a short
history. This can be insufficient when:

- an important object becomes temporarily occluded;
- the camera viewpoint changes substantially;
- the robot moves away from a previously observed region;
- the current frame is ambiguous but an earlier observation resolved it;
- a long-horizon instruction depends on an earlier spatial relationship.

A SLAM-style system maintains information across observations. The proposed
system uses that persistent state as an explicit memory:

\[
I_1, I_2, \ldots, I_t \rightarrow M_t \rightarrow \text{VLA} \rightarrow a_t,
\]

where \(I_t\) is the current observation, \(M_t\) is the persistent geometric
memory, and \(a_t\) is the predicted action.

The refined hypothesis is:

> Explicit persistent geometry is most useful when the current visual
> observation is insufficient to recover task-relevant spatial information;
> confidence gating is necessary when geometry can become stale or noisy.

Positive, negative, and conditional outcomes remain scientifically meaningful.
Persistent geometry may improve robustness, may be outweighed by mapping
noise, or may help only during prolonged occlusion and major viewpoint change.

## Starting point

The existing reconstruction project already provided:

- feature extraction and matching;
- visual tracking and multiview tracks;
- calibrated camera-pose estimation;
- triangulation and bundle adjustment;
- sparse and dense multiview reconstruction;
- RGB-D trajectory and mapping utilities;
- geometric diagnostics and visualization.

The embodied extension converts sequential geometric observations into a
compact persistent memory:

\[
(M_{t-1}, I_t) \rightarrow (T_t, \mathcal{P}_t) \rightarrow M_t,
\]

where \(T_t\) is the camera pose and \(\mathcal{P}_t\) is newly observed
geometry.

## Implemented system

### Confidence-aware persistent memory

Each memory token stores:

- a stable identifier;
- a world-frame 3D position;
- a feature vector;
- confidence;
- the last-seen frame.

Confidence decays while a token is unseen. Spatially consistent observations
are fused. A confident observation that is inconsistent with stale geometry
replaces the old state, allowing the memory to represent object movement.

The implementation supports confidence-sorted queries, padded token matrices,
attention masks, and portable JSON export.

### VLA adapter

A lightweight adapter converts memory into fixed-size numeric tokens of the
form:

\[
[x, y, z, \text{confidence}, \text{age}, \text{feature}],
\]

with an attention mask and stable token ordering. A prompt-based route can also
express world-aligned facts to an unmodified text-conditioned policy.

These paths verify the memory-to-policy interface. The numeric geometry tokens
have not yet been fused into and trained with the real SmolVLA policy.

### VLA environment and checkpoint

An isolated Python environment was created with LeRobot 0.4.4 and CUDA-enabled
PyTorch 2.8.0. The official `lerobot/smolvla_base` checkpoint was downloaded
locally and verified on an NVIDIA GeForce RTX 2080 with 8 GB VRAM.

Real inference returned a finite six-dimensional action. Model loading plus the
first inference took 38.9 seconds and peak PyTorch allocation was 1,213 MiB.
This establishes that SmolVLA is practical on the available hardware.

### Real-data pipeline

The official `lerobot/svla_so100_pickplace` dataset was downloaded and
integrated. It contains 50 episodes, two RGB cameras, a six-dimensional robot
state, and six-dimensional actions. LIBERO was not used because its state and
action dimensions do not match this SmolVLA checkpoint.

The evaluator performs deterministic, offline action-chunk prediction from the
two images, current motor state, and language instruction. It reports raw MAE,
raw RMSE, normalized RMSE, per-dimension errors, per-episode results, timing,
and VRAM. A hold-state control repeats the current motor state over the target
horizon.

### Low-memory fine-tuning pilot

The pilot freezes the SmolVLA vision-language model and action expert. It trains
only 1,635,152 state/action projection and time-conditioning parameters.

Training configuration:

- SO-100 episodes 1-10;
- five anchors per episode;
- 500 deterministic optimization steps;
- batch size one;
- 256-pixel images;
- learning rate \(10^{-4}\).

Training completed in 106.3 seconds with 1,277 MiB peak PyTorch VRAM. The
minimum observed batch loss was 0.0148. This is a bounded projection-layer
pilot, not full task fine-tuning.

## Experiments and results

### Controlled synthetic memory experiment

The controlled benchmark contains 480 seeded episodes across 12 combinations
of occlusion length, pose noise, depth noise, and object movement. It compares:

1. current-view observation only;
2. a three-frame history;
3. persistent geometry without confidence gating;
4. confidence-aware persistent geometry.

| Input or memory condition | Action error | Position error | Confidence Brier | Availability |
|---|---:|---:|---:|---:|
| Current view only | 0.6230 | 0.0715 | 0.0630 | 0.5747 |
| Short history | 0.4269 | 0.0929 | 0.1013 | 0.7662 |
| Ungated persistent geometry | 0.2054 | 0.2054 | 0.2522 | 1.0000 |
| Confidence-aware persistence | **0.1748** | **0.1748** | **0.1217** | **1.0000** |

Confidence-aware persistence reduces action error 14.9% relative to ungated
persistence and reduces its confidence Brier score by 51.7%. This supports the
mechanism under controlled partial observability and scene change. It is
synthetic evidence, not evidence of real-VLA or closed-loop improvement.

### Initial one-episode adaptation check

At matched 256-pixel resolution, the first held-out check used episode 0 with
three anchors and a 50-action horizon per anchor.

| Method | MAE | RMSE | Normalized RMSE |
|---|---:|---:|---:|
| Base SmolVLA | 10.8110 | 15.8030 | 0.7107 |
| Projection adapter | 5.8263 | 9.1560 | **0.4914** |
| Hold-state | **5.1759** | **8.2502** | 0.4924 |

The adapter reduced normalized RMSE 30.9% relative to base and narrowly beat
hold-state on that single normalized metric. It remained worse than hold-state
in raw MAE and RMSE, motivating a broader evaluation rather than promotion.

### Six-episode promotion gate

The matched base and adapted checkpoints were evaluated on held-out episodes
0, 11, 12, 13, 14, and 15. Each episode used anchors 0, 50, and 100 with a
50-action horizon, giving 900 target actions.

The predeclared gate required the adapted model to beat both base SmolVLA and
hold-state in aggregate RMSE and normalized RMSE.

| Method | MAE | RMSE | Normalized RMSE |
|---|---:|---:|---:|
| Base SmolVLA | 15.5077 | 22.4846 | 0.9318 |
| Projection adapter | 9.7764 | 13.4545 | 0.6756 |
| Hold-state | **7.9786** | **11.3306** | **0.5386** |

Relative to base SmolVLA, the adapter improved:

- MAE by 37.0%;
- RMSE by 40.2%;
- normalized RMSE by 27.5%.

It beat base on all six episodes for all three metrics. Against hold-state it
won 0/6 episodes on MAE, 1/6 on RMSE, and 1/6 on normalized RMSE. Aggregate
adapter RMSE was 18.7% worse than hold-state, and normalized RMSE was 25.4%
worse.

**Promotion decision: failed.** The adapter learned useful task-specific
behavior relative to the pretrained base, but it is not yet a competitive
action predictor under this open-loop control.

### Bounded baseline refinement

A follow-up run expanded training to all 44 non-gate episodes and six anchors
per episode. Projection-only training regressed to `16.4831` RMSE. Selectively
unfreezing the final action-expert layer recovered `13.6423` RMSE. A scalar,
then six-channel, residual around the current motor state was calibrated using
episodes 1-10 only. The best diagonal residual reached `12.1506` RMSE and
`0.5557` normalized RMSE, versus `11.3306` and `0.5386` for hold-state.

This closes 61% of the original adapter-to-hold RMSE gap but does not cross the
gate. Full results and artifacts are in
`deliverables/BASELINE_REFINEMENT.md`. No geometry-conditioned result should be
claimed until a non-geometric policy passes a newly frozen confirmatory split.

## Effect on the original proposal

The results do not invalidate the proposal. They change its present scientific
status and the required experimental order.

### Claims currently supported

- SmolVLA inference and lightweight adaptation fit comfortably on the local
  8 GB GPU.
- The memory representation, confidence logic, export, and policy interface are
  implemented and tested.
- Confidence-aware persistence improves the synthetic controlled mechanism
  test under occlusion and scene change.
- Projection-only SmolVLA adaptation produces substantial and consistent
  improvement over the pretrained base model.

### Claims not yet supported

- Persistent geometry improves a real pretrained or task-adapted VLA.
- Geometry improves action prediction relative to a competitive non-geometric
  control.
- The reconstruction system and SmolVLA operate as one trained end-to-end
  system.
- Persistent memory improves closed-loop task success or long-horizon success.
- The method is robust on CALVIN, LIBERO, or a physical robot.

The real-data evaluation is open-loop action prediction. It must not be
described as robot task success.

### Why geometry integration is postponed

If geometry were added to the current weak baseline, a negative result would be
ambiguous: it could reflect ineffective geometry, inadequate task adaptation,
or both. A positive result could also be misleading if it still failed to beat
hold-state. The RGB-only task policy must therefore first pass the non-geometric
control.

Hold-state is now a required baseline and promotion criterion, rather than only
an implementation diagnostic.

## Current phase status

| Original phase | Status | Evidence and remaining work |
|---|---|---|
| Phase 1: sequential mapping | Partial | Memory, stable tokens, rich-PLY conversion, same-map snapshots, causal prefix reconstruction, stable point IDs, and real-data pose/map/confidence validation are complete. Efficient live mapping, delayed-pose backfill, and object-level association remain. |
| Phase 2: pretrained VLA baseline | Engineering complete; scientific gate failed | Environment, checkpoint, real inference, dataset integration, adaptation, and evaluation work. The adapted policy does not beat hold-state. |
| Phase 3: geometry-conditioned VLA | Partial | Numeric and prompt adapters plus a tested SmolVLA geometry cross-attention connector exist. Base and geometry-conditioned inference now run offline on aligned RoboCasa RGB/state/memory frames, but the connector and a RoboCasa-native action head have not been trained or evaluated. |
| Phase 4: controlled memory tests | Synthetic complete; real incomplete | Synthetic occlusion/change evaluation is complete. Matched real-policy occlusion, viewpoint, and long-horizon experiments remain. |

The smaller Wednesday-Friday milestone is complete. The full research proposal
is approximately halfway complete because the central real-policy comparison
has not yet been performed.

## Revised experimental plan

### Gate 1: establish a competitive task policy

Improve the RGB-only SmolVLA adaptation until it reliably beats hold-state on a
frozen multi-episode split in raw and normalized metrics. Candidate changes
include broader training coverage, improved sampling, a validation split,
longer training, and selectively unfreezing more of the action model while
remaining within the 8 GB memory limit.

This stage must preserve the existing base and hold-state controls and report
per-episode results.

### Gate 2: connect real sequential geometry

Connect sequential SfM or RGB-D outputs to the persistent memory using explicit
world-frame coordinates, confidence derived from geometric diagnostics, stable
association, and change detection. Validate map accuracy before feeding tokens
to the policy.

The initial file-level bridge is implemented in
`src/persistent_scene_memory/reconstruction_bridge.py` and documented in
`docs/RECONSTRUCTION_MEMORY_INTEGRATION.md`. It consumes rich SfM PLY snapshots,
derives confidence from track support, reprojection residual, and triangulation
angle, and writes memory plus fixed-size adapter snapshots. Live incremental
export is now wired into the SfM CLI; explicit cross-run alignment/association
remains required for independently reconstructed maps.

The SfM CLI now also supports `--write-incremental-memory`. It snapshots the
same continuing reconstruction after initialization and every successful PnP
registration, removes future-camera observations from each snapshot, and runs
the bridge automatically. Events currently follow best-PnP registration order,
not necessarily image acquisition order; real sequential validation remains
the next gate.

A stricter causal reference path is implemented in
`src/persistent_scene_memory/causal_mapping.py`. It reconstructs acquisition
prefixes using only available inputs, fixes the initial gauge, carries stable
point IDs, and reports pose error, map drift, nearest-GT proxy error,
confidence calibration, and quality versus age. The verified result is in
`docs/CAUSAL_MAPPING_VALIDATION.md`.

### Gate 3: matched geometry ablation

Freeze the dataset, task-policy checkpoint, training budget, and evaluation
protocol. Compare:

1. RGB-only SmolVLA;
2. RGB plus short temporal history;
3. RGB plus per-frame geometry;
4. RGB plus persistent geometry without confidence gating;
5. RGB plus confidence-aware persistent geometry.

This expands the original three-way proposal so the specific contribution of
temporal history and confidence gating can be isolated.

### Gate 4: controlled partial-observability tasks

Create or select sequences in which a relevant object is first visible and
then deliberately occluded, displaced, or viewed from a substantially different
camera pose while the instruction still depends on its remembered location.

Measure:

- open-loop action error;
- task success rate;
- success under occlusion and viewpoint shift;
- degradation versus time unseen;
- stale-memory failures after object movement;
- confidence calibration;
- long-horizon sequence success.

### Gate 5: closed-loop evaluation

Only after the matched open-loop geometry ablation passes should the project
claim or test closed-loop task improvement. A physical robot is optional; a
controlled simulator is sufficient for the first causal test.

Current status: the native SmolVLA bridge has completed one 300-step seed-1
RGB/proprioception rollout, but task success is false. The held-out open-loop
checkpoint beats zero action overall (`0.2689` versus `0.4014` RMSE), while its
arm and base fields remain worse. The execution interface is validated; policy
competence and the memory-conditioned causal comparison are not yet validated.

A subsequent 500-step phase-balanced run improves training fit but worsens
frozen seed-1 action RMSE (`0.3066` versus the earlier `0.2689`) and again fails
closed loop. Six additional deterministic candidate seeds (5–10) fail native
task success and are excluded from training. This isolates demonstration
coverage—not optimizer exposure alone—as the current blocking variable.

The official human task bundle has subsequently been integrated: 32 successful
object-diverse episodes plus two local mobile-base episodes train a larger
native action head, but held-out RMSE is `0.3391` and closed-loop success remains
false. The released human episodes contain no base movement, isolating matched
mobile-navigation demonstration coverage as the more specific data gap.

The matched mobile-base set now contains nine native-success keyboard episodes
from two layouts: 42,624 actions, 2,122 causal RGB anchors, and 1,778 raw frames
with nonzero base commands. A motion-balanced 1,000-step RGB-only run on these
nine episodes plus the two oracle episodes reaches `0.3043` frozen seed-1 RMSE,
beating zero action (`0.4014`) but not the retained two-oracle checkpoint
(`0.2689`). Base-field RMSE improves over the earlier two-teleoperation run
(`0.1239` versus `0.1482`) but remains worse than the retained checkpoint
(`0.1082`). The larger dataset therefore improves the targeted behavior without
passing the fixed promotion gate; no closed-loop claim is made for this model.

A targeted anti-forgetting checkpoint subsequently freezes the retained VLA,
preserves its normalization contract, and updates only the three final
end-effector-position rows for 200 balanced steps. It improves frozen seed-1
RMSE from `0.2689` to `0.2662`, with every non-target projection row verified
byte-identical. This passes the open-loop comparison but not the scientific
task-success gate: a 300-step native rollout remains unsuccessful, closes the
gripper for 99.3% of frames, moves the end effector `0.664 m`, and moves the
base `0.047 m`.

A subsequent temporal intervention shortens action chunks from 50 to 10 and
balances four gripper/control event classes. The gripper-only candidate
marginally improves matched 10-step RMSE (`0.3010` versus `0.3013`) and reduces
closed-loop gripper-close frequency from 99.3% to 94.3%, but native task success
remains false and base displacement falls to `0.014 m`. Automated oracle
takeover after 30 or 5 policy frames cannot recover seed 0, so both failed
episodes are excluded. The remaining competence bottleneck is successful
correction data from policy-visited states.

The subsequent hierarchical scaffold cleanly separates task reasoning from
motor execution. Its privileged low-level controller succeeds on a fresh
seed-1 rollout in 356 frames, demonstrating that the decomposition can cross
the motor-control barrier. A frozen SmolVLA six-skill selector trained on seeds
0/2 reaches only 48.5% monotonic held-out accuracy from context and 55.9% with
privileged relative-world geometry, with zero recall on insert and release.
Consequently, the selector is not deployed and this is not yet a learned VLA
success. A tested provider contract now permits the same geometric fields to
come from simulator state, reconstruction, or persistent memory.

## Central controlled failure case

The most informative experiment remains:

1. The policy observes object \(A\).
2. The robot or camera moves.
3. Object \(A\) becomes occluded or leaves the field of view.
4. The instruction still requires an action relative to \(A\).

An RGB-only or per-frame geometric policy cannot recover an invisible object
from the current observation. Persistent memory may retain its world position,

\[
p_A^{\text{world}},
\]

while confidence decay and replacement prevent indefinite reliance on stale
geometry.

## Research contribution and project framing

The intended contribution is not merely to fine-tune an existing VLA or to
append generic 3D features. It is to test whether world-aligned persistent
geometry provides an explicit, useful memory mechanism for embodied action:

\[
\text{SfM/RGB-D mapping}
\rightarrow \text{persistent 3D memory}
\rightarrow \text{VLA conditioning}
\rightarrow \text{action}.
\]

The current honest project statement is:

> Confidence-aware persistent geometry improves a controlled synthetic
> occlusion/change mechanism test. Real SmolVLA adaptation substantially
> improves the pretrained checkpoint, but it remains worse than motor-state
> persistence. A stronger task baseline is required before the central
> real-policy geometry comparison can be interpreted.

## Relationship to the master's thesis

The master's thesis, **VISTA: Video-Guided Stylized Human Motion Synthesis**,
remains a separate research line focused on multimodal representation learning,
video conditioning, generative modelling, and 3D human motion.

This project demonstrates complementary experience in multiview geometry,
camera pose estimation, persistent representations, embodied perception, VLA
models, and action evaluation. Together they support a broader research profile
at the intersection of video, 3D geometry, multimodal foundation models, and
embodied AI.

## Reproducibility and project artifacts

The repository contains:

- `src/persistent_scene_memory/memory.py`: persistent confidence-aware memory;
- `src/persistent_scene_memory/adapter.py`: geometry-token adapter;
- `src/persistent_scene_memory/benchmark.py`: controlled synthetic benchmark;
- `src/persistent_scene_memory/vla_smoke.py`: mock, OpenVLA, and SmolVLA paths;
- `src/persistent_scene_memory/fine_tune_pilot.py`: low-memory adaptation pilot;
- `src/persistent_scene_memory/real_data_eval.py`: multi-episode real-data evaluator;
- `src/persistent_scene_memory/robocasa_smolvla_replay.py`: guarded offline RoboCasa-to-SmolVLA replay;
- `src/persistent_scene_memory/promotion.py`: fixed promotion comparison;
- `docs/VLA_SETUP.md`: environment and reproduction commands;
- `deliverables/BASELINE_TABLE.md`: synthetic results;
- `deliverables/FINETUNE_PILOT.md`: training and initial evaluation;
- `deliverables/MULTI_EPISODE_PROMOTION_GATE.md`: final promotion decision;
- `deliverables/RESEARCH_ONE_PAGER.md`: concise research summary;
- `artifacts/smolvla_promotion_gate.json`: machine-readable gate result.
- `artifacts/robocasa_smolvla_offline_replay.json`: real RoboCasa policy-interface check.

The latest full test run reports **184 passing tests**, and the SmolVLA
environment reports no broken Python requirements.

## Recommended title

**Confidence-Aware Persistent 3D Scene Memory for Vision-Language-Action
Policies**

This title reflects both the original persistent-memory contribution and the
implemented confidence mechanism that addresses stale geometry.
