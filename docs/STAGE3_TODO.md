# Stage 3 TODO

This document records the completed Stage 3 work after the RGB-D visual
odometry baseline, loop-closure post-process, pose graph, keyframe optimizer,
official evaluator, and visualization passes.

## Current Baseline

Implemented:

- Stage 3 dataset validation for `rgb/`, `depth/`, `camera_parameters.json`,
  and `gt_camera_trajectory.txt`.
- Trajectory IO in assignment format:

```text
timestamp tx ty tz qx qy qz qw
```

- RGB-D visual odometry with descriptor matching plus PnP.
- Experimental `lk_pnp` and 3D-3D `rgbd` motion models.
- Per-frame odometry report CSV.
- Translation metrics against ground truth when available.
- Conservative loop-closure post-processing that closes final translation to
  the first pose over the last part of the trajectory.

Current measured reference points:

| Run | Translation RMSE |
|---|---:|
| Tuned PnP, 500 frames full resolution | 0.2166 |
| Tuned PnP, 1000 frames full resolution | 0.7587 |
| Full half-scale VO | 8.4356 |
| Full half-scale VO with current loop closure | 6.4475 |

Full half-scale pose-graph run (`outputs/stage3_pose_graph_full`, 4396 frames,
1 accepted loop edge frame 4395 -> 24, 113 inliers / 0.46 px):

| Trajectory | Internal RMSE | Official ATE (Horn) |
|---|---:|---:|
| Raw VO | 7.1671 | 3.7007 |
| Loop-closed post-process | 6.4217 | 5.3178 |
| Pose-graph optimized | 6.6549 | **3.6164** |

On the official evaluator the original full-node pose-graph trajectory beats
raw VO (3.6164 vs 3.7007) and is far better than the naive loop-closed
post-process (which Horn alignment penalizes). This was superseded by the
keyframe pose-graph run below.

Follow-up experiment (`outputs/stage3_pose_graph_full_v2`): 3 loop edges plus
`--optimizer-max-nfev 1000` did **not** help and slightly regressed the
official ATE to 3.9612. The two extra accepted edges (4395 -> 32, 4347 -> 40)
are near-duplicates of the first end-to-start constraint, so they over-pull the
trajectory tail; internal RMSE improved marginally (6.5552, median 2.1365) but
the Horn-aligned ATE got worse and the eval.py scale rose 1.05 -> 1.12. The
extra iterations also barely moved the cost (0.0213 vs 0.0251 at 200 nfev), so
the practical recommendation is a single high-quality loop edge with the default
optimizer budget. Distinct (non-redundant) loop closures, if the sequence
revisits more than one place, would be the way to add more constraints.

Final promoted run (`outputs/stage3_pose_graph_keyframe_stride8_nfev50`):
keyframe pose graph with stride 8, duplicate loop-endpoint spacing 100,
`--optimizer-max-nfev 50`, 552 optimized nodes, 1 selected loop edge
(`4387 -> 32`, 456 inliers / 0.50 px), full 4396-pose trajectory exported
after interpolating keyframe corrections.

| Trajectory | Internal RMSE | Official ATE (Horn) |
|---|---:|---:|
| Keyframe pose graph, stride 16 / 50 evals | 6.0304 | 3.4319 |
| Keyframe pose graph, stride 8 / 50 evals | **5.2430** | **3.0305** |

This stride-8 keyframe run is the current chosen result. It improves both the
direct internal translation RMSE and the official ATE versus the older
full-node pose graph while running substantially faster than optimizing all
4396 poses directly.

## Status (Priorities 1-5 And Stretch Visualization Implemented)

Implemented in `src/sfm_reconstruction/pose_graph.py` and wired into the
Stage 3 CLI via `--run-pose-graph` (see `sfm-stage3-setup`):

- P1 Visual loop-edge estimation: end-to-start candidate search, AKAZE feature
  reuse, both reference-depth PnP and two-sided 3D-3D RANSAC, full scoring and
  rejection, writes `loop_candidates.csv` and `loop_candidate_summary.json`.
- P2 Pose graph: `PoseGraph` of camera-to-world nodes plus odometry and loop
  edges with weights/inlier metadata, writes `pose_graph_edges.csv` and
  `pose_graph_summary.json`.
- P3 Optimization: per-node rotvec+translation, fixed first pose, weighted
  `soft_l1` residuals via SciPy `least_squares` with a sparse Jacobian
  (mirrors `bundle_adjustment.py`).
- P4 Export/compare: writes `estimated_camera_trajectory_pose_graph.txt`,
  `trajectory_metrics_pose_graph.json`, `pose_graph_optimization_summary.json`,
  and `stage3_comparison_summary.json`.
- P5 Official evaluator wrapper (`src/sfm_reconstruction/official_eval.py`,
  `--run-official-eval`): computes the `eval.py` scale, runs the provided
  `eval_ate.py` via the current interpreter (`sys.executable`, not `python3`),
  injects a vendored TUM `associate` module onto `PYTHONPATH` (the assignment
  ships `eval_ate.py` without it), and writes `official_eval_<label>_stdout.txt`
  plus `official_eval_summary.json`. The assignment files are never modified.
- Keyframe pose graph acceleration: `--pose-graph-keyframe-stride` optimizes a
  reduced graph and interpolates keyframe corrections back to the complete
  trajectory.
- Duplicate loop-edge control: `--loop-endpoint-spacing` prevents near-copy
  loop closures from over-pulling the same trajectory region.
- Static visualization: `src/sfm_reconstruction/stage3_visualize.py` writes a
  top-down aligned trajectory PNG plus aligned translation error plot.
- Interactive visualization: `src/sfm_reconstruction/stage3_open3d_viewer.py`
  opens trajectory overlays, sampled RGB-D scene clouds, optional voxel/outlier
  cleanup, optional Poisson mesh preview, and GT-pose-vs-estimated-pose scene
  comparisons.

Tests: `tests/test_pose_graph.py` (SE(3) helpers, candidate selection,
acceptance/rejection, graph construction, optimization recovery + loop-edge
endpoint pull + fixed first pose, writers, and a synthetic RGB-D loop-edge +
full-pipeline integration test) and `tests/test_official_eval.py` (scale
computation, summary writer, and an integration run against the provided
`eval_ate.py`). Visualization coverage lives in `tests/test_stage3_visualize.py`
and `tests/test_stage3_open3d_viewer.py`.

Remaining: no required Stage 3 implementation items. Further accuracy work is
research-oriented: find more distinct loop closures or improve the visual
matching frontend with learned matching.

## Priority 1: Visual Loop Edge Estimation

Goal: estimate a reliable relative transform between late frames and early
frames where the sequence returns near the start.

Steps:

1. Select loop candidate frame pairs.
   Use pairs such as last frame to first frame, plus a small window:

```text
end frames: last 30-100 frames
start frames: first 30-100 frames
```

2. Reuse Stage 3 feature extraction and matching.
   Start with AKAZE descriptor matching because it is already used by the
   promoted PnP baseline.

3. Estimate candidate loop transforms.
   Try both:

```text
reference depth + PnP
two-sided depth + 3D-3D RANSAC
```

4. Score candidates.
   Record:

```text
matched features
valid depth points
RANSAC/PnP inliers
median reprojection error
relative translation size
relative rotation angle
```

5. Reject unsafe loop edges.
   Reject if:

```text
too few inliers
translation is implausibly large
rotation is implausibly large
median reprojection error is high
```

6. Write loop candidate diagnostics.
   Suggested output:

```text
loop_candidates.csv
loop_candidate_summary.json
```

## Priority 2: Pose Graph Representation

Goal: represent the trajectory as nodes and relative-pose constraints.

Steps:

1. Define a pose graph data structure.

```text
nodes: timestamped camera-to-world poses
edges: relative transforms between node pairs
```

2. Add odometry edges.
   Build frame-to-frame edges from the current VO trajectory.

3. Add loop edges.
   Add one or more accepted end-to-start loop constraints.

4. Store edge metadata.
   Include:

```text
source index
target index
translation weight
rotation weight
inlier count
edge type: odometry or loop
```

5. Export graph diagnostics.
   Suggested output:

```text
pose_graph_edges.csv
pose_graph_summary.json
```

## Priority 3: Pose Graph Optimization

Goal: optimize all poses using odometry constraints plus accepted loop edges.

Steps:

1. Choose parameterization.
   Use a compact pose vector per frame:

```text
rotation vector: 3 values
translation: 3 values
```

2. Fix the first pose.
   Keep frame 0 at origin with identity rotation to match the assignment
   convention.

3. Implement residuals.
   For each edge, compare:

```text
predicted relative transform from current optimized poses
measured relative transform from odometry or loop edge
```

4. Weight residuals.
   Start simple:

```text
odometry edges: normal weight
loop edges: lower weight until verified
translation and rotation weights configurable
```

5. Use SciPy `least_squares`.
   Reuse patterns from `bundle_adjustment.py`.

6. Add robust loss.
   Start with:

```text
loss="soft_l1"
```

7. Write optimized trajectory.

```text
estimated_camera_trajectory_pose_graph.txt
trajectory_metrics_pose_graph.json
pose_graph_optimization_summary.json
```

## Priority 4: Evaluation And Comparison

Goal: make it easy to compare raw VO, current loop-closed post-process, and
pose-graph optimized trajectories.

Steps:

1. Add a comparison command or script.

2. Compare metrics:

```text
translation RMSE
mean translation error
median translation error
max translation error
endpoint translation error
tracked frame transitions
accepted loop edges
```

3. Generate a compact JSON summary.

```text
stage3_comparison_summary.json
```

4. Add trajectory plot support if useful.
   A simple top-down Matplotlib plot is enough:

```text
ground truth
raw VO
loop-closed post-process
pose-graph optimized
```

## Priority 5: Official Evaluator Wrapper

Goal: run the assignment evaluator consistently from the project CLI.

Steps:

1. Wrap the provided `eval.py` / `eval_ate.py`.

2. Handle Windows Python command differences.
   The provided script calls `python3`, which may need a local replacement.

3. Save evaluator output next to internal metrics.

Suggested output:

```text
official_eval_stdout.txt
official_eval_summary.json
```

## Acceptance Criteria

Minimum useful Stage 3 completion:

- [x] A loop edge is estimated from RGB-D data, not manually assumed.
- [x] Pose graph optimization runs end to end.
- [x] Optimized trajectory is exported in assignment format.
- [x] Metrics are written for raw VO, loop-closed post-process, and pose graph
  (`stage3_comparison_summary.json` when ground truth is available).
- [x] The optimized trajectory improves over raw VO on the full sequence
  (official ATE 3.0305 vs 3.7007 on the full 4396-frame half-scale run).
- [x] The implementation has focused unit tests for graph residuals and
  trajectory export.

Stretch goals:

- [x] Multiple loop edges instead of one (`--max-loop-edges`).
- [x] Automatic loop candidate search (windowed end-to-start search).
- [x] Trajectory visualization.
- [x] Official evaluator integration (`--run-official-eval`).
- [x] Keyframe pose graph acceleration.
- [x] RGB-D scene cloud visualization and GT-pose diagnostic overlay.
