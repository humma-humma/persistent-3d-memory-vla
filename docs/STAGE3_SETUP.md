# Stage 3 Setup And Final Run

Stage 3 is the RGB-D SLAM trajectory task. The assignment data is expected to
contain:

```text
rgb/
depth/
camera_parameters.json
gt_camera_trajectory.txt
eval.py
eval_ate.py
```

The trajectory format is:

```text
timestamp tx ty tz qx qy qz qw
```

where translation is the optical center position and the quaternion is ordered
as `qx qy qz qw`.

## Environment

Use the Open3D-enabled environment for Stage 3 runs and visualization:

```powershell
$env:PYTHONPATH = "src"
$python_open3d = "python"
```

## Validate A Dataset

```powershell
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_setup"
```

This writes `stage3_manifest.json` with frame timestamps, RGB/depth paths,
intrinsics, and ground-truth availability.

## RGB-D Visual Odometry Baseline

The baseline estimates pairwise RGB-D motion with AKAZE/SIFT matching,
reference-depth PnP, and accumulated camera-to-world poses.

Full-sequence half-scale baseline:

```powershell
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_rgbd_vo_full_guarded_scale05" `
  --run-rgbd-odometry `
  --image-scale 0.5 `
  --max-features 1800 `
  --min-matches 20 `
  --min-pnp-inliers 8
```

Reference baseline metrics:

| Run | Translation RMSE |
|---|---:|
| Tuned PnP, 500 frames full resolution | 0.2166 |
| Tuned PnP, 1000 frames full resolution | 0.7587 |
| Full half-scale VO | 8.4356 |
| Full half-scale VO with naive loop closure | 6.4475 |

The full-sequence VO baseline drifts substantially and is not the promoted
final Stage 3 result.

## Pose Graph Result

Implemented:

- Visual loop-edge estimation from RGB-D data.
- Pose graph construction from odometry plus accepted loop constraints.
- Robust SciPy `least_squares` pose-graph optimization.
- Keyframe-stride optimization for faster full-sequence runs.
- Duplicate loop-endpoint spacing to avoid over-weighting near-identical
  end-to-start loop edges.
- Official evaluator wrapper around the assignment `eval.py` / `eval_ate.py`.

Current promoted run:

```powershell
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_pose_graph_keyframe_stride8_nfev50" `
  --input-trajectory "outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --run-pose-graph `
  --pose-graph-keyframe-stride 8 `
  --max-loop-edges 3 `
  --loop-endpoint-spacing 100 `
  --optimizer-max-nfev 50 `
  --run-official-eval
```

Outputs:

```text
estimated_camera_trajectory_pose_graph.txt
loop_candidates.csv
loop_candidate_summary.json
pose_graph_edges.csv
pose_graph_summary.json
pose_graph_optimization_summary.json
trajectory_metrics_pose_graph.json
stage3_comparison_summary.json
official_eval_summary.json
```

Measured full-sequence results:

| Trajectory | Internal RMSE | Official ATE |
|---|---:|---:|
| Raw VO from `stage3_pose_graph_full` | 7.1671 | 3.7007 |
| Old full pose graph, 4396 optimized nodes | 6.6549 | 3.6164 |
| Keyframe pose graph, stride 16 / 50 evals | 6.0304 | 3.4319 |
| Keyframe pose graph, stride 8 / 50 evals | **5.2430** | **3.0305** |

The stride-8 keyframe run optimizes 552 graph nodes and exports a full
4396-pose trajectory after interpolating optimized keyframe corrections. It is
the current Stage 3 final result.

## Visualization

Static trajectory comparison:

```powershell
& $python_open3d -m sfm_reconstruction.stage3_visualize `
  --ground-truth "..\Experiments\Stage_3_Data\stage3\gt_camera_trajectory.txt" `
  --trajectory "Raw VO=outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --trajectory "Full pose graph=outputs\stage3_pose_graph_full\estimated_camera_trajectory_pose_graph.txt" `
  --trajectory "Keyframe pose graph=outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --output "outputs\stage3_pose_graph_keyframe_stride8_nfev50\trajectory_comparison.png"
```

Interactive trajectory viewer:

```powershell
& $python_open3d -m sfm_reconstruction.stage3_open3d_viewer `
  --ground-truth "..\Experiments\Stage_3_Data\stage3\gt_camera_trajectory.txt" `
  --trajectory "Raw VO=outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --trajectory "Full pose graph=outputs\stage3_pose_graph_full\estimated_camera_trajectory_pose_graph.txt" `
  --trajectory "Keyframe pose graph=outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt"
```

Sampled RGB-D scene fused with the estimated trajectory:

```powershell
& $python_open3d -m sfm_reconstruction.stage3_open3d_viewer `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --scene-trajectory "outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --scene-frame-stride 10 `
  --scene-pixel-stride 4 `
  --scene-max-points 1200000 `
  --scene-voxel-size 0.02 `
  --scene-remove-outliers `
  --export-scene-ply "outputs\stage3_pose_graph_keyframe_stride8_nfev50\stage3_scene_dense_voxel.ply" `
  --trajectory "Keyframe pose graph=outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --no-align `
  --trajectory-smoothing-window 9 `
  --point-size 1.5
```

GT-pose fused scene comparison:

- `stage3_scene_gt_pose_dense.ply`: RGB-D frames fused with
  `gt_camera_trajectory.txt`.
- `stage3_scene_est_pose_dense.ply`: same frames fused with the estimated
  keyframe pose-graph trajectory.
- `stage3_scene_est_pose_dense_aligned_to_gt.ply`: diagnostic-only similarity
  alignment of the estimated fused scene into the GT trajectory frame.

The GT-aligned scene overlay is useful for debugging gauge and drift, but it is
not a valid GT-free reconstruction output.

## Remaining Research Direction

The implementation is complete for the current assignment scope. The most
credible next accuracy path is learning-based matching:

- Stage 1/2: compare SuperPoint/LightGlue, DISK/LightGlue, or LoFTR against
  SIFT/SIFT+AKAZE for wider-baseline matches and object coverage.
- Stage 3: use learned matching first for loop-edge discovery, then for VO if
  loop closure alone is insufficient.
- Gaussian Splatting is a downstream visualization/demo direction once camera
  poses are stable; it should not be treated as a trajectory-drift fix.
