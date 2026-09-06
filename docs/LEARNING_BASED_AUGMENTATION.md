# Learning-Based Augmentation Roadmap

The classical pipeline is complete enough for the current Stage 1, Stage 2,
and Stage 3 deliverables. Learned loop matching is now available as an optional
Stage 3 frontend. Neural rendering remains downstream work after poses are
stable.

## Recommended Priority

1. Learned local matching for correspondences and loop closures.
2. Pose-graph / bundle-adjustment use of the stronger correspondences.
3. Dense fusion or mesh reconstruction for qualitative geometry.
4. Gaussian Splatting as a downstream visualization/demo.

Gaussian Splatting should not be treated as a trajectory-drift fix. It depends
on camera poses; inaccurate poses produce smeared or duplicated splats.

## Stage 1

Stage 1 uses supplied correspondences, so learned matching is not required for
the official assignment path. It is still useful as an ablation:

- Recreate the supplied correspondence graph with SuperPoint/LightGlue,
  DISK/LightGlue, or LoFTR.
- Compare track counts, track conflicts, initialization pair quality, and final
  pose metrics against the supplied correspondences.
- Keep the existing geometric checks: essential matrix RANSAC, cheirality,
  triangulation angle, reprojection thresholds, and bundle adjustment.

Success criterion: learned matches should reproduce or improve the Stage 1
pose metrics without increasing unstable tracks.

For this ablation, run the Stage 2 correspondence-generation path against the
Stage 1 image layout with `--stage 2 --feature-mode superpoint-lightglue`; the
official Stage 1 command continues to use the supplied correspondences.

The full ablation now supports `--matching-pair-source supplied`, which reuses
the exact 366-pair Stage 1 graph while replacing only the match data.
`--learned-cycle-filter` applies exact three-view feature-cycle filtering before
track construction; pairs with no possible triangle retain their essential
matrix inliers. The supplied Chamfer definition is available through
`sfm_reconstruction.stage1_evaluation`.

Full box benchmark:

| Correspondences | Cameras | Points | Conflicts | Mean rotation | Mean translation | Chamfer |
|---|---:|---:|---:|---:|---:|---:|
| Supplied baseline | **46/46** | 5,746 | **239** | **0.875 deg** | **0.0666** | **0.5124** |
| LightGlue 0.2, same graph | 46/46 | 10,964 | 6,851 | 1.682 deg | 0.1680 | 0.7787 |
| LightGlue 0.3, same graph | 46/46 | **11,205** | 5,847 | 2.087 deg | 0.1980 | 0.6111 |
| LightGlue 0.3 + cycle filtering | 46/46 | 4,324 | 695 | 2.057 deg | 0.2355 | 0.8175 |
| Supplied + cycle-filtered learned augmentation, supplied anchor | 46/46 | 10,084 | 934 | 9.331 deg | 0.6480 | 0.6204 |

Cycle filtering reduces learned conflicts by almost an order of magnitude, but
none of the learned replacement or augmentation variants improves the supplied
Stage 1 solution. The supplied correspondences remain promoted. The learned
implementation is retained as a reproducible negative ablation, not as a
replacement path.

### Fixed-pose learned point augmentation

The productive integration keeps the promoted supplied reconstruction
unchanged and uses learned tracks only to add independently validated points.
`sfm_reconstruction.stage1_augmentation` freezes all supplied cameras and
points, triangulates cycle-filtered learned tracks in at least three views,
optimizes each new point alone, and applies:

- positive depth in every supporting camera;
- median/max reprojection limits of 1.0/2.5 px;
- at least 2 degrees triangulation angle;
- agreement between independent view-pair triangulations;
- robust bounds derived only from the baseline cloud;
- duplicate rejection against existing supplied points.

| Metric | Supplied baseline | Fixed-pose augmentation |
|---|---:|---:|
| Camera poses | promoted | **identical/frozen** |
| Points | 5,746 | **6,888** |
| Accepted learned points | 0 | 1,142 |
| GT-to-estimate distance | 0.40805 | **0.40171** |
| Estimate-to-GT distance | 0.10437 | **0.10291** |
| Total Chamfer | 0.51242 | **0.50461** |

Both directed distances and total Chamfer improve while camera metrics remain
exactly those of the supplied baseline. Promote this as an augmented point
cloud; supplied correspondences remain the official camera-reconstruction path.

```powershell
& $python_open3d -m sfm_reconstruction.stage1_augmentation `
  --dataset "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box" `
  --baseline-result "outputs\stage1_box_filter_regression" `
  --learned-correspondence-dir `
    "outputs\stage1_box_lightglue_supplied_graph_filter03_cycle15\matching_cache\correspondences" `
  --output-dir "outputs\stage1_box_fixed_pose_learned_augmentation"
```

Representative pure learned command:

```powershell
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box" `
  --output-dir "outputs\stage1_box_lightglue_supplied_graph_filter03_cycle15" `
  --feature-mode superpoint-lightglue `
  --matching-pair-source supplied `
  --learned-filter-threshold 0.3 `
  --learned-cycle-filter `
  --learned-min-cycle-matches 15 `
  --learned-device cuda `
  --max-features 2500 `
  --bundle-adjustment-max-nfev 20 `
  --write-reprojection-diagnostics
```

```powershell
& $python_open3d -m sfm_reconstruction.stage1_evaluation `
  --ground-truth "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box\gt_points.ply" `
  --result-dir "outputs\stage1_box_lightglue_supplied_graph_filter03_cycle15"
```

## Stage 2

Stage 2 is the strongest immediate target for learned matching because object
coverage and wide-baseline matching still limit dense reconstruction quality.

Candidate integrations:

- Add `--feature-mode superpoint-lightglue` or a separate learned-matching
  cache command.
- Preserve the current correspondence cache format where possible, so the
  existing track builder and SfM backend can stay unchanged.
- Compare against SIFT-only and SIFT+AKAZE on milk and boot.
- Use the existing reprojection diagnostics and dense-fusion outputs to judge
  whether coverage improves without introducing geometric noise.

The `superpoint-lightglue` feature mode is now implemented for Stage 2. It
keeps AprilTag masking, essential-matrix filtering, correspondence caching,
track construction, and incremental reconstruction unchanged:

```powershell
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --output-dir "outputs\stage2_milk_superpoint_lightglue" `
  --feature-mode superpoint-lightglue `
  --learned-device auto `
  --learned-filter-threshold 0.2 `
  --max-features 2500 `
  --mask-apriltags `
  --min-track-observations 3 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20
```

Initial milk benchmark:

| Frontend | Points | Track conflicts | Mean rotation | Mean translation | Median translation |
|---|---:|---:|---:|---:|---:|
| SIFT baseline | 1,739 | 116 | 6.779 deg | **0.06254** | 0.05262 |
| SuperPoint+LightGlue, threshold 0.1 | 1,764 | 3,135 | 7.268 deg | 0.14646 | 0.13408 |
| SuperPoint+LightGlue, threshold 0.2 | **1,923** | 2,645 | **6.756 deg** | 0.06275 | **0.04864** |

Threshold `0.2` is the learned Stage 2 default. It recovers a competitive pose
solution with more points, but its high conflict count and heavier reprojection
tail mean it remains an experiment rather than the promoted milk result.

Wide-baseline augmentation is also implemented as an opt-in experiment. DINOv2
retrieves nonlocal image pairs, SuperPoint+LightGlue verifies them, essential
geometry and a 4x4 spatial-coverage test reject weak pairs, and triangle-cycle
consistency filters individual matches. `--wide-max-pairs-per-image` limits
retrieved-edge degree so visually redundant proposals cannot saturate the track
graph. Every candidate and its rejection reason is recorded in
`matching_cache/pair_diagnostics.csv`.

```powershell
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --output-dir "outputs\stage2_milk_lightglue_wide_degree1" `
  --feature-mode superpoint-lightglue `
  --learned-device cuda `
  --learned-filter-threshold 0.2 `
  --max-features 2500 `
  --pair-window 3 `
  --wide-baseline `
  --wide-min-frame-gap 8 `
  --wide-max-pairs-per-image 1 `
  --wide-retrieval-device cuda `
  --mask-apriltags `
  --min-track-observations 3 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20 `
  --write-reprojection-diagnostics
```

Milk wide-baseline benchmark:

| Frontend | Wide pairs | Points | Conflicts | Median / p90 reprojection | Mean mesh proxy | Mean translation |
|---|---:|---:|---:|---:|---:|---:|
| SuperPoint+LightGlue local | 0 | **1,923** | **2,645** | **0.787 / 3.667 px** | **0.01510** | **0.06275** |
| Wide, unrestricted cycle-filtered | 77 | 1,638 | 3,898 | 0.832 / **3.015 px** | 0.03036 | 0.06367 |
| Wide, one retrieved edge per image | 15 | 1,623 | 2,968 | 0.830 / 3.176 px | 0.02539 | 0.06427 |

The degree cap materially reduces conflicts relative to unrestricted retrieval,
but neither wide run passes the promotion gate: both worsen milk pose error and
nearest-mesh-vertex distance. Therefore boot and dense-fusion regeneration are
intentionally deferred. The local learned run remains the stronger Stage 2
learned experiment.

### Pose-only wide constraints

`--wide-pose-only` keeps the stable local track graph and writes retrieved
correspondences to a separate cache. Consistent wide pairs become low-weight
relative-pose constraints; they are never unioned into tracks. After robust pose
graph optimization, the unchanged local tracks are retriangulated and locally
bundle-adjusted. Candidate decisions are written to
`wide_pose_constraints.csv`, and optimizer statistics to
`stage2_pose_refinement.json`.

The strongest GT-free consistency gate retained one loop-like pair,
`84 -> 1387`, with 340 relative-pose inliers, 0.87-degree rotation
disagreement, and 0.87-degree translation-direction disagreement.

| Milk result | Points | Mean / median rotation | Mean / median translation | Median / p90 reprojection | Mesh proxy |
|---|---:|---:|---:|---:|---:|
| Local learned baseline | 1,923 | 6.756 / 5.421 deg | **0.06275 / 0.04864** | 0.787 / 3.667 px | **0.01510** |
| Retriangulation control, zero wide weight | **2,154** | 6.792 / 5.074 deg | 0.06416 / 0.05130 | **0.730 / 1.868 px** | 0.02048 |
| One wide pose constraint | 2,128 | **6.736 / 4.935 deg** | 0.06341 / 0.04912 | 0.753 / 1.906 px | 0.01874 |

The learned constraint genuinely improves the zero-weight control in rotation,
translation, and mesh distance, while the second triangulation/BA pass supplies
most of the point-count and reprojection gain. It still does not beat the local
learned baseline on translation or mesh distance, so it is not promoted and
boot/dense regeneration remains gated off.

```powershell
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --output-dir "outputs\stage2_milk_lightglue_wide_pose_single_ba" `
  --feature-mode superpoint-lightglue `
  --learned-device cuda `
  --learned-filter-threshold 0.2 `
  --max-features 2500 `
  --pair-window 3 `
  --wide-baseline `
  --wide-pose-only `
  --wide-min-frame-gap 8 `
  --wide-max-pairs-per-image 1 `
  --wide-pose-rotation-weight 0.15 `
  --wide-pose-translation-weight 0.05 `
  --wide-retrieval-device cuda `
  --mask-apriltags `
  --min-track-observations 3 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20 `
  --write-reprojection-diagnostics
```

The mesh proxy is now reproducible with:

```powershell
& $python_open3d -m sfm_reconstruction.stage2_evaluation `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --result-dir "outputs\stage2_milk_lightglue_wide_degree1"
```

It writes `stage2_mesh_proxy.json` with full-cloud historical metrics and the
fixed-crop variants used by the supplied milk evaluator. Both use the evaluator
scale and nearest supplied-mesh-vertex distance. This is a validation proxy,
not a hidden-evaluator replacement.

Success criteria:

- All cameras still register.
- Reprojection error remains controlled.
- Milk mesh-proxy distance improves or stays comparable.
- Boot gains useful object coverage without returning to marker-dominated or
  distant-cluster artifacts.

## Stage 3

Stage 3 should use learned matching first for loop closure, not as a full VO
replacement. The current best result uses one strong end-to-start RGB-D loop
edge; further ATE gains likely require distinct non-redundant loop constraints.

Implemented experimental frontend:

```text
--loop-matcher superpoint-lightglue
```

Global learned place recognition is also implemented:

```text
--loop-candidate-mode dinov2
```

DINOv2 ViT-S/14 descriptors retrieve long-range candidates across the full
sequence. A separate 500-frame temporal exclusion prevents ordinary local
motion from becoming a loop edge, proposal spacing preserves alternate
representatives of a revisit, and +/-4-frame refinement lets geometric scoring
recover the strongest exact endpoints.

It replaces only the AKAZE/SIFT correspondence frontend used to propose loop
edges. Depth lookup, RGB-D PnP / 3D-3D verification, acceptance thresholds,
endpoint spacing, pose-graph optimization, and official evaluation are shared
with the classical baseline.

Install the optional runtime dependencies, followed by the official LightGlue
repository (the pretrained weights download on first use):

```powershell
python -m pip install -e .[learned]
python -m pip install --no-deps `
  "git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8"
```

Run a directly comparable learned loop-closure experiment:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_pose_graph_lightglue_stride8" `
  --input-trajectory "outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --run-pose-graph `
  --loop-matcher superpoint-lightglue `
  --learned-device auto `
  --pose-graph-keyframe-stride 8 `
  --max-loop-edges 3 `
  --loop-endpoint-spacing 100 `
  --optimizer-max-nfev 50 `
  --run-official-eval
```

`loop_candidates.csv` and `loop_candidate_summary.json` record the matcher so
the learned and classical runs can be compared without changing output format.

Recommended retrieval command:

```powershell
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_pose_graph_dinov2_akaze_refine4" `
  --input-trajectory "outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --run-pose-graph `
  --loop-candidate-mode dinov2 `
  --retrieval-stride 8 `
  --retrieval-min-separation 500 `
  --retrieval-max-candidates 50 `
  --retrieval-proposal-spacing 8 `
  --retrieval-refine-radius 4 `
  --retrieval-refine-top-k 10 `
  --retrieval-min-similarity 0.75 `
  --retrieval-device cuda `
  --loop-matcher classical `
  --feature-mode akaze `
  --pose-graph-keyframe-stride 8 `
  --max-loop-edges 1 `
  --loop-endpoint-spacing 100 `
  --optimizer-max-nfev 50 `
  --run-official-eval
```

Initial benchmark on the 25 existing end-to-start candidates:

| Frontend | Resolution | Selected edge | Inliers | Median error | Internal RMSE | Official ATE |
|---|---|---|---:|---:|---:|---:|
| AKAZE baseline | full | 4387 -> 32 | 456 | 0.500 px | 5.2430 | **3.0305** |
| SuperPoint+LightGlue | half | 4395 -> 16 | 145 | 1.123 px | **3.3014** | 3.0541 |
| SuperPoint+LightGlue | full | 4379 -> 16 | 343 | 0.760 px | 5.3864 | 3.3225 |

The learned frontend is operational but is not promoted over AKAZE yet. The
half-resolution run improves the unaligned internal translation metric, while
the classical edge remains best under the official Horn-aligned evaluator.
This motivated the global retrieval benchmark below.

Global retrieval benchmark:

| Candidate frontend | Local verifier | Result |
|---|---|---|
| DINOv2, 500-frame exclusion | SuperPoint+LightGlue | One valid long-range edge; official ATE 3.0776 |
| DINOv2 stride 8 + local refinement | AKAZE | Recovered `4387 -> 32`; official ATE **3.0305** |

The retrieval pipeline autonomously recovers the promoted loop edge, but found
no additional distinct long-range edge that improves the official ATE on this
sequence. Short-separation DINOv2 matches must not be used as loop closures:
they pass local geometry but merely duplicate odometry constraints.

Implemented order:

1. Use learned matching to search wider loop-candidate windows.
2. Keep the same RGB-D PnP / 3D-3D geometric verification.
3. Enforce loop-endpoint spacing to avoid duplicate constraints.
4. Only then test learned matching for frame-to-frame VO.

Success criteria:

- More accepted loop edges must be spatially distinct, not near-copies of the
  existing end-to-start edge.
- Official ATE must improve over `3.0305`.
- Scene fusion should show less duplicate geometry without using GT alignment.

## Gaussian Splatting

3D Gaussian Splatting is applicable as a visual demo initialized from the
estimated cameras and RGB frames, but it is downstream of pose quality.

Recommended usage:

- Export cameras/images to a COLMAP-style layout.
- Initialize from the current sparse/dense reconstruction when possible.
- Train on Stage 2 object scenes first, where the scene is bounded and camera
  motion is shorter.
- Use Stage 3 only after trajectory drift is acceptable, because long-loop
  drift will smear the learned radiance field.

Success criterion: splatting should be presented as a rendering demo, not as
evidence that the geometric reconstruction or trajectory is correct.
