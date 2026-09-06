# Reproducibility

This project is organized so the code can be committed separately from large
datasets, generated outputs, and local caches.

## What To Commit

Commit:

```text
README.md
handoff.md
pyproject.toml
requirements-repro.txt
src/
tests/
scripts/
docs/
assets/demo/boot_dense_open3d.png
assets/demo/boot_dense_open3d.mp4
assets/demo/milk_dense_open3d.png
assets/demo/milk_dense_open3d.mp4
deliverables/README.md
```

Do not commit:

```text
outputs/
deliverables/*.zip
datasets
matching caches
Python cache files
matplotlib cache folders
```

## Environment Snapshot

Verified on Windows with Python `3.10.13`.

Pinned package snapshot:

```text
numpy==1.23.5
opencv-python==4.12.0.88
scipy==1.15.3
pytest==8.3.3
trimesh==3.23.5
matplotlib==3.8.4
open3d==0.19.0
```

Install from the project root:

```powershell
python -m pip install -e .[dev,evaluation,visualization,open3d]
```

For the local machine used during development:

```powershell
$python = "python"
$python_open3d = "python"
```

## Verification

Run unit tests plus artifact regression checks:

```powershell
.\scripts\run_verification.ps1
```

Or specify an interpreter:

```powershell
.\scripts\run_verification.ps1 -Python "C:\path\to\open3d-env\python.exe"
```

The verification runner performs:

- `pytest -q -p no:cacheprovider`
- summary metric checks for the selected output folders
- PLY vertex-count checks for sparse and cleaned dense deliverables
- MP4 metadata checks for demo videos
- SHA-256 check for the local deliverables zip if it exists

Current expected unit test result:

```text
76 passed
```

Current focused Stage 3 result:

```text
outputs/stage3_pose_graph_keyframe_stride8_nfev50/
official ATE: 3.030521
internal translation RMSE: 5.242974
```

Stage 3 visualization and diagnostics may generate additional ignored outputs:

```text
trajectory_comparison.png
stage3_scene_sampled.ply
stage3_scene_dense_voxel.ply
stage3_scene_gt_pose_dense.ply
stage3_scene_est_pose_dense.ply
stage3_scene_est_pose_dense_aligned_to_gt.ply
```

The GT-aligned scene cloud is diagnostic only and should not be used as a
GT-free reconstruction result.

## Deliverables Archive

Local archive:

```text
deliverables/reconstruction_deliverables_2026-06-18.zip
```

SHA-256:

```text
D5C1BBD635358D5FB7B9BFA1B533CCCD039F864E46927BC0F86AB600D9E359EA
```

The archive contains:

```text
outputs/stage1_box_filter_regression/
outputs/stage2_milk_masked_track3_ba/
outputs/stage2_boot_improved/
outputs/stage2_milk_masked_track4_ba/
outputs/stage2_milk_sift_akaze_probe/
outputs/stage2_boot_sift_akaze_probe/
```

The archive is intentionally ignored by Git. Store it externally, attach it to
a release, or keep it as a local backup.

## Before Pushing

Initialize Git from `sfm_reconstruction/`, then inspect staged files carefully:

```powershell
git init
git add .
git status
```

Before committing, confirm that these are not staged:

```text
outputs/
deliverables/reconstruction_deliverables_2026-06-18.zip
__pycache__/
pytest-cache-files-*/
.matplotlib*/
```

Then commit:

```powershell
git commit -m "Package SfM reconstruction project"
```
