# Reconstruction-to-memory integration

`persistent_scene_memory.reconstruction_bridge` is the first real connection
between the SfM stack and persistent VLA inputs. It reads the rich PLY produced
by `sfm_reconstruction`, turns each tracked point into a `GeometryObservation`,
updates the confidence-aware memory, and writes both portable memory JSON and a
fixed-size adapter tensor for every input snapshot.

The SfM CLI can now create those inputs directly from one continuing map. This
preserves the reconstruction gauge and track IDs instead of relying on
independently restarted reconstructions.

## Integrated incremental run

Add `--write-incremental-memory` to a normal Stage 1 or Stage 2 reconstruction:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\boot" `
  --output-dir "outputs\stage2_boot_with_memory" `
  --write-incremental-memory
```

The reconstruction emits a snapshot after the initialization pair and after
each successful PnP camera registration. Snapshot tracks contain observations
only from cameras registered at that point, preventing future-frame RGB or
diagnostics from leaking into earlier memory frames.

The integrated output adds:

- `incremental_snapshots/manifest.json` and one camera JSON, full-map rich PLY,
  and current-observation rich PLY per registration event;
- `incremental_memory/memory_frame_*.json`;
- `incremental_memory/adapter_frame_*.json`;
- `incremental_memory/summary.json`.

These snapshots are online estimates before final global bundle adjustment.
The ordinary final reconstruction outputs retain the existing post-adjustment
behavior. Registration events follow SfM's best-PnP order, which can differ
from image acquisition order; the manifest records the image IDs explicitly.
Only `observed_points_rich.ply` is sent to memory at each event. Previously
mapped points absent from the newly registered camera are therefore not updated:
their `last_seen` age increases and their confidence decays normally.

## Input contract

Each input must be an ASCII `estimated_points_rich.ply` containing world XYZ,
RGB, `track_id`, registered-observation count, mean reprojection error, and
maximum triangulation angle. Inputs are processed in command-line order.

For a sequence, every snapshot must use:

- the same world coordinate frame and scale;
- persistent `track_id` values for the same physical points;
- unique track IDs within each snapshot.

An independently restarted SfM reconstruction usually does not satisfy this
contract. Such outputs must be aligned and associated before using them as a
sequence. A single final reconstruction is valid as a one-snapshot smoke test.

## Confidence and features

The portable point feature is normalized RGB. Geometric confidence is:

```text
clip(registered_observations / 3)
* exp(-mean_reprojection_error / 2 px)
* clip(max_triangulation_angle / 5 deg)
```

The three scales are CLI options and must be calibrated on validation data
before treating confidence as probabilistic. Confidence then decays inside the
persistent memory when a track is absent from later snapshots.

## Run

```powershell
$env:PYTHONPATH = "src"
& $python -m persistent_scene_memory.reconstruction_bridge `
  "outputs\snapshot_000\estimated_points_rich.ply" `
  "outputs\snapshot_001\estimated_points_rich.ply" `
  --output-dir "artifacts\reconstruction_memory"
```

The output directory contains:

- `memory_frame_*.json`: stable keys, world positions, RGB features,
  confidence, and last-seen indices;
- `adapter_frame_*.json`: padded `[xyz, confidence, age, rgb]` values, mask,
  and row-to-track keys;
- `summary.json`: inputs, counts, confidence settings, and the coordinate
  contract.

## Remaining Gate 2 work

The file/interface gap and same-map incremental export are implemented, but the
scientific gate is not complete. Next, run the path on sequential real data,
evaluate map accuracy and confidence calibration, and decide whether policy
experiments require strict acquisition-order registration or object-level
association beyond point tracks. Independently restarted maps still require
explicit alignment and association.

## Verified real-data smoke run

The integrated path was run on the first eight images of the supplied Stage 1
box sequence:

```powershell
& $python -m sfm_reconstruction `
  --dataset "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box" `
  --max-images 8 `
  --output-dir "outputs\stage1_box_incremental_memory_8" `
  --bundle-adjustment-max-nfev 20 `
  --write-incremental-memory
```

Measured reconstruction result:

- 8/8 cameras registered and 741 final points;
- mean rotation error `0.5322 deg`;
- mean translation error `0.04696`;
- seven registration/memory events.

At the last event, the full map contained 741 points. The current camera
observed 320; the confidence threshold retained 725 memory tokens, comprising
320 current and 405 persisted tokens with ages up to six events. The policy
adapter contained 32 unique active keys in a `32 x 8` tensor. All referenced
snapshot files were present, and the full repository verification remained
`114 passed` after adding the interactive viewer camera-fit regression test.

## Interactive timeline viewer

Open the generated memory sequence with Open3D:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m persistent_scene_memory.memory_viewer `
  --memory-dir "outputs\stage1_box_incremental_memory_8\incremental_memory"
```

Controls:

- `N`: next memory event;
- `P`: previous memory event;
- `M`: cycle current/persisted, confidence, and RGB coloring;
- `R`: reset and fit the camera to the memory cloud;
- `Q`: close the viewer.

In the default status mode, current observations are green and persisted
memory is orange, with brightness modulated by confidence.
