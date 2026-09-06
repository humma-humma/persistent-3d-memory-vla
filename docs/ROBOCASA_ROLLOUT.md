# RoboCasa RGB-D Rollout Bridge

`persistent_scene_memory.robocasa_rollout` connects the local RoboCasa
installation to the existing memory and geometry-token interfaces. It records
only locally generated simulation data; no demonstration dataset is required.

## Run

From the repository root, with the local RoboCasa and RoboSuite sources on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src;..\third_party\robocasa;..\third_party\robosuite"
$env:MUJOCO_GL = "glfw"
$python = "python"

& $python -m persistent_scene_memory.robocasa_rollout `
  --output-dir outputs\robocasa_pick_place `
  --steps 20 `
  --seed 0
```

The command explicitly uses the supported `pretrain` split and the installed
`lightwheel` object registry. Actions are zero-valued for this first sensing
baseline; a policy or scripted expert can be supplied through the Python
`record_episode(..., action_fn=...)` interface.

To generate privileged scripted labels and a standalone per-frame viewer:

```powershell
& $python -m persistent_scene_memory.robocasa_rollout `
  --output-dir outputs\robocasa_oracle_pick_place `
  --steps 200 `
  --seed 0 `
  --camera-size 96 `
  --action-source oracle `
  --viewer
```

The task-specific oracle uses simulator object and cabinet poses only to
produce action labels. Those privileged values are recorded as diagnostics but
are not included in the policy observation. It coordinates the mobile base,
torso, arm, and gripper through approach, grasp, lift, cabinet approach,
insertion, release, and retreat phases. The seed-0 validation rollout reaches
RoboCasa success at frame 128, latches into a zero-action `done` state on the
next frame, and remains successful through frame 179.

## Output contract

`episode/manifest.json` defines the task text, camera names, state/action keys,
and ordered frame files. Each compressed `frame_*.npz` contains:

- synchronized RGB and metric-depth images from all three policy cameras;
- the intrinsic matrix and camera-to-world transform for every camera;
- robot state at decision time and the action applied from that state;
- transition reward, termination, truncation, and success in the manifest.

Camera extrinsics are recorded every frame because the wrist camera can move.
`RoboCasaEpisode.world_points()` backprojects the metric depth directly into
MuJoCo world coordinates.

`memory/` contains JSON memory snapshots plus fixed-size NPZ policy tensors.
The initial identity baseline quantizes points into world voxels, uses distinct
camera support as confidence, and feeds the resulting observations through the
existing `PersistentSceneMemory` and `GeometryTokenAdapter`. This validates the
complete data path but is not object-level association; learned or simulator
segmentation-based object identity is a later experiment.

`viewer.html` is self-contained and can be opened directly. Its frame slider
shows the three synchronized cameras, current-versus-persisted world voxels,
confidence, oracle phase and target diagnostics, applied action components,
reward, and RoboCasa success. The separate `robocasa-rollout-viewer` entry point
can rebuild the page from any compatible episode and memory directory.
