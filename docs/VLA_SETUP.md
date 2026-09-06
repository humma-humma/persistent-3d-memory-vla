# VLA milestone setup

## Environment

Python 3.10 or newer is required. The reproducible core environment is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-vla.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The final `--no-deps` matters: the lock file owns dependency resolution. For a
smaller memory-only installation, use `python -m pip install -e .[dev]`.

## Verify the milestone

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m persistent_scene_memory.benchmark --output-dir artifacts\controlled_demo --seed 7 --trials 40
.\.venv\Scripts\python.exe -m persistent_scene_memory.vla_smoke --backend mock --output artifacts\vla_smoke.json
```

The mock backend is an offline contract test. It verifies image, instruction,
world-token adapter, and seven-dimensional action plumbing without claiming
pretrained-policy quality.

## Optional pretrained OpenVLA smoke test

Install the VLA extras and run on a CUDA GPU with enough memory for OpenVLA-7B:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[vla]
.\.venv\Scripts\python.exe -m persistent_scene_memory.vla_smoke --backend openvla --model-id openvla/openvla-7b `
  --output artifacts\openvla_smoke.json
```

This downloads model weights on first use. The smoke image is synthetic and
the command uses OpenVLA's BridgeData action normalization key, so its action
is only an interface check. The geometry is supplied through the augmented
instruction; learned geometry-token injection requires downstream training and
is outside this milestone.

## Local SmolVLA smoke test (8 GB GPU)

The verified local setup uses LeRobot 0.4.4 and CUDA PyTorch 2.8.0 in a
dedicated environment:

```powershell
$basePython = "python"
& $basePython -m venv .venv-smolvla
$smolPython = ".\.venv-smolvla\Scripts\python.exe"
& $smolPython -m pip install --upgrade pip
& $smolPython -m pip install "lerobot[smolvla]==0.4.4"
& $smolPython -m pip install --force-reinstall `
  torch==2.8.0+cu126 torchvision==0.23.0+cu126 `
  --index-url https://download.pytorch.org/whl/cu126
& $smolPython -m pip install --no-deps fsspec==2026.2.0
& $smolPython -m pip install `
  numpy==2.2.6 opencv-python==4.11.0.86 `
  opencv-python-headless==4.11.0.86 scipy==1.15.3
& $smolPython -m pip install -e . --no-deps --no-build-isolation
& $smolPython -m pip check
```

The official public checkpoint is stored in the ignored local model directory:

```powershell
& $smolPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/smolvla_base', local_dir='models/smolvla_base', allow_patterns=['config.json','model.safetensors','policy_preprocessor*','policy_postprocessor*','README.md'])"
```

Run real inference:

```powershell
& $smolPython -m persistent_scene_memory.vla_smoke `
  --backend smolvla `
  --checkpoint models\smolvla_base `
  --device cuda `
  --output artifacts\smolvla_smoke.json
```

Verified on the local RTX 2080 (8,192 MiB): the checkpoint returned a finite
`1 x 6` action, with 1,213 MiB peak PyTorch allocation and 46.4 seconds for
model load plus first inference. The input is synthetic, so the action is an
interface result rather than a meaningful robot-control evaluation. The base
checkpoint is intended for task-specific fine-tuning.

## Real-data pre-fine-tuning baseline

The compatible public dataset is `lerobot/svla_so100_pickplace`: 50 episodes,
two RGB cameras, six-dimensional state, and six-dimensional action. It is
stored under the ignored `datasets/svla_so100_pickplace/` directory. LIBERO was
not used because its eight-dimensional state and seven-dimensional action do
not match this checkpoint.

Run the deterministic, fully offline episode-0 evaluation:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& $smolPython -m persistent_scene_memory.real_data_eval `
  --checkpoint models\smolvla_base `
  --dataset-dir datasets\svla_so100_pickplace `
  --anchors 0 50 100 `
  --device cuda `
  --output artifacts\smolvla_real_data_baseline.json
```

Each anchor predicts a 50-step action chunk from the real top/wrist images,
motor state, and instruction. Dataset statistics normalize the input and return
predictions to recorded motor units. The zero-shot base checkpoint reaches
normalized RMSE `0.7736`; holding the anchor motor state reaches `0.4924`.
SmolVLA therefore does not beat the trivial control before task fine-tuning.
This is a useful negative control, not a task-success result.

## Low-memory task-adaptation pilot

The verified adapter freezes the VLM and action expert and trains only 1.64M
projection/time-conditioning parameters. Episodes 1–10 provide 50 training
anchors; episode 0 remains held out.

```powershell
& $smolPython -m persistent_scene_memory.fine_tune_pilot `
  --checkpoint models\smolvla_base `
  --dataset-dir datasets\svla_so100_pickplace `
  --output-dir outputs\smolvla_projection_pilot_500_ep1_10 `
  --episodes 1 2 3 4 5 6 7 8 9 10 `
  --anchors 0 50 100 150 200 `
  --steps 500 `
  --image-size 256 `
  --device cuda
```

Evaluate the held-out episode at the matched resolution:

```powershell
& $smolPython -m persistent_scene_memory.real_data_eval `
  --checkpoint outputs\smolvla_projection_pilot_500_ep1_10 `
  --dataset-dir datasets\svla_so100_pickplace `
  --anchors 0 50 100 `
  --image-size 256 `
  --device cuda `
  --output artifacts\smolvla_projection_pilot_500_ep1_10_eval.json
```

The adapter reduces normalized RMSE from `0.7107` to `0.4914` relative to the
matched base model and narrowly beats hold-state (`0.4924`). Raw RMSE remains
worse than hold-state, so this is a pilot result rather than a promoted policy.

### Six-episode promotion gate

Run matched base and adapted evaluations on six held-out episodes, then compare
them with the fixed gate:

```powershell
& $smolPython -m persistent_scene_memory.real_data_eval `
  --checkpoint models\smolvla_base `
  --dataset-dir datasets\svla_so100_pickplace `
  --episodes 0 11 12 13 14 15 `
  --anchors 0 50 100 --image-size 256 --device cuda `
  --output artifacts\smolvla_base_heldout6_eval.json

& $smolPython -m persistent_scene_memory.real_data_eval `
  --checkpoint outputs\smolvla_projection_pilot_500_ep1_10 `
  --dataset-dir datasets\svla_so100_pickplace `
  --episodes 0 11 12 13 14 15 `
  --anchors 0 50 100 --image-size 256 --device cuda `
  --output artifacts\smolvla_adapted_heldout6_eval.json

& $smolPython -m persistent_scene_memory.promotion `
  --base artifacts\smolvla_base_heldout6_eval.json `
  --adapted artifacts\smolvla_adapted_heldout6_eval.json `
  --output artifacts\smolvla_promotion_gate.json
```

Across 900 target actions, adaptation improves RMSE 40.2% versus base and wins
all six episodes. It is nevertheless 18.7% worse than hold-state in aggregate
RMSE and 25.4% worse in normalized RMSE. The promotion gate therefore fails.

## Outputs

- `baseline_results.csv`: every controlled condition and baseline.
- `baseline_summary.json`: mean metrics used in the milestone table.
- `tokens.json`: portable token export with stable IDs.
- `vla_smoke.json`: offline VLA adapter/action contract.
- `smolvla_smoke.json`: real local SmolVLA CUDA inference result and peak VRAM.

The benchmark is synthetic and deterministic. It varies occlusion length,
camera-pose noise, depth noise, and an unobserved object movement. It is a
mechanism check, not evidence of closed-loop robot-task success.

`requirements-repro.txt` remains the larger historical reconstruction lock,
including Open3D and visualization packages. It is not needed for the VLA
milestone harness.
