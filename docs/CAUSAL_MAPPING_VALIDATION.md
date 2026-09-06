# Causal acquisition-order mapping and validation

`persistent_scene_memory.causal_mapping` is the causal reference path for
sequential experiments. For acquisition frame `t`, it reconstructs only the
image prefix `0..t`; no image or correspondence with a future endpoint is
available. Every prefix is anchored to the same initial pair, and observation
associations carry stable numeric track IDs across reconstructions.

This deliberately favors scientific clarity over speed: it recomputes each
prefix instead of maintaining an optimized live mapper.

## Run

```powershell
$env:PYTHONPATH = "src"
& $python -m persistent_scene_memory.causal_mapping `
  --dataset "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box" `
  --max-images 8 `
  --output-dir "outputs\stage1_box_causal_memory_8" `
  --bundle-adjustment-max-nfev 20
```

Outputs:

- `causal_snapshots/frame_*/`: prefix cameras, full map, current observations,
  and reconstruction summary;
- `causal_memory/`: persistent memory and fixed-size adapter frames;
- `causal_validation.json`: causal contract, per-frame pose/map statistics,
  stable-ID diagnostics, point drift, confidence calibration, error by token
  age, and aggregate metrics.

## Verified eight-image result

- final registration: 8/8 cameras;
- final map: 741 points and 741 stable IDs;
- identity conflicts: zero merges and zero splits;
- final pose error: `0.5322 deg` mean rotation and `0.04696` mean translation;
- final retained memory: 711 tokens, comprising 320 current and 391 persisted;
- maximum token age: six acquisition events;
- final mean nearest-GT-point proxy error: `0.0610`;
- final confidence Brier score: `0.1664`;
- all-frame aggregate nearest-GT proxy error: mean `0.0760`, median `0.0575`;
- all-frame fraction within the `0.15` correctness threshold: `0.9076`;
- all-frame confidence Brier score: `0.1816`.

Image 2 could not be posed when first acquired. Its frame therefore produced
zero current world-frame updates and aged all 67 existing tokens by one. This
is intentional: a 2D association cannot update world memory without a
registered camera pose.

The calibration bins show that geometric confidence is generally conservative.
For example, tokens in `[0.4, 0.6)` have mean confidence `0.533` but empirical
correctness `0.900`. The formula needs calibration before being interpreted as
a probability.

## Limitations

- Prefix reconstruction is causal but not computationally incremental.
- Best-PnP may register an earlier failed camera during a later prefix. The
  implementation does not backfill that old image as newly current.
- Stable identities are point-track identities, not object identities.
- Nearest-ground-truth-point distance is a proxy, not known correspondence.
- The first two-view initialization has worse point accuracy than later maps.

The next policy-facing step is to freeze this causal sequence and compare
per-frame, ungated-persistent, and confidence-aware geometry under one action
protocol, after the RGB task policy passes hold-state.
