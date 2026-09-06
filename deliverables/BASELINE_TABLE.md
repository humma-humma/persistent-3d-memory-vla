# Controlled synthetic baseline

Mean over 40 seeded trials for each of 12 occlusion/noise conditions. Lower is
better for the three error metrics; higher availability is better.

| Input/memory condition | Action error | Position error when available | Confidence Brier | Availability |
|---|---:|---:|---:|---:|
| Current RGB only | 0.6230 | 0.0715 | 0.0630 | 0.5747 |
| Short RGB history | 0.4269 | 0.0929 | 0.1013 | 0.7662 |
| Persistent geometry, no gating | 0.2054 | 0.2054 | 0.2522 | 1.0000 |
| Confidence-aware persistent geometry | **0.1748** | **0.1748** | **0.1217** | **1.0000** |

The current-view baseline has low error only when it is available. Persistent
memory improves action error during occlusion. Confidence-aware replacement
reduces the stale-map penalty after object movement by 14.9% relative to the
ungated persistent baseline and roughly halves its Brier score. These are
controlled simulation results, not real-VLA or closed-loop results.

Source: `artifacts/controlled_demo/baseline_summary.json`, generated with
`scene-memory-demo --seed 7 --trials 40`.
