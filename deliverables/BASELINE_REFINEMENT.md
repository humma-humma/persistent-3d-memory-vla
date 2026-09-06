# RGB-policy baseline refinement

## Outcome

The bounded refinement improved substantially over base SmolVLA, but did not
beat hold-state on the frozen six-episode, 900-action gate. Geometry-token
fusion is therefore not promoted yet.

| Policy | MAE | RMSE | Normalized RMSE | Gate |
|---|---:|---:|---:|---|
| Base SmolVLA | 15.5077 | 22.4846 | 0.9318 | fail |
| Original projection adapter | 9.7764 | 13.4545 | 0.6756 | fail |
| Broad projection adapter | 11.7938 | 16.4831 | 0.7601 | fail |
| Broad adapter + final expert layer | 9.6797 | 13.6423 | 0.6335 | fail |
| Expert policy + scalar hold residual | 8.6790 | 12.2185 | 0.5700 | fail |
| Expert policy + per-dimension residual | 8.2763 | 12.1506 | 0.5557 | fail |
| Hold-state | **7.9786** | **11.3306** | **0.5386** | control |

The strongest residual policy improves RMSE 46.0% relative to base SmolVLA,
but is 7.2% worse than hold-state. It beats hold-state on two of six episodes
by RMSE and one of six by normalized RMSE.

## Protocol

- Frozen evaluation episodes: `0, 11, 12, 13, 14, 15`.
- Evaluation anchors: `0, 50, 100`; horizon: 50 actions.
- Broad training episodes: `1-10` and `16-49`, with anchors
  `0, 50, 100, 150, 200, 250`.
- Sampling uses deterministic shuffled epochs instead of repeatedly favoring
  early episodes.
- The expert-layer run trains projections, the final action-expert layer, and
  expert norm; the VLM and earlier expert layers remain frozen.
- Residual coefficients are least-squares fits on episodes `1-10`, never on a
  frozen evaluation episode. The per-dimension coefficients are
  `[0, 0.7866, 0.6398, 0.7895, 0, 0.7667]` in
  `hold + alpha * (SmolVLA - hold)`.
- Results are open-loop recorded-action diagnostics, not closed-loop task
  success.

## Reproducible artifacts

- `outputs/smolvla_projection_broad44_2000/pilot_report.json`
- `outputs/smolvla_expert1_broad44_1000/pilot_report.json`
- `artifacts/smolvla_expert1_broad44_1000_gate.json`
- `artifacts/smolvla_expert1_residual_calibration.json`
- `artifacts/smolvla_expert1_residual_per_dim_calibration.json`
- `artifacts/smolvla_expert1_residual_per_dim_heldout6_eval.json`
- `artifacts/smolvla_expert1_residual_per_dim_gate.json`

The implementation adds leak checks for the fixed evaluation episodes,
balanced example ordering, bounded expert-layer selection, explicit residual
calibration, named-policy promotion scoring, and raw-policy reporting beside
the residual policy. The full repository verification is `125 passed`.
