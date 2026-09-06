# SmolVLA multi-episode promotion gate

The 500-step projection adapter and matched base checkpoint were evaluated on
held-out SO-100 episodes 0, 11, 12, 13, 14, and 15. Each episode uses anchors
0, 50, and 100 with a 50-action horizon, for 900 target actions in total.

The predeclared gate requires the adapted checkpoint to beat both base SmolVLA
and hold-state in aggregate raw RMSE and normalized RMSE.

| Method | MAE | RMSE | Normalized RMSE |
|---|---:|---:|---:|
| Base SmolVLA | 15.5077 | 22.4846 | 0.9318 |
| Projection adapter | 9.7764 | 13.4545 | 0.6756 |
| Hold current motor state | **7.9786** | **11.3306** | **0.5386** |

The adapter improves MAE 37.0%, RMSE 40.2%, and normalized RMSE 27.5% versus
base. It beats base on all six episodes for all three metrics. Against
hold-state it wins 0/6 episodes on MAE, 1/6 on RMSE, and 1/6 on normalized
RMSE. Aggregate adapter RMSE is 18.7% worse than hold-state and normalized RMSE
is 25.4% worse.

**Decision: promotion failed.** The projection-only adaptation demonstrates
real learning relative to the base checkpoint, but it is not yet a competitive
action predictor under this open-loop control. It is not evidence of
closed-loop task success or of a benefit from the persistent 3D memory.

Machine-readable sources:

- `artifacts/smolvla_base_heldout6_eval.json`
- `artifacts/smolvla_adapted_heldout6_eval.json`
- `artifacts/smolvla_promotion_gate.json`
