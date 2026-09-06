# SmolVLA projection-adapter pilot

A low-memory adapter freezes the SmolVLA VLM and action expert and trains only
1,635,152 state/action projection and time-conditioning parameters. Training
uses 50 anchors across SO-100 pick-place episodes 1–10; episode 0 is reserved
for evaluation. Images are resized to 256 px. The pilot runs 500 deterministic
steps with batch size one on the RTX 2080.

Training took 106.3 seconds and used 1,277 MiB peak PyTorch VRAM. The minimum
observed batch loss was 0.0148. Held-out evaluation uses three episode-0 anchors
and 50 target actions per anchor.

| Method | MAE | RMSE | Normalized RMSE |
|---|---:|---:|---:|
| Base SmolVLA, 256 px | 10.8110 | 15.8030 | 0.7107 |
| Projection adapter, 500 steps | 5.8263 | 9.1560 | **0.4914** |
| Hold current motor state | **5.1759** | **8.2502** | 0.4924 |

The adapter reduces normalized RMSE 30.9% relative to the matched base model
and narrowly beats hold-state by 0.2% on that scale. It does not beat hold-state
in raw MAE or RMSE. This is a small open-loop pilot on one held-out episode, not
evidence of closed-loop task success.

The expanded six-episode promotion gate is now complete. Across 900 target
actions, the adapter improves aggregate RMSE 40.2% and normalized RMSE 27.5%
versus base, and beats base on all six episodes. It remains worse than
hold-state by 18.7% RMSE and 25.4% normalized RMSE, so the checkpoint is not
promoted. See `MULTI_EPISODE_PROMOTION_GATE.md`.

Machine-readable sources:

- `outputs/smolvla_projection_pilot_500_ep1_10/pilot_report.json`
- `artifacts/smolvla_base_256_eval.json`
- `artifacts/smolvla_projection_pilot_500_ep1_10_eval.json`
- `artifacts/smolvla_promotion_gate.json`
