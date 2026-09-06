# SmolVLA real-data baseline

Dataset: `lerobot/svla_so100_pickplace`, episode 0. The evaluation uses three
anchors (`0`, `50`, `100`) and a 50-action horizon at each anchor. Inputs are
the recorded top and wrist RGB frames, six motor states, and the instruction
“Pick up the cube and place it in the box.” Predictions are unnormalized with
the dataset's own statistics.

| Method | MAE | RMSE | Normalized RMSE |
|---|---:|---:|---:|
| Zero-shot SmolVLA base | 11.7704 | 15.4159 | 0.7736 |
| Hold current motor state | **5.1759** | **8.2502** | **0.4924** |

The zero-shot model is worse than motor-state persistence. This establishes the
pre-fine-tuning control and shows that the base checkpoint must be adapted to
the task before testing whether geometric memory improves it. These metrics
cover 150 recorded target actions from one episode and are not closed-loop task
success.

Machine-readable source: `artifacts/smolvla_real_data_baseline.json`.
