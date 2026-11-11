# Zlab Cloud · Ball sprint

Fine-tune the Z01 to turn toward a randomly placed ball, sprint, and brake
to a stable stop nearby. The ball starts 4–11 m away and can appear in any
direction. Simulated ball position drives the policy's existing forward-speed
and yaw commands. Training commands range up to 3.0 m/s, with lower speeds
during turns and near the ball.

The rewards favor closing distance, staying upright, alternating footfalls,
and brief airborne phases while moving forward. An arrival requires the
robot to be 0.45–0.90 m from the ball and moving below 0.25 m/s. The bundled
`model_15000.pt` is the starting locomotion checkpoint.

## Train on Linux

Use Python 3.10–3.13, CUDA-enabled PyTorch, `flock`, and `unzip`. From the repo root, run

```bash
cd Zlab_Gym
python -m pip install -e . onnx
./train_ball_sprint.sh 4500 64
```

The third argument optionally selects an explicit checkpoint. Otherwise the
launcher uses the latest valid ball-sprint save in the log directory, or
`model_15000.pt` for a fresh run. Set `ZLAB_LOG_ROOT` to an absolute path to
use persistent storage.

For Kaggle with two T4s, run the same script from `Zlab_Gym` after setup.

```bash
ZLAB_USE_ALL_GPUS=1 ZLAB_LOG_ROOT=/kaggle/working/zlab-ball-sprint \
  ./train_ball_sprint.sh 4500 1025
```

The environment count is per GPU. Checkpoints save every 500 updates and
at the end under the selected log directory.

## Evaluate

```bash
python scripts/play.py --checkpoint-file /path/to/model.pt \
  --eval-episodes 100 --num-envs 16 --device cuda:0 --seed 2026
```

The evaluator counts clean arrivals, falls, and successful approaches with
at least three qualifying flight samples and a peak forward speed of at
least 2.0 m/s. It also reports mean peak speed. Omit `--eval-episodes` for
interactive playback.
