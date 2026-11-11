# Zlab Cloud · Stage 3 robust walking

Fine-tune the Z01 on gentle heightfield waves with amplitudes of 0.8–2.4 cm.
Stage 3 varies foot friction from 0.3 to 1.1 and added joint damping from
0.05 to 0.30. It applies horizontal velocity pushes up to 0.18 m/s per axis
every 5–8 seconds. Training commands go up to 1.8 m/s.

Stage 3 keeps the robot facing world +X through the existing yaw command. During forward walking it penalizes heading error and world-frame lateral speed above 0.04 m/s, leaving room for small balance corrections. Evaluation reports world-frame sideways drift and heading error alongside the usual walking and stopping metrics.

The bundled Stage 2 checkpoint `model_10400.pt` supplies the initial policy
and optimizer state. A new run starts from it. Later runs in the same log
directory automatically resume the highest valid Stage 3 checkpoint.

## Kaggle

1. Import the [Stage 3 notebook](kaggle/robust_walking_3.ipynb). Keep it private and select **GPU T4 x2** and **Internet On**.
2. Add and enable a Kaggle Secret named `ZLAB_GITHUB_TOKEN` with read access to this repository.
3. Run the setup and smoke-test cells, then start training. Use **Save Version → Save & Run All** to retain the run outputs.

The notebook defaults to 1,500 additional PPO updates and 1,025 environments per
GPU (2,050 total). The smoke test checks both GPUs at that size. Checkpoints save
every 500 iterations and at the end, under
`/kaggle/working/zlab-stage3/rsl_rl/z01_robust_walking_3/`. To resume in a new
session, attach the previous notebook's output and set `RESUME_CHECKPOINT`
to its `.pt` file under `/kaggle/input`.

## Linux with one NVIDIA GPU

From the repository root, use Python 3.10–3.13 and a CUDA-enabled PyTorch build.
The launcher needs `flock` and `unzip`.

```bash
cd Zlab_Gym
python -m pip install -e . onnx
./train_stage3.sh 1500 64
```

The arguments are additional PPO updates, environments per GPU, and an optional
checkpoint path. Set `ZLAB_LOG_ROOT` to an absolute path on persistent storage
to save logs outside the checkout.

## Evaluate a checkpoint

Run 100 seeded trials with a stand, acceleration, cruise, braking, and stop
sequence on Stage 3 terrain. The evaluator reports upright and clean-stop
counts, speed tracking, stalls, and step measurements.

```bash
python scripts/play.py --checkpoint-file checkpoints/model_10400.pt \
  --eval-run-speed 1.55 --num-envs 100 --device cuda:0 --seed 2026
```

Use the same command with the new Stage 3 checkpoint to compare it against the
starting policy. Omit `--eval-run-speed` and use one environment for interactive
playback.
