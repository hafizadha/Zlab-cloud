# Zlab Cloud

Train Z01 locomotion policies in MuJoCo on a Linux cloud machine with an NVIDIA
GPU. The repo includes the gym, robot assets, SDK, and resume checkpoints for
walking, fast walking, stride walking, and Stages 1 and 2 of robust walking.

## Setup

On a GPU instance with a working NVIDIA driver and Python 3.10 or 3.11, run:

```bash
# Ubuntu or Debian, if unzip or flock is missing
sudo apt-get install -y unzip util-linux
nvidia-smi
cd Zlab_Gym
python3 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install onnx tensorboard
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python scripts/list_envs.py
```

Use a CUDA-enabled PyTorch build compatible with your cloud instance if the
last check reports CUDA unavailable. The MuJoCo and Warp versions are pinned
in [setup.py](Zlab_Gym/setup.py). Set `ZLAB_PYTHON` to a Python executable if
you want to run the launcher from a different environment.

## Kaggle

Import [the Stage 2 notebook](kaggle/robust_walking_2.ipynb) into Kaggle
and set **Accelerator** to **GPU T4 x2** and **Internet** to **On**. Keep the
notebook private and add a Kaggle Secret named `ZLAB_GITHUB_TOKEN` with read
access to this repository. The notebook installs dependencies and runs a
one-update smoke test on the first T4 before starting training from the
included Stage 2 checkpoint. Use **Save Version → Save & Run All** to keep the run outputs.
The notebook saves checkpoints every 500 updates and at the end. Checkpoints
and TensorBoard logs go to `/kaggle/working/zlab-training`.

## Stage 2 robust walking

After setup, run this from `Zlab_Gym`:

```bash
./train_walking.sh --robust 2 1500 64
```

This runs 1,500 additional PPO updates with 64 parallel environments. It resumes
the latest usable Stage 2 save in the log directory, or starts from the bundled
Stage 2 checkpoint `model_7300.pt` on a fresh instance. Stage 2 varies foot
friction and joint damping and applies horizontal pushes on flat ground.
Use the same command to continue, and increase the update count for a longer run.
Checkpoints save every 100 updates and at the end under
`logs/rsl_rl/z01_robust_walking_2/`. Set `ZLAB_LOG_ROOT` to an absolute path on
persistent storage to keep saves across instance restarts.

To replay Stage 2 training from the bundled Stage 1 checkpoint:

```bash
./train_walking.sh --robust 2 1500 64 checkpoints/robust-walking-1-model_6199.pt
```

## Other training modes

From `Zlab_Gym`, choose one phase:

| Phase | Resume command |
| --- | --- |
| Walking | `./train_walking.sh 500 64 checkpoints/walking-model_499.pt` |
| Fast walking | `./train_walking.sh --fast 500 64 checkpoints/fast-walking-model_1000.pt` |
| Stride walking | `./train_walking.sh --stride 500 64 checkpoints/stride-walking-model_1100.pt` |

To start balance recovery from the included Stage 2 checkpoint, run:

```bash
./train_walking.sh --recovery 1500 64 checkpoints/robust-walking-2-model_6698.pt
```

This trains on horizontal pushes and roll/pitch disturbances during walking
and standing. To continue from the latest recovery save, run
`./train_walking.sh --recovery`. Each run saves every 100 updates and at the end.
Pass a larger update count after `--recovery` for a longer run.

The first argument after the mode (including the stage number for `--robust`) is the number of additional PPO
updates. The next is the number of parallel environments. With a new log
directory, `./train_walking.sh` starts from the bundled Zoomlion walking
policy. After the first run, omit the checkpoint argument to resume the
furthest usable save for that phase. Checkpoints and TensorBoard logs are
written under `Zlab_Gym/logs/rsl_rl/`. Set `ZLAB_LOG_ROOT` to an absolute
path on persistent cloud storage if the instance disk is temporary.

See [policy phases](Zlab_Gym/POLICY_PHASES.md) for rewards, penalties, and
measured results, and [the gym README](Zlab_Gym/README.md) for evaluation.

## Upstream sources

| Directory | Upstream and base revision |
| --- | --- |
| `Zlab_Gym/` | [Zvalley-Robotics/Zlab_Gym](https://github.com/Zvalley-Robotics/Zlab_Gym), `48a01b6`, plus local training and playback changes |
| `zv_robot_assets/` | [Zvalley-Robotics/zv_robot_assets](https://github.com/Zvalley-Robotics/zv_robot_assets), `244246c` |
| `zv_robot_sdk/` | [Zvalley-Robotics/zv_robot_sdk](https://github.com/Zvalley-Robotics/zv_robot_sdk), `4ce4906` |
