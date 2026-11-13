# Zlab Cloud · Straight BallSprint

Train the Z01 to sprint straight ahead, brake, and stop in front of a ball.
Each trial places the ball 5–11 m ahead. The controller corrects heading
toward the lane center and reduces the speed command as the robot approaches.

A full pass holds a measured forward speed of at least 4 m/s for 0.2 seconds,
then stays upright within 1 m of the ball at under 0.25 m/s for 0.4 seconds.
Overshooting or displacing the ball by more than 0.15 m ends the trial.

Torso pushes can be added during training. At full strength, force components
range from −45 to +45 N and each push lasts 0.08–0.16 seconds. Foot friction
is resampled between 0.35 and 1.15 at each reset. Training also varies the
initial base pose and velocity, joint damping, motor gains, and body mass.

## Train

Open [zlab-training.ipynb](zlab-training.ipynb) in Kaggle. The notebook clones
`main` into a fresh directory, prints the commit, and calls
`train_straight_ball_sprint.sh`. The launcher prints the active task ID,
ball range, command cap, and torso push strength before training. Keep the
run outputs under `/kaggle/working` and the clone outside that directory.

On Linux, use Python 3.10–3.13, CUDA-enabled PyTorch, `flock`, and `unzip`:

```bash
cd Zlab_Gym
python -m pip install -e . onnx
./train_straight_ball_sprint.sh 500 64
```

The environment count is per GPU. Use `ZLAB_USE_ALL_GPUS=1` for all visible GPUs.
The launcher starts from the bundled actor unless you pass a checkpoint from
this task as the third argument. Checkpoints save every 100 updates and at the
end under `$ZLAB_LOG_ROOT/rsl_rl/z01_straight_ball_sprint/` (`logs` by default).

### Training blocks

1. Start with 500 updates at a 2.8 m/s cap, with foot friction and motor
   randomization and zero torso pushes. Evaluate saves on 48 trials at the same
   cap. Use clean stops as the primary measure. Aim for at least 38/48 clean
   stops and at most 4/48 falls before increasing speed.
2. Resume the strongest save in 300–500-update blocks at command caps of 3.2,
   3.6, 4.0, and 4.4 m/s. Use `ZLAB_COMMAND_SPEED` to set each cap. Evaluate
   after each block and advance only when clean stops and falls meet the same
   gate. At 4.4 m/s, rank checkpoints by the combined 4 m/s and clean-stop
   criterion. Aim for at least 38/48 combined passes and at most 4/48 falls
   before adding pushes. Inspect failures on the shortest placements before
   spending more updates at that cap.
3. Once the policy can sprint and brake reliably, add torso pushes with
   `ZLAB_TORSO_PUSH_SCALE=0.25`, then 0.5 and 1.0. Compare performance with
   and without pushes. Use the combined pass rate and fall count on a fresh
   evaluation seed to choose the final checkpoint.

## Save checkpoints to Hugging Face

From `Zlab_Gym`, install the optional uploader and set the destination model
repository. Use a Hugging Face token with write access to that repo.

```bash
python -m pip install -e '.[huggingface]' onnx
export ZLAB_HF_REPO_ID=your-username/zlab-checkpoints
./train_straight_ball_sprint.sh 500 64
```

On Kaggle, add `HF_TOKEN` under **Add-ons → Secrets**, enable it for the notebook,
and load it before launching training. Keep Internet access enabled.

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["ZLAB_HF_REPO_ID"] = "your-username/zlab-checkpoints"
```

On other hosts, inject `HF_TOKEN` from the host's secret store or use a cached
Hugging Face login. Each scheduled checkpoint and the final save are kept on
local disk and mirrored to `checkpoints/<run-name>/model_<iteration>.pt` in
the model repo. The uploader creates a private repo if the configured repo is
missing and the token permits creation. Check for `[HF] Uploaded` in the
training log. An upload failure leaves the local checkpoint available for
manual upload. Omit `ZLAB_HF_REPO_ID` for local-only training.

## Evaluate

```bash
python scripts/play.py --checkpoint-file /path/to/model.pt \
  --eval-episodes 48 --num-envs 48 --device cuda:0 --seed 2026 --command-speed 2.8
```

The evaluator reports the combined sprint-and-stop pass rate, clean stops,
falls, misses, timeouts, and mean peak forward speed. Match the command cap
to the training block. Use `--disturbances True --torso-push-scale 0.25` to
evaluate with quarter-strength torso pushes and randomized starting poses.
