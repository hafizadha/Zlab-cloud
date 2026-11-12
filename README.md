# Zlab Cloud · Straight BallSprint

Train the Z01 to sprint straight ahead, brake, and stop in front of a ball.
Each trial places the ball 5–11 m ahead. The speed command ramps from 2.6 to
4.4 m/s over the first 1,000 PPO updates. The controller corrects heading
toward the lane center and reduces the speed command as the robot approaches.

A full pass holds a measured forward speed of at least 4 m/s for 0.2 seconds,
then stays upright within 1 m of the ball at under 0.25 m/s for 0.4 seconds.
Overshooting or displacing the ball by more than 0.15 m ends the trial.

Training applies short, randomly timed pushes to the torso. Force components
range from −45 to +45 N and each push lasts 0.08–0.16 seconds. Foot friction
is resampled between 0.35 and 1.15 at each reset. Training also varies the
initial base pose and velocity, joint damping, motor gains, and body mass.

## Train

Use Python 3.10–3.13, CUDA-enabled PyTorch, `flock`, and `unzip`. From the repo root:

```bash
cd Zlab_Gym
python -m pip install -e . onnx
./train_ball_sprint.sh 2500 64
```

For a Kaggle session with two T4s:

```bash
ZLAB_USE_ALL_GPUS=1 ZLAB_LOG_ROOT=/kaggle/working/zlab-straight-ball-sprint \
  ./train_ball_sprint.sh 2500 1024
```

The environment count is per GPU. With no saved checkpoint, the launcher
starts from the bundled BallSprint actor. It resumes the newest valid
checkpoint for this task in the selected log directory. Pass a checkpoint
from this task as the third argument to select a specific save. Checkpoints
save every 100 updates and at the end under
`$ZLAB_LOG_ROOT/rsl_rl/z01_straight_ball_sprint/` (`logs` by default).

## Save checkpoints to Hugging Face

From `Zlab_Gym`, install the optional uploader and set the destination model
repository. Use a Hugging Face token with write access to that repo.

```bash
python -m pip install -e '.[huggingface]' onnx
export ZLAB_HF_REPO_ID=your-username/zlab-checkpoints
./train_ball_sprint.sh 2500 64
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
  --eval-episodes 48 --num-envs 48 --device cuda:0 --seed 2026
```

The evaluator reports the combined sprint-and-stop pass rate, clean stops,
falls, misses, timeouts, and mean peak forward speed. Add `--disturbances True`
to evaluate with torso pushes and randomized starting poses.
