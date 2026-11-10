# Zlab_Gym — A Motion Imitation & Basic Locomotion Training Framework for the Z01 Robot, Enabling Users to Train Custom Motion Models by Adapting the Framework

A minimal secondary-development framework built on
[mjlab](https://github.com/mujocolab/mjlab) (`mjlab==1.2.0`) + MuJoCo.
It registers **one robot** (`Z01_std_add_head`, 29 DOF) with motion-tracking,
velocity, and single-robot soccer tasks. Tracking commands 27 of its joints.

| Task id | Group | Description |
|---|---|---|
| `Z01-StdAddHead-Tracking` | `tracking` | whole-body motion imitation (flat terrain) |
| `Z01-StdAddHead-Velocity-Rough` | `velocity` | blind twist-velocity locomotion, rough terrain + terrain curriculum |
| `Z01-StdAddHead-Velocity-Flat` | `velocity` | blind twist-velocity locomotion, flat plane |
| `Z01-StdAddHead-Velocity-Walk` | `velocity` | fine-tune the bundled walking policy on flat ground |
| `Z01-StdAddHead-Velocity-FastWalk` | `velocity` | continue walking training at speeds up to 1.6 m/s |
| `Z01-StdAddHead-Velocity-Run` | `velocity` | straight-line running practice with alternating foot contacts and arm swing |
| `Z01-StdAddHead-Soccer-Kick` | `soccer` | shoot a nearby ball into a goal |
| `Z01-StdAddHead-Soccer-Approach` | `soccer` | find a ball anywhere on the training pitch and shoot into a full-size goal |
| `Z01-StdAddHead-Soccer-Sprint` | `soccer` | accelerate on straight approaches and slow down for the kick setup |

---

## 1. Installation

```bash
conda create -n zlab_tracking python=3.11 -y
conda activate zlab_tracking

# Install torch first, matching your CUDA, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

4rr       # installs mjlab==1.2.0 automatically
```

`src/` is prepended to `sys.path` by `scripts/train.py` / `scripts/play.py`
so the **vendored, patched `rsl_rl`** under `src/rsl_rl` takes precedence over
any site-packages copy. Do not remove those `sys.path` lines.

## 2. List available tasks

```bash
python scripts/list_envs.py            # all registered tasks
python scripts/list_envs.py --keyword Z01   # filter
```

## 3. Quick start

### 3.1 Tracking (motion imitation) — `--motion-file` is mandatory

```bash
# Train
python scripts/train.py Z01-StdAddHead-Tracking \
  --motion-file=src/assets/motions/z01_std_add_head/demo_act/z01_std_base_heart.npz \
  --env.scene.num-envs=256 \
  --agent.experiment-name=z01_std_add_head_tracking
# logs -> logs/rsl_rl/z01_std_add_head_tracking/<timestamp>/  (model_<iter>.pt every 500 iters)

# Play
python scripts/play.py Z01-StdAddHead-Tracking \
  --checkpoint-file=logs/rsl_rl/z01_std_add_head_tracking/<RUN_DIR>/model_0.pt \
  --motion-file=src/assets/motions/z01_std_add_head/demo_act/z01_std_base_heart.npz \
  --num-envs 1 --device cpu --viewer native
```

### 3.2 Velocity tasks — no motion file needed

```bash
# Train (rough terrain, terrain-level curriculum)
python scripts/train.py Z01-StdAddHead-Velocity-Rough \
  --env.scene.num-envs=4096 \
  --agent.experiment-name=z01_std_add_head_velocity

# Train (flat plane)
python scripts/train.py Z01-StdAddHead-Velocity-Flat \
  --env.scene.num-envs=4096 \
  --agent.experiment-name=z01_std_add_head_velocity

# Play
python scripts/play.py Z01-StdAddHead-Velocity-Rough \
  --checkpoint-file=logs/rsl_rl/z01_std_add_head_velocity/<RUN_DIR>/model_0.pt \
  --num-envs 1 --device cpu --viewer native
```

Run `./train_walking.sh` to fine-tune the bundled Zoomlion walking policy
with forward speed commands from 0.2 to 0.8 m/s and standing intervals. The
default block is 500 PPO updates with 64 environments. Checkpoints save
every 100 iterations under `logs/rsl_rl/z01_walking/`. Rerun the launcher
to resume from the furthest usable walking checkpoint. For playback, pass
`Z01-StdAddHead-Velocity-Walk` and that checkpoint to `scripts/play.py`.

Run `./train_walking.sh --fast` for the faster stage. It starts from a
walking save and continues from its own saves on later runs. Fast-stage
checkpoints are stored under `logs/rsl_rl/z01_fast_walking/`. Use
`Z01-StdAddHead-Velocity-FastWalk` for playback and `--eval-run-speed=1.5`
for a headless stand, accelerate, cruise, and stop comparison.

Run `./train_walking.sh --stride` to continue a fast-walking policy with
rewards for maintaining forward motion and planting longer, quicker steps.
The stall penalty applies while a forward command is active. The stride
task plays at a fixed 1.35 m/s command and saves in
`logs/rsl_rl/z01_stride_walking/`. The evaluation reports time below half
the commanded speed, forward foot separation at landing, and supported
steps per second alongside completion and stopping results.

Run `./train_walking.sh --long-stride` to continue from a stride-walking
checkpoint with forward commands up to 1.8 m/s, a longer foot placement
target, and a 0.19 s swing target. Pass a checkpoint as the third argument
to start from a specific save. The new checkpoints live under
`logs/rsl_rl/z01_long_stride_walking/`. Use
`Z01-StdAddHead-Velocity-LongStrideWalk` for playback at 1.55 m/s.

Use the robustness stages after establishing a stable fast gait. Each stage
keeps the stride rewards and resumes the preceding stage on its first run.

| Command | Disturbances |
| --- | --- |
| `./train_walking.sh --robust 1` | Flat ground with foot friction 0.5–0.9 and added joint damping 0.05–0.15 |
| `./train_walking.sh --robust 2` | Wider friction and damping ranges, plus horizontal velocity pushes up to 0.12 m/s per axis every 5–8 seconds |
| `./train_walking.sh --robust 3` | Gentle heightfield waves with 0.8–2.4 cm amplitude and pushes up to 0.18 m/s per axis |

Pass the update count, number of environments, and optionally a checkpoint
after the stage number. For example, `./train_walking.sh --robust 1 100 64`
runs a short first block. Evaluate each save on its own stage and on the flat
task before advancing. Use task IDs `Z01-StdAddHead-Velocity-RobustWalk1`
through `Z01-StdAddHead-Velocity-RobustWalk3` with `--eval-run-speed=1.55`.
Keep motor delays enabled when comparing robustness.

Run `./train_walking.sh --recovery` to fine-tune the Stage 2 policy with
more standing practice and small roll and pitch perturbations on flat
ground. The default block is 1,500 updates with 64 environments. The
policy receives a penalty for backward velocity above 0.15 m/s while
commanded forward. Checkpoints save under `logs/rsl_rl/z01_balance_recovery/`.
Evaluate both the Stage 2 and recovery checkpoints with
`Z01-StdAddHead-Velocity-BalanceRecovery` and `--eval-run-speed=0` for
standing recovery or `--eval-run-speed=1.55` for walking recovery.

For straight-line running practice, run `./train_human_run.sh`. It resumes
the furthest saved running checkpoint, or initializes from the bundled walking
policy when no running checkpoints exist. It trains for 500 additional updates with 64
environments, and saves under `logs/rsl_rl/z01_human_run_balanced/`. The
running actor keeps its imported walking observation normalization fixed.
This experiment directory separates the corrected runs from earlier ones.
To continue from
a save, pass the checkpoint as the third argument, for example
`./train_human_run.sh 1000 64 /absolute/path/to/model_499.pt`.
For playback, pass `Z01-StdAddHead-Velocity-Run` and that checkpoint to
`scripts/play.py`.

The first running stage uses forward commands up to 1.35 m/s and starts from
near the nominal standing pose. Compare the walking baseline and each
checkpoint at the same speed and seed before increasing the training range.
The evaluation holds a stand, ramps up to cruise, then brakes to a stop.

```bash
python scripts/play.py Z01-StdAddHead-Velocity-Run \
  --agent=walking --eval-run-speed=0.5 --num-envs=16 --seed=2026
python scripts/play.py Z01-StdAddHead-Velocity-Run \
  --checkpoint-file=/absolute/path/to/model.pt \
  --eval-run-speed=0.5 --num-envs=16 --seed=2026
```

Repeat at 1.3 m/s. Compare how many worlds finish upright and stop, forward
speed error, and sideways drift. The airborne fraction is a gait diagnostic.
Watch the policy to check that its feet alternate through successive strides.

### 3.3 Soccer

For this workspace, the prepared `../.venv-soccer` environment and field
checkpoint are picked up automatically by the launcher.

```bash
./run_soccer.sh        # Watch the latest field checkpoint
./run_soccer.sh train  # Continue field training
./run_soccer.sh eval   # Report goals and falls
```

For a multi-hour run, use `./train_soccer.sh`. It resumes the latest field
checkpoint for 6,000 additional PPO updates, saves every 50 iterations, and
streams a console log to `logs/soccer-console/`. Pass an iteration count to
change the length, for example `./train_soccer.sh 10000`. Press Ctrl+C to stop.
Running the script again resumes from the latest saved checkpoint.

The soccer policy uses ball state from MuJoCo. Start with `Soccer-Kick`, then
continue from its checkpoint on `Soccer-Approach`. Both tasks use the same
observation and action layout. The approach task uses a 6 × 5.5 m pitch and a
2.6 × 1.2 m opposing goal. Each reset samples a new ball position ahead, beside,
or behind the robot, with clearance around its feet. The approach command
steers around the ball when the robot starts between it and the goal.

```bash
# Short GPU run with periodic checkpoints.
python scripts/train.py Z01-StdAddHead-Soccer-Kick \
  --warm-start-walking True --env.scene.num-envs=128 \
  --agent.max-iterations=64 --agent.save-interval=16 --agent.run-name=kick_test

# Longer GPU run. Adjust the number of worlds to fit your GPU.
python scripts/train.py Z01-StdAddHead-Soccer-Kick \
  --warm-start-walking True --env.scene.num-envs=1024 --agent.max-iterations=2000

# Visualize the unmodified walking actor approaching the ball.
python scripts/play.py Z01-StdAddHead-Soccer-Approach --agent walking --num-envs=1

# Evaluate a trained checkpoint headlessly, then omit --eval-episodes to watch.
python scripts/play.py Z01-StdAddHead-Soccer-Kick \
  --checkpoint-file=/absolute/path/to/model_1999.pt --device=cpu \
  --num-envs=4 --eval-episodes=20 --seed=42

# Continue from a kick checkpoint with randomized ball positions and the larger goal.
python scripts/train.py Z01-StdAddHead-Soccer-Approach \
  --agent.resume True --agent.load-checkpoint=/absolute/path/to/model_1999.pt \
  --env.scene.num-envs=1024 --agent.max-iterations=2000
```

Runs are stored in `logs/rsl_rl/z01_soccer_kick/` and
`logs/rsl_rl/z01_soccer_approach/`. The first run may spend several minutes
compiling Warp kernels. Use goal and fall rates from held-out evaluation seeds
to judge policy quality. To train from scratch, omit `--warm-start-walking True`.
For rolling-ball starts, add
`--env.events.reset-ball.params.speed-range='(-0.2, 0.2)'` to sample initial
planar velocity in m/s.

### 3.4 Sprint tracking and soccer

Use the [LaFAN1 BVH clips](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
with [GMR](https://github.com/YanjieZe/GMR) installed in the Python environment. The
converter adapts GMR's LaFAN1 mapping to the Z01 and writes a 50 Hz reference.
The example segments cover a fast stride and a run to a stop. They play below
human speed for initial Z01 simulation trials.

```bash
mkdir -p logs/motion-references
python tools/retarget_lafan1_to_z01.py /path/to/sprint1_subject2.bvh \
  logs/motion-references/z01_lafan1_sprint.npz --start 3580 --end 3626 --speed 0.45
python tools/retarget_lafan1_to_z01.py /path/to/run1_subject2.bvh \
  logs/motion-references/z01_lafan1_run_to_stop.npz --start 4187 --end 4260 --speed 0.55

python scripts/train.py Z01-StdAddHead-Tracking-Residual \
  --motion-file=logs/motion-references/z01_lafan1_sprint.npz \
  --env.scene.num-envs=64 --agent.max-iterations=60 --agent.save-interval=30
```

The tracker uses one clip per run. Tracking-Residual adds scaled policy actions
to the reference joint positions and samples reference phases during training.
Use the same task and motion file for playback. Its motion ONNX marks the
reference offset with `action_reference=motion_joint_pos`.

Run `./train_lafan_sprint.sh` to train on the corrected standing-start,
running, and braking reference. It defaults to 11,000 PPO updates with 64
environments and saves every 500 iterations under
`logs/rsl_rl/z01_lafan1_stand_run_stop/`. The first run starts fresh.
Rerunning the command resumes from the latest usable checkpoint for this
reference. Use `./train_lafan_sprint.sh 500 64` for a shorter run. The
console log under `logs/lafan1-console/` shows the checkpoint and new run
directory. For playback, use `Z01-StdAddHead-Tracking-Residual` with
`logs/motion-references/z01_lafan1_stand_run_stop_clearance.npz`.

To continue training the original LaFAN1 sprint clip, run
`./train_lafan_sprint.sh --sprint` for 1,000 additional PPO updates with
64 environments.

Soccer-Sprint can resume a Soccer-Approach
checkpoint because the actor observations and actions have the same layout.
It commands up to 1.8 m/s when aligned with a distant waypoint and reduces the
target on approach. Train and evaluate it separately from the clip tracker.

```bash
python scripts/train.py Z01-StdAddHead-Soccer-Sprint \
  --agent.resume True --agent.load-checkpoint=/absolute/path/to/approach/model.pt \
  --env.scene.num-envs=64 --agent.max-iterations=150 --agent.save-interval=50

python scripts/play.py Z01-StdAddHead-Soccer-Sprint \
  --checkpoint-file=/absolute/path/to/sprint/model.pt \
  --num-envs=32 --eval-episodes=32 --seed=2026
```

Compare goals, falls, closing speed, setup completion, and time to first reach
the setup with the same evaluation settings on Soccer-Approach.

## 4. Common flags

| Flag | Applies to | Description |
|---|---|---|
| `--motion-file` | train / play (tracking) | motion `.npz`, **required** for tracking tasks |
| `--checkpoint-file` | play | local `.pt` checkpoint, required for `--agent trained` |
| `--agent.experiment-name` | train | log sub-directory name |
| `--agent.logger` | train | default `tensorboard`; `wandb` also available |
| `--agent.max-iterations` | train | override training length |
| `--env.scene.num-envs` | train | number of parallel envs |
| `--num-envs` | play | number of parallel envs |
| `--device` | play | `cpu` / `cuda:0`, defaults to `cuda:0` if available |
| `--gpu-ids` | train | `[0]`, `all`, or explicit ids (multi-GPU via torchrunx) |
| `--video` / `--video-length` | train / play | record video into `<log_dir>/videos/` |
| `--viewer` | play | `auto` / `native` / `viser`; `auto` falls back to `viser` without `DISPLAY` |
| `--agent` | play | `trained` loads a checkpoint, `zero` and `random` debug the task, `walking` previews the soccer approach |
| `--no-terminations` | play | disable all terminations (watch the full motion) |
| `--log-root` / `--log-dir` / env `ZLAB_LOG_ROOT` | train | log root (default `logs`) |
| `--enable-nan-guard` | train | enable NaN guard |

## 5. Bundled assets

```
src/assets/motions/z01_std_add_head/demo_act/
├── z01_std_base_heart.npz        # demo motion ("heart" gesture), input for --motion-file
├── z01_std_base_heart.onnx       # pretrained tracking policy (motion baked in)
└── z01_velocity_flat_base.onnx   # pretrained flat-velocity policy
```

The two ONNX files are exported artifacts of the corresponding trained runs,
provided as references for downstream deployment. This repo does **not** ship a
script that plays ONNX directly — run the `.pt` checkpoints with `scripts/play.py`.

## 6. ONNX export

Every `save()` of the tracking runner `MotionTrackingOnPolicyRunner` writes two
models next to the checkpoint:

- `<ckpt_stem>_policy.onnx` — the policy network (`actor.as_onnx()`).
- `<ckpt_stem>_motion.onnx` — `_OnnxMotionModel`: policy + reference motion
  buffers baked in as ONNX constants, so the deployed policy replays the clip
  without external data files.

Both carry mjlab metadata (`attach_metadata_to_onnx`). Velocity tasks use
mjlab's `VelocityOnPolicyRunner` (policy-only export).

## 7. Repo layout

```
scripts/                                    # train.py / play.py / list_envs.py
tools/                                      # offline motion conversion utilities
src/
├── assets/
│   ├── robots/Z01_std_add_head/            # constants + xmls/(loaded) + xml/(generator output)
│   │                                       #   + urdf/ csv/ meshes/ urdf_to_xml.yaml
│   └── motions/z01_std_add_head/demo_act/  # demo motion + ONNX artifacts
├── rsl_rl/                                 # vendored patched PPO (keep as-is!)
└── tasks/
    ├── __init__.py                         # registers "tracking" + "velocity" groups
    ├── tracking/
    │   ├── tracking_env_cfg.py             # make_tracking_env_cfg(): shared env skeleton
    │   ├── config/z01_std_add_head/        # env/rl cfg + task registration
    │   ├── rl/runner.py                    # MotionTrackingOnPolicyRunner (ONNX export)
    │   └── mdp/                            # commands / observations / rewards / terminations
    └── velocity/
        ├── config/z01_std_add_head/        # env/rl cfg + task registration
        └── mdp/rewards.py                  # repo-specific reward terms
```

`tools/`:

- `convert_t4_motion_to_z01.py` — T4 (27-DOF) → Z01 (29-DOF) motion conversion, the
  main tool: `python tools/convert_t4_motion_to_z01.py <motion.npz>`
  (`--dry-run` to preview, a directory argument to batch).
- `convert_pkl_to_50hz_npz.py` — pkl → 50 Hz npz motion.
- `convert_npz_to_csv.py` / `csv_to_npz.py` — **legacy** G1-specific scripts with
  hard-coded paths, not used by the Z01 workflow.

## 8. Extension points

1. **New motion**: drop a `.npz` with the same schema under `src/assets/motions/...`
   and pass it via `--motion-file`.
2. **New reward / termination**: implement in `src/tasks/<group>/mdp/` and mount it
   in the group's `env_cfgs.py`.
3. **New robot**: clone `Z01_std_add_head` constants + `xmls/` + `meshes/`, exposing
   `get_<robot>_robot_cfg()` and `<ROBOT>_ACTION_SCALE`.
4. **New task id**: duplicate a `register_mjlab_task` block and add the module path
   to `_TASK_GROUPS` in `src/tasks/__init__.py`.

## 9. Notes & caveats

- `src/rsl_rl` is a **patched** copy of rsl-rl (entropy annealing / std clamp).
  Do not delete it, and do not replace it with the site-packages version.
- The policy controls **27 joints**: `head_pitch` / `head_yaw` are excluded from
  actions and from the `joint_pos` / `joint_vel` observations, and are locked at
  zero by passive stiffness (`jnt_stiffness=100`) + damping (`dof_damping=10`)
  in `env_cfgs.py`; `HeadAgnosticMotionCommand` also strips them from the command
  observation. The `pose` std dicts, however, still cover all 29 joints.
- `Z01_STD_ADD_HEAD_ACTION_SCALE` is **hard-coded to 0.25** for every joint
  (not derived from `effort_limit / stiffness`). Override it in
  `z01_std_add_head_constants.py` if needed.
- Z01 joint names carry **no `_joint` suffix** (`left_knee`, `waist_yaw`,
  `head_pitch`, …), so `target_names_expr` / `joint_names` regexes must follow that.
- The velocity **Flat** variant differs from Rough: it drops
  `foot_clearance` / `foot_swing_height` / `soft_landing`, sets `air_time` weight to 0,
  and adds `base_height` (0.5), `foot_gait` (0.8), `feet_lateral_distance` (-0.5),
  `arm_symmetry` (0.5), `leg_symmetry` (0.3). It also applies a yaml run profile
  (`z01_velocity_flat_base`): a fixed default joint pose plus a full set of PD gains
  (`_YAML_PD_GROUPS`, standing height 0.823 m).
- The Z01 body directly below the torso is `waist_yaw_link` (no `pelvis`); velocity
  tasks use the root body `torso_link` for `terrain_scan`, `upright`, `body_ang_vel`
  and the `self_collision` sensor.
- The Z01 MJCF ships two sole-centre sites, `left_foot` / `right_foot`, required by
  the foot-based reward terms.
- `neg_power` (negative mechanical power / regenerative braking) is mounted at
  `cfg.rewards["neg_power"]` with `weight=0` — off by default.
- Trained playback loads a local `--checkpoint-file`.
