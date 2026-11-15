# Zlab Cloud · Ball approach stability

Train the Z01 ball approach at a fixed 2.8 m/s command cap. Full-length
trials place the ball 5–11 m ahead. A clean stop requires the robot to settle within 1 m
of the ball for 1.5 continuous seconds, below 0.25 m/s, within 0.25 m laterally, and with yaw
within 12° of the lane. Robot–ball contact ends the approach and costs 25
reward units. Other failed attempts cost 10. Lane offset, sideways velocity,
and yaw error are penalized throughout the approach. Foot spacing uses signed
left-to-right separation to penalize cross-steps during both running and braking.

The controller holds head pitch/yaw and all six wrist joints at their nominal
targets. The actor still has 29 action channels so checkpoint 1400 can load.
The exported ONNX metadata carries a zero action scale for each held joint.
The robot uses its existing position servos to hold the targets.

## Kaggle T4 training

Open [zlab-training.ipynb](zlab-training.ipynb) in Kaggle, select a T4 GPU,
and run all cells. The notebook clones `main` and starts from the bundled
`Zlab_Gym/checkpoints/model_999.pt`. The lateral pilot adds body-frame
sideways correction capped at 0.35 m/s through the existing lateral
velocity command. The forward speed cap and braking curve stay the same.
It also scores checkpoint 699 for reference. Compare against the starting model
in the same report since all checkpoints use the new controller.

The default pilot runs one block of 300 updates at 512 environments per GPU.
During training, 35% of resets place the ball 1.5–3 m ahead and
0.35–0.75 m to either side of the nominal starting lane. The robot starts
with 0.35–0.90 m/s of forward velocity and uses the ball's lateral position
as its lane target. The remaining resets keep the normal 5–11 m approach. Every evaluation uses full-length approaches with pushes and noise
disabled. A candidate must earn more clean stops and fewer timeouts on each
of two 48-trial seeds. The lateral pilot also requires fewer timeouts
outside the lateral band on each seed. Falls and touches must stay at or below the starting
checkpoint’s counts on each seed, and lane drift may increase by at most 5 cm.
The pilot stops after the first block with no qualifying checkpoint.

Read `progress.json` in `/kaggle/working/zlab-robocup-braking-alignment-pilot` for
the selected checkpoint and the path to `comparison.json`. The report
includes the checkpoint 699 control, crossed-foot incidence, and overlapping
timeout counts for gap, lateral offset, yaw, and speed. Start a new work
directory when changing traction.

### Rewards for the running approach and stop

- Forward command tracking and progress reward encourage acceleration and
  running within the 2.8 m/s cap. Upright posture, smooth actions, low slip,
  and signed foot spacing discourage unstable and crossed steps.
- Lane and heading costs apply across the approach. When off-center, the
  sideways part of the reward favors tracking the correction command. The
  target blends toward zero sideways speed between 25 and 15 cm of lane
  error. Sideways tracking gets more weight near the ball while off-center.
  Brake tracking penalizes excess forward speed near the ball.
- A clean 1.5 s hold earns 24 terminal reward units. Ball contact costs 25 and
  other failed trials cost 10. The evaluation does not reward jumping or
  claim that successful stops establish a human running gait. Check stride
  alternation and foot clearance in replay before adding stronger disturbances.

For optional Hugging Face backups, supply a write-capable `HF_TOKEN` in
Kaggle Secrets and enable uploads in the notebook. Periodic training
checkpoints are stored under `checkpoints/<run-name>/model_<iteration>.pt`.
The selection file identifies which checkpoint to use.

## Stability training plan

1. Establish stopping performance at 2.8 m/s with zero pushes and sensor noise. Maintain
   randomized start position and heading, surface friction, mass, encoder bias,
   motor gains, and actuator delay. Review lane offset and heading alongside
   successful stops.
2. Once the clean profile passes on a fresh seed, use the optional robustness
   curriculum at 2.8 m/s. Increase torso pushes and sensor noise together
   through quarter, half, and full configured strength.
   Every block evaluates its final checkpoint on the same 48 trials as the
   incumbent. Favor passing the gate, then more clean stops, fewer touches and falls,
   and lower lane and heading error. Each profile must pass on a second seed
   and the clean profile before the next increase. The gates are at least
   46/48 clean stops, zero touches, at most one fall, ≤0.30 m at the 95th
   percentile of maximum lane offset, and ≤15° at the 95th percentile of
   maximum heading error.
3. Run an independent 96-trial check after tuning. Review failures by starting
   distance. For hardware transfer, validate joint mapping, held-joint targets,
   action scale, default pose, sign conventions, PD gains, and observation
   ordering before enabling the policy. Feed the same 29 raw previous-action
   values used during training into the observations. Apply the ONNX per-joint
   scales to motor targets, including the zero scales. If a deployment adapter
   hardcodes a uniform action scale, update it before running this policy.
4. Verify the state estimator and command latency against logs. Start on
   hardware at a lower command and advance through short approaches on the
   intended turf. Test friction changes, battery states, starting yaw, and
   lateral offsets before repeated runs at 2.8 m/s. Keep the strongest policy
   based on clean stops, ball touches, and falls. For match play, separately
   measure successful kicks and time from approach to kick. Revisit higher
   speed only after that sequence is reliable.

## Linux

Run a single block with Python 3.10–3.13, CUDA PyTorch and `flock`.

```bash
cd Zlab_Gym
python -m pip install -e . onnx
./train_ball_stability.sh checkpoints/model_1400.pt 300 512
```

Set `ZLAB_TORSO_PUSH_SCALE` and `ZLAB_OBSERVATION_NOISE_SCALE` for a manual
robustness block. The automated blocks can also run outside Kaggle.

```bash
python scripts/train_stability_blocks.py checkpoints/model_1400.pt \
  --work-dir logs/stability-blocks --hours 9
```

Evaluate the selected checkpoint at 2.8 m/s.

```bash
python scripts/play.py --checkpoint-file /absolute/path/to/model.pt \
  --eval-episodes 48 --num-envs 48 --device cuda:0 --seed 9187 \
  --disturbances True --torso-push-scale 0.25 --observation-noise-scale 0.25
```

## Transfer to the 22 × 14 m artificial-turf field

Each simulated world uses a rectangular 22 × 14 m playing boundary. The drill
starts 6 m behind the center line and places the ball 5–11 m ahead along the
center lane. Leaving the pitch is counted as a miss. The simulator uses a
flat plane with randomized foot traction and motor lag. The default friction range (0.35–1.15) represents simulation
uncertainty. This model approximates traction and samples 0–15 ms of motor
command lag. Turf compliance, surface roughness, gearbox backlash, voltage
sag, and motor temperature still need physical characterization. Measure
end-to-end control latency, motor current, temperature, slip, and braking on
the event turf with the competition
boot soles, then adjust `FOOT_FRICTION_RANGE` in the notebook or
`ZLAB_FOOT_FRICTION_MIN` and `ZLAB_FOOT_FRICTION_MAX` on Linux. Evaluate the
endpoints with `--foot-friction-min LOW --foot-friction-max HIGH` and test both running directions
across the turf. Check performance near seams or worn patches.

Measure speed, lane deviation, yaw, stop distance, touches, and falls from
center-field trials. Repeat from farther upfield and with the approach
rotated or offset. Keep path planning and sideline margins in the match
controller. The policy assumes a locally straight approach and uses forward
speed and yaw commands. Verify the real ball-location estimate and braking
trigger with the deployed camera and state estimator.
