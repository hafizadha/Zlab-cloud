# Zlab Cloud · Ball approach stability

Train the Z01 ball approach at a fixed 2.8 m/s command cap. Full-length
trials place the ball 5–11 m ahead. A clean stop requires the robot to settle within 1 m
of the ball for 5 continuous seconds, below 0.25 m/s, within 0.25 m laterally, and with yaw
within 12° of the lane. Robot–ball contact ends the approach and costs 25
reward units. Other failed attempts cost 10. Lane offset, sideways velocity,
and yaw error are penalized throughout the approach. Foot spacing uses signed
left-to-right separation to penalize cross-steps during both running and braking.

The controller holds head pitch/yaw and all six wrist joints at their nominal
targets. The actor still has 29 action channels so checkpoint 1400 can load.
The exported ONNX metadata carries a zero action scale for each held joint.
The robot uses its existing position servos to hold the targets.

## Kaggle T4 training

Open [zlab-training.ipynb](zlab-training.ipynb), select T4 and run all
cells. The notebook clones `main`, starts from the validated original 1199,
and scores 1499 from Kaggle version 21 as a reference. It runs 300 PPO updates
at 512 environments per GPU and saves every 100 updates.

This pilot uses the earlier braking profile tested on 1499. It starts
deceleration sooner and targets a 0.75 m gap. When the robot is close and
centered, the training reward penalizes lateral speed and torso tilt to
help it finish braking and stand. About 65% of resets are full approaches,
25% are rolling off-center approaches, and 10% start in the stop zone at rest.

The evaluation uses full approaches at 2.8 m/s with the earlier braking
profile and a continuous five-second hold. It compares each save against
1199 on seeds 75169 and 85313 and records 1499 as a reference. Read
`progress.json` and `comparison.json` under
`/kaggle/working/zlab-robocup-early-braking-pilot`. To preview the pilot's
controller in `scripts/play.py`, pass `--early-braking True`.

### Rewards for the running approach and stop

- Forward command tracking and progress reward encourage acceleration and
  running within the 2.8 m/s cap. Upright posture, smooth actions, low slip,
  and signed foot spacing discourage unstable and crossed steps.
- Lane and heading costs apply across the approach. When off-center, the
  sideways part of the reward favors tracking the correction command. The
  target blends toward zero sideways speed between 25 and 15 cm of lane
  error. Sideways tracking gets more weight near the ball while off-center.
  Brake tracking penalizes excess forward speed near the ball. The pilot also
  penalizes forward speed above 2.9 m/s and backward drift during the final approach.
- A clean 5 s hold earns 24 terminal reward units. Ball contact costs 25 and
  other failed trials cost 10. The evaluation does not reward jumping or
  claim that successful stops establish a human running gait. Check stride
  alternation and foot clearance in replay before adding stronger disturbances.

For optional Hugging Face backups during training, configure the uploader with
a write-capable `HF_TOKEN`. Periodic training
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
