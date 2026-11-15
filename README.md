# Zlab Cloud · Ball approach stability

Fine-tune checkpoint 1400 at a fixed 2.8 m/s command cap. Each trial places
the ball 5–11 m ahead. A clean stop requires the robot to settle within 1 m
of the ball for 0.4 s, below 0.25 m/s, within 0.25 m laterally, and with yaw
within 12° of the lane. Robot–ball contact ends the approach and costs 25
reward units. Other failed attempts cost 10. Lane offset, sideways velocity,
and yaw error are penalized throughout the approach.

The controller holds head pitch/yaw and all six wrist joints at their nominal
targets. The actor still has 29 action channels so checkpoint 1400 can load.
The exported ONNX metadata carries a zero action scale for each held joint.
The robot uses its existing position servos to hold the targets.

## Kaggle T4 training

Open [zlab-training.ipynb](zlab-training.ipynb) in Kaggle, select a T4 GPU,
and run all cells. The notebook clones `main` and uses the bundled block 50
checkpoint, `Zlab_Gym/checkpoints/model_199_b50.pt`. The original 1400
checkpoint is also available.

The default pilot runs one block of 300 updates at 512 environments per GPU.
Each checkpoint at 100, 200, and 300 updates is compared with the starting
policy on two sets of 48 trials, with pushes and sensor noise disabled. To
replace the baseline, a candidate must improve clean stops and reduce
95th-percentile lane drift by at least 0.05 m on both seeds, with falls and
touches at or below baseline on each seed. The pilot stops after the first
block with no qualifying checkpoint. Increase `PILOT_BLOCKS` to allow further
blocks if training improves.

Read `progress.json` in `/kaggle/working/zlab-robocup-block50-pilot` for the
selected checkpoint and the path to `comparison.json`. Use a new work
directory for a fresh comparison or different traction settings.

For optional Hugging Face backups, supply a write-capable `HF_TOKEN` in
Kaggle Secrets and enable uploads in the notebook. Periodic training
checkpoints are stored under `checkpoints/<run-name>/model_<iteration>.pt`.
The selection file identifies which checkpoint to use.

## Stability training plan

1. Establish stopping performance at 2.8 m/s with zero pushes and sensor noise. Maintain
   randomized start position and heading, surface friction, mass, encoder bias,
   motor gains, and actuator delay. Review lane offset and heading alongside
   successful stops.
2. For the optional robustness curriculum, keep 2.8 m/s throughout. Increase torso pushes and
   sensor noise together through quarter, half, and full configured strength.
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
