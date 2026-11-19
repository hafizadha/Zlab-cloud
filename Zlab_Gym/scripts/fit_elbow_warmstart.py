"""Fit a small conditional elbow change while preserving the original MLP layout.

Input observations must be frozen-policy rollouts from --checkpoint. Reference
motion stays local. This is supervised preparation for AMP, not PPO training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT)]
from rsl_rl.models import MLPModel
from scripts.restricted_actor import penultimate_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(args):
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["actor_state_dict"]
    data = torch.load(args.observations, map_location="cpu", weights_only=True)
    observations, commands = data["observations"], data["commands"]
    steps, envs, dim = observations.shape
    if not all(torch.isfinite(data[key]).all() for key in ("observations", "commands", "actions")):
        raise ValueError("Rollout observations, commands and actions must be finite.")
    names = data["action_names"]
    if len(set(names)) != len(names) or envs < 2:
        raise ValueError("Need unique action names and at least two rollout environments.")
    obs = TensorDict({"actor": observations.reshape(-1, dim)}, batch_size=[steps * envs])
    keys = sorted((k for k in state if k.startswith("mlp.") and k.endswith(".weight")),
                  key=lambda k: int(k.split(".")[1]))
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", len(names),
                     hidden_dims=tuple(state[k].shape[0] for k in keys[:-1]),
                     activation="elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution",
                                       "init_std": 1., "std_type": "scalar"})
    actor.load_state_dict(state, strict=True)
    actor.eval()
    with torch.inference_mode():
        baseline = actor(obs)
        features = penultimate_features(actor, obs).double()
        reconstructed = features.float() @ actor.mlp[-1].weight.T + actor.mlp[-1].bias
        torch.testing.assert_close(reconstructed, baseline, atol=1e-5, rtol=1e-5)
        replay = baseline - data["actions"].reshape(-1, len(names))
        replay_max = float(replay.abs().max())
        replay_rms = float(replay.square().mean().sqrt())
    if replay_max >= .005 or replay_rms >= .001:
        raise ValueError(f"Source policy replay mismatch: max={replay_max}, RMS={replay_rms}")
    canonical = json.loads(args.profile.read_text())["joint_names"]
    columns = [names.index(n) for n in ("left_elbow", "right_elbow")]
    ref_columns = [canonical.index(n) for n in ("left_elbow", "right_elbow")]
    with np.load(args.reference, allow_pickle=False) as motion:
        poses = motion["joint_pos"]
        if poses.shape[1] != len(canonical) or not np.isfinite(poses).all():
            raise ValueError("Reference joint positions do not match the motor profile.")
        target_pose = torch.tensor(poses[:, ref_columns].mean(0)).double()
    scale = data["scale"].reshape(-1, len(names))[0, columns].double()
    offset = data["offset"].reshape(-1, len(names))[0, columns].double()
    if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or (scale <= 0).any():
        raise ValueError("Elbow actions must have positive, nonzero scales.")
    desired = (target_pose - offset) / scale
    command = commands.flatten()
    blend = ((command - 1.5) / .5).clamp(0, 1).double()
    cap = args.max_target_step_rad / scale
    target = (desired - baseline[:, columns].double()).clamp(-cap, cap) * blend[:, None]
    train_count = max(1, min(envs - 1, int(envs * .75)))
    train = torch.arange(len(command)) % envs < train_count
    validation = ~train
    slow, running = command < 1.5, command >= 2.
    if not (slow & validation).any() or not (running & validation).any():
        raise ValueError("Need both low-speed and running validation observations.")
    mean = features[train].mean(0)
    std = features[train].std(0).clamp_min(.05)
    x = torch.cat(((features - mean) / std,
                   torch.ones(len(features), 1, dtype=torch.float64)), 1)
    best, grid = None, []
    for stop_weight in (4., 16., 64.):
        weights = torch.where(slow, stop_weight, 1.).double()
        xx = x[train] * weights[train, None].sqrt()
        yy = target[train] * weights[train, None].sqrt()
        gram, rhs = xx.T @ xx / len(xx), xx.T @ yy / len(xx)
        for ridge in (1e-4, 1e-3, .01, .1, 1.):
            coeff = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64), rhs)
            prediction = x @ coeff
            slow_max = float((prediction[slow] * scale).abs().max().clamp_min(1e-8))
            shrink = min(1., float((cap / prediction.abs().amax(0).clamp_min(1e-8)).min()),
                         args.max_slow_drift_rad / slow_max)
            coeff *= shrink
            delta_q = (x @ coeff) * scale
            slow_drift = float(delta_q[validation & slow].abs().max())
            running_shift = float(-delta_q[validation & running].mean())
            row = dict(ridge=ridge, stop_weight=stop_weight,
                       validation_max_slow_drift_rad=slow_drift,
                       validation_mean_running_bend_rad=running_shift,
                       recorded_max_target_change_rad=float(delta_q.abs().max()))
            grid.append(row)
            if (slow_drift <= args.max_slow_drift_rad + 1e-10
                    and running_shift >= args.min_running_step_rad
                    and (best is None or running_shift > best[0])):
                best = (running_shift, coeff.clone(), row)
    report = dict(accepted=best is not None, source_sha256=digest(args.checkpoint),
                  observations_sha256=digest(args.observations), reference_sha256=digest(args.reference),
                  source_replay_max_action_error=replay_max, source_replay_rms_action_error=replay_rms,
                  grid=grid, validation_envs=list(range(train_count, envs)),
                  note="Validation environments select regularization; independent physical rollouts are required. Bounds apply only to recorded observations. No AMP/PPO updates performed.")
    if best is not None:
        _, coeff, chosen = best
        dw = (coeff[:-1] / std[:, None]).T.float()
        db = (coeff[-1] - mean @ (coeff[:-1] / std[:, None])).float()
        with torch.no_grad():
            actor.mlp[-1].weight[columns] += dw
            actor.mlp[-1].bias[columns] += db
        with torch.inference_mode():
            updated = actor(obs)
        delta = updated[:, columns] - baseline[:, columns]
        torch.testing.assert_close(delta, (x @ coeff).float(), atol=2e-5, rtol=2e-5)
        actual_q = delta.double() * scale
        if actual_q[slow].abs().max() > args.max_slow_drift_rad + 1e-5:
            raise ValueError("Exported actor violates the low-speed target bound.")
        if actual_q.abs().max() > args.max_target_step_rad + 1e-5:
            raise ValueError("Exported actor violates the total target-change bound.")
        protected = torch.ones(len(names), dtype=torch.bool)
        protected[columns] = False
        if not torch.equal(updated[:, protected], baseline[:, protected]):
            raise ValueError("Protected action outputs changed.")
        head_keys = {keys[-1], keys[-1].replace(".weight", ".bias")}
        for key, value in actor.state_dict().items():
            if key not in head_keys and not torch.equal(value, state[key]):
                raise ValueError(f"Protected actor state changed: {key}")
        report.update(chosen=chosen, feature_reconstruction_passed=True,
                      actual_max_slow_drift_rad=float(actual_q[slow].abs().max()),
                      actual_max_target_change_rad=float(actual_q.abs().max()),
                      protected_output_max_difference=0.)
        checkpoint["actor_state_dict"] = actor.state_dict()
        checkpoint["optimizer_state_dict"]["state"] = {}
        checkpoint["elbow_warmstart"] = {k: report[k] for k in
                                        ("source_sha256", "observations_sha256", "reference_sha256", "chosen")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output)
        reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
        actor.load_state_dict(reloaded["actor_state_dict"], strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(actor(obs), updated, atol=0, rtol=0)
        report["checkpoint"] = str(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "observations", "reference", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile", type=Path,
                        default=PROJECT / "src/assets/robots/Z01_std_add_head/walking_motor_profile.json")
    parser.add_argument("--max-target-step-rad", type=float, default=.25)
    parser.add_argument("--max-slow-drift-rad", type=float, default=.025)
    parser.add_argument("--min-running-step-rad", type=float, default=.04)
    args = parser.parse_args()
    budgets = (args.max_target_step_rad, args.max_slow_drift_rad, args.min_running_step_rad)
    if not all(np.isfinite(x) and x > 0 for x in budgets):
        parser.error("All target-change budgets must be positive.")
    fit(args)
