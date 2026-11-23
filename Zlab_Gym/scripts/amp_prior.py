"""Adversarial motion reward for the existing MLP PPO actor."""

from pathlib import Path
import hashlib
import copy

import numpy as np
import torch
from torch import nn


class MotionPriorPPO:
    """Delegate PPO and learn a discriminator from genuine motion transitions."""

    def __init__(self, ppo, env, clip, weight=0.2, reset_prior=False, teacher_checkpoint=None,
                 joint_position_std_floor=0.1, pose_guidance_weight=0.0, arm_only=False,
                 leg_only=False, braking_guard_data=None, braking_guard_anchor=None,
                 braking_guard_budget=0.01, feature_mode="full",
                 braking_guard_projection="uniform", pose_guidance_std=0.75, discriminator_learning_rate=1e-4, braking_guard_mode="head", braking_guard_scope="all"):
        if braking_guard_mode not in ("head", "functional"):
            raise ValueError("Unknown braking guard mode.")
        if braking_guard_mode == "functional" and not braking_guard_data:
            raise ValueError("Functional guard requires calibration and anchor.")
        if braking_guard_mode == "functional" and (arm_only or leg_only):
            raise ValueError("Functional guard requires full mean adaptation, not restricted rows.")
        if braking_guard_scope not in ("all", "locomotion") or (braking_guard_scope != "all" and braking_guard_mode != "functional"):
            raise ValueError("Locomotion scope requires the functional guard.")
        self.braking_guard_scope = braking_guard_scope
        self.braking_guard_mode = braking_guard_mode
        self.braking_guard_projection = braking_guard_projection
        self.braking_guard = None
        self.braking_guard_config = (braking_guard_data, braking_guard_anchor, braking_guard_budget)
        if bool(braking_guard_data) != bool(braking_guard_anchor):
            raise ValueError("Braking guard needs both calibration data and anchor.")
        if braking_guard_data and not leg_only and braking_guard_mode != "functional":
            raise ValueError("Braking guard currently requires leg-only adaptation.")
        self.ppo = ppo
        self.env = env.unwrapped
        self.weight = weight
        self.reset_prior = reset_prior
        self.arm_only = arm_only
        self.leg_only = leg_only
        if arm_only and leg_only:
            raise ValueError("Choose either arm-only or leg-only adaptation.")
        self.arm_adapter = None
        self.arm_action_names = None
        if (arm_only or leg_only) and teacher_checkpoint:
            raise ValueError("Restricted adaptation cannot use the whole-body braking teacher.")
        if not np.isfinite(joint_position_std_floor) or joint_position_std_floor <= 0:
            raise ValueError("Joint position normalization floor must be finite and positive.")
        self.joint_position_std_floor = float(joint_position_std_floor)
        if not np.isfinite(pose_guidance_weight) or pose_guidance_weight < 0:
            raise ValueError("Pose guidance weight must be finite and nonnegative.")
        self.pose_guidance_weight = float(pose_guidance_weight)
        if not np.isfinite(pose_guidance_std) or pose_guidance_std <= 0:
            raise ValueError("Pose guidance standard deviation must be finite and positive.")
        self.pose_guidance_std = float(pose_guidance_std)
        if feature_mode not in ("full", "locomotion", "locomotion_joints", "wholebody_joints"):
            raise ValueError("Unknown AMP reference feature mode.")
        if feature_mode.startswith("locomotion") and arm_only:
            raise ValueError("Locomotion features cannot guide arm-only adaptation.")
        self.feature_mode = feature_mode
        self.robot = self.env.scene["robot"]
        pattern = (r"(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))"
                   if feature_mode.startswith("locomotion") else r"(?!head_.*|.*_wrist_.*).*")
        self.joint_ids, names = self.robot.find_joints((pattern,), preserve_order=True)
        self.reference_joint_names = list(names)
        if feature_mode.startswith("locomotion") and len(names) != 12:
            raise ValueError("Locomotion references require exactly twelve leg joints.")
        with np.load(clip, allow_pickle=False) as motion:
            if abs(float(motion["fps"].item()) * self.env.step_dt - 1) > 1e-5:
                raise ValueError("Reference and control timesteps must match.")
            # References retain the canonical walking motor profile order.
            import json
            profile = Path(__file__).resolve().parents[1] / "src/assets/robots/Z01_std_add_head/walking_motor_profile.json"
            canonical = json.loads(profile.read_text())["joint_names"]
            columns = [canonical.index(name) for name in names]
            features = np.concatenate((motion["joint_pos"][:, columns], motion["joint_vel"][:, columns]), axis=1)
            self.extended_features = "body_features" in motion and not feature_mode.endswith("_joints")
            if self.extended_features:
                if str(motion["body_feature_schema"].item()) != "root-gravity-feet-v1":
                    raise ValueError("Unknown AMP body feature schema.")
                if motion["body_features"].shape != (len(features), 21):
                    raise ValueError("Expected 21 root and foot reference features per frame.")
                features = np.concatenate((features, motion["body_features"]), axis=1)
            lengths = motion["clip_lengths"].tolist() if "clip_lengths" in motion else [len(features)]
            if sum(lengths) != len(features) or any(length < 2 for length in lengths):
                raise ValueError("Invalid demonstration clip boundaries.")
        if len(features) < 100 or not np.isfinite(features).all():
            raise ValueError("Need at least 100 finite demonstration frames.")
        self.identity = hashlib.sha256(Path(clip).read_bytes()).hexdigest()
        device = ppo.device
        demo = torch.tensor(features, device=device, dtype=torch.float32)
        self.pose_reference = demo[:, :len(self.joint_ids)]
        self.mean = demo.mean(0)
        self.std = demo.std(0).clamp_min(0.1)
        self.std[:len(self.joint_ids)].clamp_min_(self.joint_position_std_floor)
        pairs = []
        offset = 0
        for length in lengths:
            segment = demo[offset:offset + length]
            pairs.append(torch.cat((segment[:-1], segment[1:]), dim=1))
            offset += length
        self.demo = torch.cat(pairs)
        dim = self.demo.shape[1]
        self.discriminator = nn.Sequential(nn.Linear(dim, 256), nn.ELU(),
                                           nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
        # Keep the PPO optimizer interface delegated through __getattr__.
        if not np.isfinite(discriminator_learning_rate) or discriminator_learning_rate <= 0:
            raise ValueError("Discriminator learning rate must be finite and positive.")
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=float(discriminator_learning_rate))
        self.samples = []
        self.previous = None
        self.running = None
        self.style_sum = 0.0
        self.pose_sum = 0.0
        self.style_count = 0
        self.updates = 0
        self.teacher = None
        self.teacher_identity = None
        self.teacher_observations = []
        if teacher_checkpoint:
            teacher_state = torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
            self.teacher = copy.deepcopy(ppo.get_policy())
            self.teacher.load_state_dict(teacher_state["actor_state_dict"], strict=True)
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            self.teacher_identity = hashlib.sha256(Path(teacher_checkpoint).read_bytes()).hexdigest()
        if self.extended_features:
            self.foot_ids, _ = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        if self.arm_only or self.leg_only:
            self.configure_arm_adapter()

    def configure_arm_adapter(self):
        from scripts.restricted_actor import ArmOutputAdapter, LegOutputAdapter
        if self.arm_adapter is not None:
            self.arm_adapter.close()
        names = self.env.action_manager.get_term("joint_pos").target_names
        self.arm_action_names = list(names)
        adapter = ArmOutputAdapter if self.arm_only else LegOutputAdapter
        self.arm_adapter = adapter(self.ppo.get_policy(), self.ppo.optimizer, names)
        print("[Policy] Adapting", "arm" if self.arm_only else "leg", "outputs only:", [name for name, selected
              in zip(names, self.arm_adapter.mask.tolist()) if selected], flush=True)

    def __getattr__(self, name):
        return getattr(self.ppo, name)

    def features(self):
        data = self.robot.data
        joints = (data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids])
        if not self.extended_features:
            return torch.cat(joints, dim=1)
        from mjlab.utils.lab_api.math import quat_apply_inverse
        quaternion = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        foot_position = data.site_pos_w[:, self.foot_ids] - data.root_link_pos_w[:, None, :]
        foot_velocity = data.site_lin_vel_w[:, self.foot_ids]
        return torch.cat((*joints, data.root_link_lin_vel_b, data.root_link_ang_vel_b,
                          data.projected_gravity_b,
                          quat_apply_inverse(quaternion, foot_position).flatten(1),
                          quat_apply_inverse(quaternion, foot_velocity).flatten(1)), dim=1)

    def normalize(self, pair):
        mean = self.mean.repeat(2)
        std = self.std.repeat(2)
        return ((pair - mean) / std).clamp(-10, 10)

    def pose_similarity(self, features):
        """Broad nearest-pose guidance; AMP still scores temporal transitions."""
        positions = features[:, :len(self.joint_ids)]
        distance = torch.cdist(positions, self.pose_reference)
        error = distance.square().amin(dim=1) / (
            len(self.joint_ids) * self.pose_guidance_std ** 2
        )
        return torch.exp(-error)

    def act(self, obs):
        self.previous = self.features().clone()
        command = self.env.command_manager.get_command("twist")[:, 0]
        forward_speed = self.robot.data.root_link_lin_vel_w[:, 0]
        # Reference clips cover established running, starting at 1.5 m/s.
        self.running = (command >= 1.5) & (forward_speed >= 1.0)
        if self.teacher is not None and (command < 1.5).any():
            self.teacher_observations.append(obs[command < 1.5].detach().clone())
        if self.braking_guard_config[0] and self.braking_guard is None:
            raise RuntimeError("Load an anchor-compatible checkpoint before guarded training.")
        if self.braking_guard is not None:
            self.braking_guard.observe(obs, command < 1.5)
        return self.ppo.act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        current = self.features()
        pair = torch.cat((self.previous, current), dim=1)
        valid = ~dones.bool()
        running = self.running
        with torch.no_grad():
            score = self.discriminator(self.normalize(pair)).squeeze(-1)
            style = (1 - 0.25 * (score - 1).square()).clamp_min(0)
            ramp = min(1.0, self.updates / 50)
            bonus = self.weight * ramp * self.env.step_dt * style * valid * running
            pose_bonus = torch.zeros_like(bonus)
            if self.pose_guidance_weight:
                pose_bonus = (self.pose_guidance_weight * ramp * self.env.step_dt
                              * self.pose_similarity(current) * valid * running)
            self.style_sum += bonus.sum().item()
            self.pose_sum += pose_bonus.sum().item()
            self.style_count += len(bonus)
        if self.weight:
            self.samples.append(pair[valid & running].detach())
        self.ppo.process_env_step(obs, rewards + bonus + pose_bonus, dones, extras)

    def update(self, iteration=0):
        losses = self.ppo.update(iteration)
        losses["amp_mean_step_bonus"] = self.style_sum / max(1, self.style_count)
        losses["amp_mean_pose_bonus"] = self.pose_sum / max(1, self.style_count)
        self.style_sum = 0.0
        self.pose_sum = 0.0
        self.style_count = 0
        if self.teacher_observations:
            observations = torch.cat(self.teacher_observations, dim=0).clone()
            self.teacher_observations.clear()
            for _ in range(4):
                indices = torch.randint(len(observations), (256,), device=self.ppo.device)
                batch = observations[indices]
                with torch.no_grad():
                    target = self.teacher(batch, stochastic_output=False)
                prediction = self.ppo.get_policy()(batch, stochastic_output=False)
                teacher_loss = (prediction - target).square().mean()
                self.ppo.optimizer.zero_grad()
                teacher_loss.backward()
                nn.utils.clip_grad_norm_(self.ppo.get_policy().parameters(), self.ppo.max_grad_norm)
                self.ppo.optimizer.step()
            losses["amp_braking_teacher_mse"] = teacher_loss.item()
        if self.samples:
            generated = torch.cat(self.samples).clone()
            self.samples.clear()
            if generated.shape[0]:
                for _ in range(4):
                    ids = torch.randint(len(self.demo), (256,), device=self.ppo.device)
                    policy_ids = torch.randint(len(generated), (256,), device=self.ppo.device)
                    expert = self.normalize(self.demo[ids]).detach().requires_grad_(True)
                    policy = self.normalize(generated[policy_ids])
                    real_score = self.discriminator(expert)
                    fake_score = self.discriminator(policy)
                    gradient = torch.autograd.grad(real_score.sum(), expert, create_graph=True)[0]
                    loss = ((real_score - 1).square().mean() + (fake_score + 1).square().mean()
                            + 10 * gradient.square().sum(1).mean())
                    self.discriminator_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
                    self.discriminator_optimizer.step()
                losses.update(amp_discriminator=loss.item(), amp_real_score=real_score.mean().item(),
                              amp_policy_score=fake_score.mean().item())
        if self.braking_guard is not None:
            losses.update(self.braking_guard.project())
        self.updates += 1
        return losses

    def save(self):
        state = self.ppo.save()
        state["motion_prior"] = dict(discriminator=self.discriminator.state_dict(),
                                     optimizer=self.discriminator_optimizer.state_dict(), updates=self.updates,
                                     reference_sha256=self.identity, weight=self.weight,
                                     feature_mode=self.feature_mode,
                                     reference_joint_names=self.reference_joint_names,
                                     joint_position_std_floor=self.joint_position_std_floor,
                                     pose_guidance_weight=self.pose_guidance_weight,
                                     pose_guidance_std=self.pose_guidance_std,
                                     arm_only=self.arm_only,
                                     leg_only=self.leg_only,
                                     arm_action_names=self.arm_action_names,
                                     teacher_sha256=self.teacher_identity,
                                     braking_guard_mode=self.braking_guard_mode,
                                     braking_guard=self.braking_guard.save() if self.braking_guard else None)
        return state

    def load(self, state, load_cfg, strict=True):
        result = self.ppo.load(state, load_cfg, strict)
        prior = state.get("motion_prior")
        if prior and not self.reset_prior:
            if prior.get("feature_mode", "full") != self.feature_mode:
                raise ValueError("AMP reference feature mode changed; explicitly reset the prior.")
            if prior.get("reference_joint_names", self.reference_joint_names) != self.reference_joint_names:
                raise ValueError("AMP reference joint order changed.")
            if (prior.get("arm_only", False) != self.arm_only
                    or prior.get("leg_only", False) != self.leg_only):
                raise ValueError("AMP actor adaptation mode changed; explicitly reset the prior.")
            if (self.arm_only or self.leg_only) and prior.get("arm_action_names") != self.arm_action_names:
                raise ValueError("AMP restricted action layout changed.")
            if (prior.get("pose_guidance_weight", 0.0) != self.pose_guidance_weight
                    or (self.pose_guidance_weight and
                        prior.get("pose_guidance_std") != self.pose_guidance_std)):
                raise ValueError("AMP pose guidance changed; explicitly reset the prior.")
            if prior.get("joint_position_std_floor", 0.1) != self.joint_position_std_floor:
                raise ValueError("AMP normalization changed; explicitly reset the prior.")
            if prior["reference_sha256"] != self.identity or prior["weight"] != self.weight:
                raise ValueError("AMP reference or reward weight changed on resume.")
            if prior.get("teacher_sha256") != self.teacher_identity:
                raise ValueError("AMP braking teacher changed on resume.")
            self.discriminator.load_state_dict(prior["discriminator"])
            self.discriminator_optimizer.load_state_dict(prior["optimizer"])
            self.updates = prior["updates"]
        if self.arm_only or self.leg_only:
            # Capture protected rows AFTER loading seed/candidate actor weights.
            self.configure_arm_adapter()
        if prior and not self.reset_prior and prior.get("braking_guard_mode", "head") != self.braking_guard_mode:
            raise ValueError("Braking guard mode changed; explicitly reset the prior.")
        guard_state = prior.get("braking_guard") if prior else None
        if not self.reset_prior and bool(guard_state) != bool(self.braking_guard_config[0]):
            raise ValueError("Braking guard mode changed; explicitly reset the prior.")
        if self.braking_guard_config[0]:
            data, anchor, budget = self.braking_guard_config
            term = self.env.action_manager.get_term("joint_pos")
            if self.braking_guard_mode == "functional":
                from scripts.functional_braking_guard import FunctionalBrakingGuard
                self.braking_guard = FunctionalBrakingGuard(self.ppo.get_policy(), data, anchor, budget,
                                                            term.target_names, term.scale, scope=self.braking_guard_scope)
            else:
                from scripts.braking_guard import BrakingGuard
                self.braking_guard = BrakingGuard(self.ppo.get_policy(), self.arm_adapter, data, anchor, budget,
                                                  term.scale, projection_mode=self.braking_guard_projection)
            if guard_state and not self.reset_prior:
                self.braking_guard.restore(guard_state)
            self.braking_guard.project()
        return result
