"""Adversarial motion reward for the existing MLP PPO actor."""

from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn


class MotionPriorPPO:
    """Delegate PPO and learn a discriminator from genuine motion transitions."""

    def __init__(self, ppo, env, clip, weight=0.2, reset_prior=False):
        self.ppo = ppo
        self.env = env.unwrapped
        self.weight = weight
        self.reset_prior = reset_prior
        self.robot = self.env.scene["robot"]
        self.joint_ids, names = self.robot.find_joints(
            (r"(?!head_.*|.*_wrist_.*).*",), preserve_order=True
        )
        with np.load(clip, allow_pickle=False) as motion:
            if abs(float(motion["fps"].item()) * self.env.step_dt - 1) > 1e-5:
                raise ValueError("Reference and control timesteps must match.")
            # References retain the canonical walking motor profile order.
            import json
            profile = Path(__file__).resolve().parents[1] / "src/assets/robots/Z01_std_add_head/walking_motor_profile.json"
            canonical = json.loads(profile.read_text())["joint_names"]
            columns = [canonical.index(name) for name in names]
            features = np.concatenate((motion["joint_pos"][:, columns], motion["joint_vel"][:, columns]), axis=1)
            lengths = motion["clip_lengths"].tolist() if "clip_lengths" in motion else [len(features)]
            if sum(lengths) != len(features) or any(length < 2 for length in lengths):
                raise ValueError("Invalid demonstration clip boundaries.")
        if len(features) < 100 or not np.isfinite(features).all():
            raise ValueError("Need at least 100 finite demonstration frames.")
        self.identity = hashlib.sha256(Path(clip).read_bytes()).hexdigest()
        device = ppo.device
        demo = torch.tensor(features, device=device, dtype=torch.float32)
        self.mean = demo.mean(0)
        self.std = demo.std(0).clamp_min(0.1)
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
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.samples = []
        self.previous = None
        self.running = None
        self.style_sum = 0.0
        self.style_count = 0
        self.updates = 0

    def __getattr__(self, name):
        return getattr(self.ppo, name)

    def features(self):
        data = self.robot.data
        return torch.cat((data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids]), dim=1)

    def normalize(self, pair):
        mean = self.mean.repeat(2)
        std = self.std.repeat(2)
        return ((pair - mean) / std).clamp(-10, 10)

    def act(self, obs):
        self.previous = self.features().clone()
        command = self.env.command_manager.get_command("twist")[:, 0]
        forward_speed = self.robot.data.root_link_lin_vel_w[:, 0]
        # Reference clips cover established running, starting at 1.5 m/s.
        self.running = (command >= 1.5) & (forward_speed >= 1.0)
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
            self.style_sum += bonus.sum().item()
            self.style_count += len(bonus)
        self.samples.append(pair[valid & running].detach())
        self.ppo.process_env_step(obs, rewards + bonus, dones, extras)

    def update(self, iteration=0):
        losses = self.ppo.update(iteration)
        losses["amp_mean_step_bonus"] = self.style_sum / max(1, self.style_count)
        self.style_sum = 0.0
        self.style_count = 0
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
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
                    self.optimizer.step()
                losses.update(amp_discriminator=loss.item(), amp_real_score=real_score.mean().item(),
                              amp_policy_score=fake_score.mean().item())
        self.updates += 1
        return losses

    def save(self):
        state = self.ppo.save()
        state["motion_prior"] = dict(discriminator=self.discriminator.state_dict(),
                                     optimizer=self.optimizer.state_dict(), updates=self.updates,
                                     reference_sha256=self.identity, weight=self.weight)
        return state

    def load(self, state, load_cfg, strict=True):
        result = self.ppo.load(state, load_cfg, strict)
        prior = state.get("motion_prior")
        if prior and not self.reset_prior:
            if prior["reference_sha256"] != self.identity or prior["weight"] != self.weight:
                raise ValueError("AMP reference or reward weight changed on resume.")
            self.discriminator.load_state_dict(prior["discriminator"])
            self.optimizer.load_state_dict(prior["optimizer"])
            self.updates = prior["updates"]
        return result
