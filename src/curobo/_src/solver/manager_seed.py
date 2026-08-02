"""Deterministic CPU/MPS action and trajectory seed manager."""

from __future__ import annotations

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator


class SeedManager:
    def __init__(
        self, device_cfg: DeviceCfg, action_dim: int,
        action_bound_lows: torch.Tensor, action_bound_highs: torch.Tensor,
        random_seed: int = 123, action_horizon: int = 1,
    ):
        self.device_cfg = device_cfg
        self.action_dim = action_dim
        self.action_bound_lows = action_bound_lows
        self.action_bound_highs = action_bound_highs
        self.random_seed = random_seed
        self.action_horizon = action_horizon
        self._generator = torch.Generator(device="cpu").manual_seed(random_seed)
        # IK/one-step rollouts still use an action horizon of one.  The shared
        # interpolation helper needs two knots, so keep that internal detail
        # separate from the public action-horizon contract.
        self._trajectory = TrajectorySeedGenerator(
            max(2, action_horizon), action_dim, device_cfg
        )

    def prepare_action_seeds(
        self, batch_size, num_seeds, seed_config=None,
        current_state=None, seed_traj=None,
    ):
        if seed_traj is not None:
            return seed_traj.position if isinstance(seed_traj, JointState) else seed_traj
        if seed_config is not None:
            value = seed_config.position if isinstance(seed_config, JointState) else seed_config
            return value.reshape(batch_size, num_seeds, self.action_horizon, self.action_dim)
        if current_state is not None:
            return self._trajectory.generate_constant_seeds(
                current_state.position, num_seeds
            )
        return self.generate_random_actions(batch_size, num_seeds)

    def prepare_trajectory_seeds(
        self, batch_size, num_seeds, current_state,
        seed_config=None, seed_traj=None,
    ):
        return self.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )

    def generate_random_actions(self, batch_size, num_seeds):
        unit = torch.rand(
            (batch_size, num_seeds, self.action_horizon, self.action_dim),
            generator=self._generator, dtype=self.device_cfg.dtype,
        ).to(self.device_cfg.device)
        return self.action_bound_lows + unit * (
            self.action_bound_highs - self.action_bound_lows
        )

    def prepare_deceleration_trajectory_seeds(
        self, batch_size, num_seeds, current_state,
        deceleration_time=None, deceleration_profile="exponential",
    ):
        del batch_size, deceleration_time
        if deceleration_profile not in ("exponential", "linear"):
            raise ValueError("deceleration_profile must be exponential or linear")
        seeds = self._trajectory.generate_constant_seeds(
            current_state.position, num_seeds
        )
        if current_state.velocity is not None:
            t = torch.linspace(
                0, 1, self.action_horizon,
                device=seeds.device, dtype=seeds.dtype,
            )
            scale = (1 - t) if deceleration_profile == "linear" else torch.exp(-5 * t)
            seeds = seeds + scale.view(1, 1, -1, 1) * current_state.velocity[:, None, None]
        return seeds

    def reset_seed(self):
        self._generator.manual_seed(self.random_seed)


__all__ = ["SeedManager"]
