"""Deterministic action and trajectory seed preparation for CPU and MPS.

The public shapes here deliberately follow the pinned cuRobo V2 optimizer
boundary: callers describe seeds as ``[batch, seed, ...]`` and this manager
returns the seed-expanded optimizer layout ``[batch * seed, ...]``.  The
implementation uses the portable Halton :class:`SampleBuffer`; it is ordinary
PyTorch state, not a CUDA graph or a raw device-buffer ABI.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator
from curobo._src.util.logging import log_and_raise, log_warn


TensorOrState = Union[torch.Tensor, JointState]


class SeedManager:
    """Prepare deterministic, bound-respecting optimizer seeds.

    The stateful sample buffer is intentionally retained between calls, as in
    V2.  ``reset_seed`` restores its initial Halton/index-selection state, so
    a solver retry is reproducible on either CPU or MPS.
    """

    def __init__(
        self,
        device_cfg: DeviceCfg,
        action_dim: int,
        action_bound_lows: torch.Tensor,
        action_bound_highs: torch.Tensor,
        random_seed: int = 123,
        action_horizon: int = 1,
    ):
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")

        self.device_cfg = device_cfg
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.random_seed = random_seed
        self.action_bound_lows = self._normalise_bounds(action_bound_lows, "action_bound_lows")
        self.action_bound_highs = self._normalise_bounds(action_bound_highs, "action_bound_highs")
        if bool((self.action_bound_highs < self.action_bound_lows).any().item()):
            raise ValueError("action_bound_highs must be greater than or equal to action_bound_lows")

        # V2 currently selects the Halton buffer after constructing a Roberts
        # buffer.  Only the selected public buffer affects observed behavior;
        # constructing a throw-away sequence would add allocation and does not
        # improve CPU/MPS compatibility.
        self.action_sample_generator = SampleBuffer.create_halton_sample_buffer(
            ndims=action_dim,
            device_cfg=device_cfg,
            up_bounds=self.action_bound_highs,
            low_bounds=self.action_bound_lows,
            seed=random_seed,
            store_buffer=2000,
        )
        self.trajectory_seed_generator: Optional[TrajectorySeedGenerator] = None
        if action_horizon > 1:
            self.trajectory_seed_generator = TrajectorySeedGenerator(
                action_horizon, action_dim, device_cfg
            )

    def _normalise_bounds(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim != 1 or value.shape[0] != self.action_dim:
            raise ValueError(f"{name} must have shape [action_dim]")
        if not (value.is_floating_point() or value.is_complex()):
            value = value.to(dtype=self.device_cfg.dtype)
        if value.is_complex():
            raise TypeError(f"{name} must be real-valued")
        value = value.to(**self.device_cfg.as_torch_dict())
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _value(value: Optional[TensorOrState], name: str) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, JointState):
            value = value.position
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor or JointState")
        return value

    @staticmethod
    def _positive_count(value: int, name: str, *, allow_zero: bool = False) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
            comparator = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {comparator} integer")

    def _to_local(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not self.device_cfg.is_same_torch_device(value.device):
            raise ValueError(f"{name} must be on {self.device_cfg.device}")
        if value.dtype != self.device_cfg.dtype:
            value = value.to(dtype=self.device_cfg.dtype)
        return value

    def _normalise_action_config(
        self, seed_config: TensorOrState, batch_size: int
    ) -> torch.Tensor:
        value = self._value(seed_config, "seed_config")
        assert value is not None
        value = self._to_local(value, "seed_config")
        if value.ndim == 2:
            if value.shape != (batch_size, self.action_dim):
                raise ValueError("seed_config must have shape [batch, seed, dof]")
            value = value.unsqueeze(1)
        elif value.ndim == 3:
            if value.shape[0] != batch_size and value.shape[1] == batch_size:
                value = value.permute(1, 0, 2)
            if value.shape[0] != batch_size or value.shape[2] != self.action_dim:
                raise ValueError("seed_config must have shape [batch, seed, dof]")
        else:
            raise ValueError("seed_config must have shape [batch, seed, dof]")
        if value.shape[1] < 1:
            raise ValueError("seed_config must contain at least one seed")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("seed_config must be finite")
        return value

    def _normalise_trajectory(
        self, seed_traj: TensorOrState, batch_size: int
    ) -> torch.Tensor:
        value = self._value(seed_traj, "seed_traj")
        assert value is not None
        value = self._to_local(value, "seed_traj")
        if value.ndim != 4 or value.shape[0] != batch_size:
            raise ValueError(
                "seed_traj must have shape [batch, seed, action_horizon, dof]"
            )
        if value.shape[2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                "seed_traj must have shape [batch, seed, action_horizon, dof]"
            )
        if value.shape[1] < 1:
            raise ValueError("seed_traj must contain at least one seed")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("seed_traj must be finite")
        return value

    def _normalise_current_state(self, current_state: JointState, batch_size: int) -> JointState:
        if not isinstance(current_state, JointState):
            raise TypeError("current_state must be a JointState")
        position = current_state.position
        if not self.device_cfg.is_same_torch_device(position.device):
            raise ValueError(f"current_state must be on {self.device_cfg.device}")
        if position.ndim != 2 or position.shape != (batch_size, self.action_dim):
            raise ValueError("current_state.position must have shape [batch, dof]")
        for field in ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt"):
            value = getattr(current_state, field)
            if value is not None and not self.device_cfg.is_same_torch_device(value.device):
                raise ValueError(f"current_state.{field} must be on {self.device_cfg.device}")
        if not bool(torch.isfinite(position).all().item()):
            raise ValueError("current_state.position must be finite")
        if position.dtype != self.device_cfg.dtype:
            # ``JointState.to`` converts every materialized derivative and
            # timing channel together, preserving the invariant required by
            # deceleration seed generation rather than mutating position only.
            current_state = current_state.to(self.device_cfg)
        return current_state

    def prepare_action_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        seed_config: Optional[torch.Tensor] = None,
        current_state: Optional[JointState] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return action seeds in optimizer layout ``[B * N, 1, dof]``.

        ``current_state`` and ``seed_traj`` are accepted for the shared solver
        call signature and intentionally do not alter single-step action seed
        selection, matching V2.  User configurations take priority, truncate
        deterministically when over-provisioned, and are Halton-padded when
        under-provisioned.
        """
        del current_state, seed_traj
        self._positive_count(batch_size, "batch_size")
        self._positive_count(num_seeds, "num_seeds", allow_zero=True)
        if num_seeds == 0:
            return torch.zeros(
                (batch_size, 1, self.action_dim), **self.device_cfg.as_torch_dict()
            )
        if seed_config is None:
            seeds = self.generate_random_actions(batch_size, num_seeds)
        else:
            provided = self._normalise_action_config(seed_config, batch_size)
            provided = provided[:, :num_seeds]
            remaining = num_seeds - provided.shape[1]
            seeds = (
                provided
                if remaining == 0
                else torch.cat((provided, self.generate_random_actions(batch_size, remaining)), dim=1)
            )
        return seeds.reshape(batch_size * num_seeds, 1, self.action_dim)

    def prepare_trajectory_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        current_state: JointState,
        seed_config: Optional[torch.Tensor] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return full trajectory seeds in optimizer layout ``[B * N, H, dof]``."""
        self._positive_count(batch_size, "batch_size")
        self._positive_count(num_seeds, "num_seeds")
        if self.trajectory_seed_generator is None:
            raise ValueError(
                "Cannot prepare trajectory seeds when action_horizon is one"
            )
        current_state = self._normalise_current_state(current_state, batch_size)
        pieces = []
        remaining = num_seeds
        if seed_traj is not None:
            available = self._normalise_trajectory(seed_traj, batch_size)
            take = min(available.shape[1], remaining)
            pieces.append(available[:, :take])
            remaining -= take
        if remaining:
            if seed_config is not None:
                configs = self._normalise_action_config(seed_config, batch_size)
                if configs.shape[1] < remaining:
                    raise ValueError(
                        f"Insufficient seed configs: {configs.shape[1]} provided, {remaining} needed"
                    )
                pieces.append(self.trajectory_seed_generator.generate_interpolated_seeds(
                    current_state.position, configs[:, :remaining], remaining
                ))
            else:
                pieces.append(self.trajectory_seed_generator.generate_constant_seeds(
                    current_state.position, remaining
                ))
        return torch.cat(pieces, dim=1).reshape(
            batch_size * num_seeds, self.action_horizon, self.action_dim
        )

    def generate_random_actions(self, batch_size: int, num_seeds: int) -> torch.Tensor:
        """Generate bound-respecting deterministic Halton action samples.

        The historical zero-seed sentinel intentionally has one all-zero seed,
        preserving V2's optimizer-buffer contract.
        """
        self._positive_count(batch_size, "batch_size")
        self._positive_count(num_seeds, "num_seeds", allow_zero=True)
        if num_seeds == 0:
            return torch.zeros(
                (batch_size, 1, self.action_dim), **self.device_cfg.as_torch_dict()
            )
        return self.action_sample_generator.get_samples(
            num_seeds * batch_size, bounded=True
        ).reshape(batch_size, num_seeds, self.action_dim)

    def prepare_deceleration_trajectory_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        current_state: JointState,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
    ) -> torch.Tensor:
        """Return dynamically integrated deceleration seeds ``[B * N, H, dof]``."""
        self._positive_count(batch_size, "batch_size")
        self._positive_count(num_seeds, "num_seeds")
        if self.trajectory_seed_generator is None:
            raise ValueError(
                "Cannot prepare trajectory seeds when action_horizon is one"
            )
        current_state = self._normalise_current_state(current_state, batch_size)
        seeds = self.trajectory_seed_generator.generate_deceleration_seeds(
            current_state,
            num_seeds,
            deceleration_time=deceleration_time,
            deceleration_profile=deceleration_profile,
        )
        return seeds.reshape(batch_size * num_seeds, self.action_horizon, self.action_dim)

    def reset_seed(self) -> None:
        """Restore the deterministic sample-buffer stream to its initial state."""
        self.action_sample_generator.reset()


__all__ = ["SeedManager"]
