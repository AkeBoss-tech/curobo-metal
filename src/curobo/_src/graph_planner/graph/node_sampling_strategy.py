"""Seeded node sampling compatible with the pinned planner."""

from __future__ import annotations

import torch


class NodeSamplingStrategy:
    def __init__(
        self,
        config: PRMGraphPlannerCfg,
        action_lower_bounds: torch.Tensor,
        action_upper_bounds: torch.Tensor,
        cspace_distance_weight: torch.Tensor,
        action_dim: int,
        check_feasibility_fn,
        device_cfg=None,
    ):
        self.config = config
        self.low = action_lower_bounds
        self.high = action_upper_bounds
        self.distance_weight = cspace_distance_weight
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn
        self.device_cfg = device_cfg or config.device_cfg
        self.reset_seed()

    def reset_seed(self):
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.config.sampler_seed)

    def generate_action_samples(
        self, n_samples: int, bounded: bool = True, unit_ball: bool = False
    ):
        values = torch.rand(
            (n_samples, self.action_dim), generator=self._generator,
            dtype=self.low.dtype, device="cpu",
        ).to(self.low.device)
        if unit_ball:
            values = values * 2 - 1
            norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1)
            values = values / norm
        elif bounded:
            values = self.low + values * (self.high - self.low)
        return values

    def check_samples_feasibility(self, action_samples):
        return self.check_feasibility_fn(action_samples)

    def get_feasible_sample_set(self, x_samples):
        return x_samples[self.check_samples_feasibility(x_samples)]

    def generate_feasible_action_samples(self, num_samples: int):
        return self.get_feasible_sample_set(
            self.generate_action_samples(num_samples * self.config.sample_rejection_ratio)
        )[:num_samples]

    def generate_feasible_samples(self, num_samples: int) -> torch.Tensor:
        return self.generate_feasible_action_samples(num_samples)

    def generate_feasible_samples_in_ellipsoid(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        num_samples: int,
        max_sampling_radius: torch.Tensor,
    ) -> torch.Tensor:
        sample = self.generate_action_samples(num_samples, unit_ball=True)
        midpoint = (x_start + x_goal) * 0.5
        sample = midpoint + sample * torch.as_tensor(
            max_sampling_radius, device=sample.device, dtype=sample.dtype
        )
        return self.get_feasible_sample_set(torch.maximum(torch.minimum(sample, self.high), self.low))

    def compute_distance_from_line(
        self, vertices: torch.Tensor, x_start: torch.Tensor, x_goal: torch.Tensor
    ):
        segment = x_goal - x_start
        phase = ((vertices - x_start) * segment).sum(-1) / segment.square().sum().clamp_min(1e-12)
        closest = x_start + phase.clamp(0, 1)[:, None] * segment
        return torch.linalg.vector_norm((vertices - closest) * self.distance_weight, dim=-1)

    # These helpers were TorchScript/Warp acceleration entry points upstream.
    # The portable versions deliberately use ordinary differentiable tensors.
    @staticmethod
    def jit_compute_distance_from_line(
        vertices: torch.Tensor, x_start: torch.Tensor, x_goal: torch.Tensor
    ):
        segment = x_goal - x_start
        phase = ((vertices - x_start) * segment).sum(-1) / segment.square().sum().clamp_min(1e-12)
        closest = x_start + phase.clamp(0, 1).unsqueeze(-1) * segment
        return torch.linalg.vector_norm(vertices - closest, dim=-1)

    @staticmethod
    def jit_transform_unit_ball_to_ellipsoid_approximate(
        x_start,
        x_goal,
        distance_weight,
        max_sampling_radius: torch.Tensor,
        action_dim: int,
        rot_frame_col: torch.Tensor,
        unit_ball_samples: torch.Tensor,
        low_bounds: torch.Tensor,
        high_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Map unit-ball samples into a clipped prolate ellipsoid.

        The CUDA implementation has three numerically distinct rotation paths.
        In the portable backend they share a differentiable Householder-free
        approximation while retaining the exact public invocation layout.
        """
        del distance_weight, action_dim, rot_frame_col
        midpoint = (x_start + x_goal) * 0.5
        sample = midpoint + unit_ball_samples * max_sampling_radius.to(
            dtype=unit_ball_samples.dtype, device=unit_ball_samples.device
        )
        return torch.maximum(torch.minimum(sample, high_bounds), low_bounds)

    @staticmethod
    def jit_transform_unit_ball_to_ellipsoid_householder(
        x_start,
        x_goal,
        distance_weight,
        max_sampling_radius: torch.Tensor,
        action_dim: int,
        rot_frame_col: torch.Tensor,
        unit_ball_samples: torch.Tensor,
        low_bounds: torch.Tensor,
        high_bounds: torch.Tensor,
    ) -> torch.Tensor:
        return NodeSamplingStrategy.jit_transform_unit_ball_to_ellipsoid_approximate(
            x_start, x_goal, distance_weight, max_sampling_radius, action_dim,
            rot_frame_col, unit_ball_samples, low_bounds, high_bounds,
        )

    @staticmethod
    def jit_transform_unit_ball_to_ellipsoid_svd(
        x_start,
        x_goal,
        distance_weight,
        max_sampling_radius: torch.Tensor,
        action_dim: int,
        rot_frame_col: torch.Tensor,
        unit_ball_samples: torch.Tensor,
        low_bounds: torch.Tensor,
        high_bounds: torch.Tensor,
    ) -> torch.Tensor:
        return NodeSamplingStrategy.jit_transform_unit_ball_to_ellipsoid_approximate(
            x_start, x_goal, distance_weight, max_sampling_radius, action_dim,
            rot_frame_col, unit_ball_samples, low_bounds, high_bounds,
        )


__all__ = ["NodeSamplingStrategy"]
