"""Seeded node sampling compatible with the pinned planner."""

import torch


class NodeSamplingStrategy:
    def __init__(
        self, config, action_lower_bounds, action_upper_bounds, cspace_distance_weight,
        action_dim, check_feasibility_fn, device_cfg=None,
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

    def generate_action_samples(self, n_samples: int, bounded: bool = True, unit_ball: bool = False):
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

    def generate_feasible_action_samples(self, num_samples):
        return self.get_feasible_sample_set(
            self.generate_action_samples(num_samples * self.config.sample_rejection_ratio)
        )[:num_samples]

    generate_feasible_samples = generate_feasible_action_samples

    def generate_feasible_samples_in_ellipsoid(
        self, x_start, x_goal, num_samples, max_sampling_radius,
    ):
        sample = self.generate_action_samples(num_samples, unit_ball=True)
        midpoint = (x_start + x_goal) * 0.5
        sample = midpoint + sample * torch.as_tensor(
            max_sampling_radius, device=sample.device, dtype=sample.dtype
        )
        return self.get_feasible_sample_set(torch.maximum(torch.minimum(sample, self.high), self.low))

    def compute_distance_from_line(self, vertices, x_start, x_goal):
        segment = x_goal - x_start
        phase = ((vertices - x_start) * segment).sum(-1) / segment.square().sum().clamp_min(1e-12)
        closest = x_start + phase.clamp(0, 1)[:, None] * segment
        return torch.linalg.vector_norm((vertices - closest) * self.distance_weight, dim=-1)

    # These helpers were TorchScript/Warp acceleration entry points upstream.
    # The portable versions deliberately use ordinary differentiable tensors.
    @staticmethod
    def jit_compute_distance_from_line(vertices, x_start, x_goal, distance_weight):
        segment = x_goal - x_start
        phase = ((vertices - x_start) * segment).sum(-1) / segment.square().sum().clamp_min(1e-12)
        closest = x_start + phase.clamp(0, 1).unsqueeze(-1) * segment
        return torch.linalg.vector_norm((vertices - closest) * distance_weight, dim=-1)

    @staticmethod
    def jit_transform_unit_ball_to_ellipsoid_approximate(samples, x_start, x_goal, max_radius):
        midpoint = (x_start + x_goal) * 0.5
        return midpoint + samples * torch.as_tensor(max_radius, dtype=samples.dtype, device=samples.device)

    jit_transform_unit_ball_to_ellipsoid_householder = jit_transform_unit_ball_to_ellipsoid_approximate
    jit_transform_unit_ball_to_ellipsoid_svd = jit_transform_unit_ball_to_ellipsoid_approximate


__all__ = ["NodeSamplingStrategy"]
