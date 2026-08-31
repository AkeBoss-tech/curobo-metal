"""Deterministic, device-resident node sampling for portable PRM planning.

The pinned cuRobo implementation uses a Halton-backed ``SampleBuffer``.  It
also has CUDA-specialized TorchScript kernels for ellipsoidal sampling.  This
module keeps the observable sampler lifecycle and all tensor results on CPU or
MPS, while implementing the projection arithmetic with ordinary PyTorch.
"""

from __future__ import annotations

import torch

from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.torch_util import get_torch_jit_decorator


_EPS = 1.0e-10


class NodeSamplingStrategy:
    """Generate deterministic PRM nodes and filter them through feasibility.

    ``SampleBuffer`` owns the sequence and its reset state.  Consequently a
    reset reproduces subsequent samples exactly on a given device, rather than
    merely reseeding a process-global random generator.
    """

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
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if not callable(check_feasibility_fn):
            raise TypeError("check_feasibility_fn must be callable")

        self.config = config
        self.device_cfg = config.device_cfg if device_cfg is None else device_cfg
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn

        self.action_bound_lows = self._validate_vector(action_lower_bounds, "action_lower_bounds")
        self.action_bound_highs = self._validate_vector(action_upper_bounds, "action_upper_bounds")
        self.cspace_distance_weight = self._validate_vector(
            cspace_distance_weight, "cspace_distance_weight", strictly_positive=True
        )
        if torch.any(self.action_bound_lows > self.action_bound_highs):
            raise ValueError("action_lower_bounds must not exceed action_upper_bounds")

        # Preserve the aliases used by the earlier portable facade.
        self.low = self.action_bound_lows
        self.high = self.action_bound_highs
        self.distance_weight = self.cspace_distance_weight

        self.action_sample_generator = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            device_cfg=self.device_cfg,
            up_bounds=self.action_bound_highs,
            low_bounds=self.action_bound_lows,
            store_buffer=getattr(config, "sampler_buffer_size", 2000),
            seed=config.sampler_seed,
        )
        self._action_dim_rot_frame = torch.eye(
            self.action_dim, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

    def _validate_vector(
        self, value: torch.Tensor, name: str, *, strictly_positive: bool = False
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1 or value.numel() != self.action_dim:
            raise ValueError(f"{name} must have shape ({self.action_dim},)")
        if not self.device_cfg.is_same_torch_device(value.device):
            raise ValueError(f"{name} device must match device_cfg.device")
        if value.dtype != self.device_cfg.dtype:
            raise TypeError(f"{name} dtype must match device_cfg.dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite values")
        if strictly_positive and not bool((value > 0).all()):
            raise ValueError(f"{name} must contain values greater than zero")
        return value

    def reset_seed(self):
        """Restore the deterministic Halton and index-generator state."""
        self.action_sample_generator.reset()

    def generate_action_samples(
        self, n_samples: int, bounded: bool = True, unit_ball: bool = False
    ):
        """Return deterministic uniform samples or points inside a unit ball."""
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 0:
            raise ValueError("n_samples must be a non-negative integer")
        if not isinstance(bounded, bool) or not isinstance(unit_ball, bool):
            raise TypeError("bounded and unit_ball must be bool")
        if unit_ball:
            samples = self.action_sample_generator.get_gaussian_samples(n_samples, variance=1.0)
            norms = torch.linalg.vector_norm(samples, dim=-1, keepdim=True)
            # A Gaussian draw has zero probability of an exact zero norm, but
            # this branch makes the public empty/degenerate contract total.
            samples = torch.where(norms > _EPS, samples / norms.clamp_min(_EPS), samples)
            if self.action_dim < 3:
                radius = self.action_sample_generator.get_samples(n_samples, bounded=False)[:, :1]
                samples = samples * radius.clamp(0.0, 1.0)
            return samples
        return self.action_sample_generator.get_samples(n_samples, bounded=bounded)

    def check_samples_feasibility(self, action_samples):
        """Return a validated boolean feasibility mask for a sample batch."""
        if not isinstance(action_samples, torch.Tensor):
            raise TypeError("action_samples must be a torch.Tensor")
        if action_samples.ndim != 2 or action_samples.shape[1] != self.action_dim:
            raise ValueError(
                "action_samples must be a 2D tensor with shape (batch_size, action_dim)"
            )
        if not self.device_cfg.is_same_torch_device(action_samples.device):
            raise ValueError("action_samples device must match device_cfg.device")
        mask = self.check_feasibility_fn(action_samples)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("check_feasibility_fn must return a torch.Tensor")
        if mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != action_samples.shape[0]:
            raise ValueError("check_feasibility_fn must return a bool mask of shape (batch_size,)")
        if mask.device != action_samples.device:
            raise ValueError("feasibility mask must be on the sample device")
        return mask

    def get_feasible_sample_set(self, x_samples):
        return x_samples[self.check_samples_feasibility(x_samples)]

    def _candidate_count(self, num_samples: int) -> int:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 0:
            raise ValueError("num_samples must be a non-negative integer")
        return num_samples + int(num_samples * self.config.sample_rejection_ratio)

    def generate_feasible_action_samples(self, num_samples: int):
        candidates = self.generate_action_samples(self._candidate_count(num_samples), bounded=True)
        return self.get_feasible_sample_set(candidates)[:num_samples]

    def generate_feasible_samples(self, num_samples: int) -> torch.Tensor:
        return self.generate_feasible_action_samples(num_samples)

    def _validate_ellipsoid_inputs(self, x_start: torch.Tensor, x_goal: torch.Tensor) -> None:
        self._validate_vector(x_start, "x_start")
        self._validate_vector(x_goal, "x_goal")

    def generate_feasible_samples_in_ellipsoid(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        num_samples: int,
        max_sampling_radius: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_ellipsoid_inputs(x_start, x_goal)
        candidate_count = self._candidate_count(num_samples)
        samples = self.generate_action_samples(candidate_count, unit_ball=True)
        radius = torch.as_tensor(
            max_sampling_radius, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        if radius.numel() != 1 or not bool(torch.isfinite(radius).all()) or not bool((radius > 0).all()):
            raise ValueError("max_sampling_radius must be a finite scalar greater than zero")
        method = self.config.ellipsoid_projection_method
        transform = {
            "householder": self.jit_transform_unit_ball_to_ellipsoid_householder,
            "svd": self.jit_transform_unit_ball_to_ellipsoid_svd,
            "approximate": self.jit_transform_unit_ball_to_ellipsoid_approximate,
        }.get(method)
        if transform is None:
            raise ValueError("ellipsoid_projection_method must be householder, svd, or approximate")
        projected = transform(
            x_start,
            x_goal,
            self.cspace_distance_weight,
            radius,
            self.action_dim,
            self._action_dim_rot_frame,
            samples,
            self.action_bound_lows,
            self.action_bound_highs,
        )
        return self.get_feasible_sample_set(projected)[:num_samples]

    def compute_distance_from_line(
        self, vertices: torch.Tensor, x_start: torch.Tensor, x_goal: torch.Tensor
    ):
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2 or vertices.shape[1] != self.action_dim:
            raise ValueError("vertices must be a 2D tensor with shape (batch_size, action_dim)")
        if vertices.device != self.device_cfg.device:
            raise ValueError("vertices device must match device_cfg.device")
        self._validate_ellipsoid_inputs(x_start, x_goal)
        return self.jit_compute_distance_from_line(vertices, x_start, x_goal)

    @staticmethod
    def _weighted_direction(x_start: torch.Tensor, x_goal: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = x_goal - x_start
        norm = torch.linalg.vector_norm(direction * weight)
        e1 = torch.zeros_like(direction)
        e1[0] = 1.0
        unit = torch.where(norm > _EPS, direction / norm.clamp_min(_EPS), e1)
        return unit, norm

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
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
        """Project samples with the pinned Householder-frame arithmetic.

        This is standard PyTorch rather than upstream's CUDA-specialized JIT
        path, so it remains valid with ``PYTORCH_ENABLE_MPS_FALLBACK=0``.
        """
        del action_dim
        direction, min_radius = NodeSamplingStrategy._weighted_direction(
            x_start, x_goal, distance_weight
        )
        e1 = torch.zeros_like(direction)
        e1[0] = 1.0
        sign = torch.where(direction[0] >= 0, torch.ones_like(direction[0]), -torch.ones_like(direction[0]))
        vector = direction - sign * e1
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(_EPS)
        frame = rot_frame_col - 2.0 * torch.outer(vector, vector)
        radius = torch.as_tensor(max_sampling_radius, dtype=x_start.dtype, device=x_start.device)
        scales = torch.zeros_like(x_start)
        scales[0] = radius / 2.0
        scales[1:] = torch.sqrt(
            (radius.square() - min_radius.square()).clamp_min(0.0)
        ) / 2.0
        transformed = ((frame @ torch.diag(scales) @ unit_ball_samples.T).T / distance_weight)
        return torch.clamp(transformed + (x_start + x_goal) / 2.0, low_bounds, high_bounds).contiguous()

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
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
        """MPS-safe SVD-mode equivalent using the Householder orthogonal frame.

        ``aten::linalg_svd`` has no native MPS kernel.  The two paths serve
        the same purpose here (align axis zero to the start-goal direction),
        so this preserves device residency rather than enabling CPU fallback.
        """
        return NodeSamplingStrategy.jit_transform_unit_ball_to_ellipsoid_householder(
            x_start, x_goal, distance_weight, max_sampling_radius, action_dim,
            rot_frame_col, unit_ball_samples, low_bounds, high_bounds,
        )

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
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
        """Projection without forming an orientation matrix (portable fast path)."""
        del action_dim, rot_frame_col
        direction, min_radius = NodeSamplingStrategy._weighted_direction(
            x_start, x_goal, distance_weight
        )
        projection = unit_ball_samples @ direction
        perpendicular = unit_ball_samples - projection.unsqueeze(-1) * direction
        perpendicular_norm = torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True)
        normalized = torch.where(
            perpendicular_norm > _EPS,
            perpendicular / perpendicular_norm.clamp_min(_EPS),
            torch.zeros_like(perpendicular),
        )
        radius = torch.as_tensor(max_sampling_radius, dtype=x_start.dtype, device=x_start.device)
        major = projection.unsqueeze(-1) * direction * radius
        minor = normalized * perpendicular_norm * min_radius
        return torch.clamp(major + minor + (x_start + x_goal) / 2.0, low_bounds, high_bounds).contiguous()

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_compute_distance_from_line(
        vertices: torch.Tensor, x_start: torch.Tensor, x_goal: torch.Tensor
    ):
        line = x_goal - x_start
        denominator = line.square().sum().clamp_min(_EPS)
        progress = ((vertices - x_start) * line).sum(dim=-1) / denominator
        closest = x_start + progress.clamp(0.0, 1.0).unsqueeze(-1) * line
        return torch.linalg.vector_norm(vertices - closest, dim=-1)


__all__ = ["NodeSamplingStrategy"]
