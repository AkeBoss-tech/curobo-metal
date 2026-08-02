"""Portable high-level depth/TSDF camera-pose refinement.

The pinned implementation owns a Warp sparse-block raycaster.  This facade
instead delegates to the actual dense PyTorch mapper renderer and returns the
same high-level result shape: ``(Pose, alignment_error, n_iterations)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from curobo._src.types.pose import Pose
from curobo_metal.ops.perception import CameraObservation


@dataclass
class BlockSparseRefinementState:
    """Portable accepted-pose state, cloneable without Warp graph buffers."""

    pose: object
    loss: object = None
    iterations: int = 0
    depth: Optional[torch.Tensor] = None
    intrinsics: Optional[torch.Tensor] = None
    best_n_valid: Optional[torch.Tensor] = None
    lambda_damping: Optional[torch.Tensor] = None

    def clone(self):
        def clone(value):
            return value.clone() if hasattr(value, "clone") else value
        return type(self)(clone(self.pose), clone(self.loss), self.iterations, clone(self.depth),
                          clone(self.intrinsics), clone(self.best_n_valid), clone(self.lambda_damping))

    def copy_(self, other):
        for name in ("pose", "loss", "depth", "intrinsics", "best_n_valid", "lambda_damping"):
            source, target = getattr(other, name), getattr(self, name)
            if source is None:
                setattr(self, name, None)
            elif hasattr(target, "copy_"):
                target.copy_(source)
            else:
                setattr(self, name, source.clone() if hasattr(source, "clone") else source)
        self.iterations = other.iterations
        return self


@dataclass
class BlockSparseRaycastRefinerCfg:
    n_points: int = (1280 * 720) // 10
    minimum_valid_depth_pixels: int = 100
    max_iterations: int = 100
    inner_iterations: int = 4
    distance_threshold: float = 0.1
    min_valid_ratio: float = 0.1
    n_samples_per_ray: int = 1
    tile_block_dim: int = 256
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0
    minimum_tsdf_weight: float = 2.0
    lambda_initial: float = 1e-3
    lambda_factor: float = 10.0
    lambda_min: float = 1e-7
    lambda_max: float = 1e7
    rho_min: float = 0.25
    # Portable renderer update magnitude.  Retained in addition to the V2
    # fields because the dense backend does not expose raw LM raycast kernels.
    learning_rate: float = 0.1
    iterations: Optional[int] = None

    def __post_init__(self) -> None:
        if self.iterations is not None:
            self.max_iterations = self.iterations
        if self.max_iterations < 0 or self.inner_iterations < 1:
            raise ValueError("max_iterations must be nonnegative and inner_iterations positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")

    @property
    def outer_iterations(self) -> int:
        return self.max_iterations // self.inner_iterations


class BlockSparseRaycastPoseRefiner:
    """Depth refinement against a portable dense mapper/TSDF integrator."""

    def __init__(self, integrator, config: Optional[BlockSparseRaycastRefinerCfg] = None, *, cfg=None):
        if config is not None and cfg is not None:
            raise TypeError("pass either config or cfg, not both")
        self.integrator = integrator
        self.mapper = integrator
        self.config = config or cfg or BlockSparseRaycastRefinerCfg()
        # Old facade exposed cfg; retaining the alias makes it source-compatible.
        self.cfg = self.config

    @staticmethod
    def _native_mapper(value):
        for attribute in ("mapper", "_mapper"):
            if hasattr(value, attribute):
                value = getattr(value, attribute)
        if not hasattr(value, "refine_pose"):
            raise TypeError("integrator must expose a portable dense mapper")
        return value

    @staticmethod
    def _matrix(value, *, device, dtype) -> torch.Tensor:
        matrix = value.get_matrix() if hasattr(value, "get_matrix") else torch.as_tensor(value)
        if matrix.ndim == 3:
            if matrix.shape[0] != 1:
                raise ValueError("portable pose refinement supports one estimated pose")
            matrix = matrix[0]
        if matrix.shape != (4, 4):
            raise ValueError("estimated_pose must be a Pose or [4, 4] transform")
        return matrix.to(device=device, dtype=dtype)

    def refine_pose(self, depth, intrinsics, estimated_pose):
        native = self._native_mapper(self.integrator)
        target_device = native.state.tsdf.device
        target_dtype = native.state.tsdf.dtype
        if not isinstance(depth, torch.Tensor) or depth.ndim != 2:
            raise ValueError("depth must be a [H, W] tensor")
        if not isinstance(intrinsics, torch.Tensor) or intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be a [3, 3] tensor")
        valid = torch.isfinite(depth) & (depth >= self.config.depth_minimum_distance) & (depth <= self.config.depth_maximum_distance)
        if int(valid.sum().item()) < self.config.minimum_valid_depth_pixels:
            raise ValueError("depth has fewer valid pixels than minimum_valid_depth_pixels")
        observation = CameraObservation(
            depth.to(target_device, dtype=target_dtype),
            intrinsics.to(target_device, dtype=target_dtype),
            self._matrix(estimated_pose, device=target_device, dtype=target_dtype),
        )
        result = native.refine_pose(
            observation,
            iterations=self.config.max_iterations,
            learning_rate=self.config.learning_rate,
        )
        pose = Pose.from_matrix(result.camera_to_world)
        return pose, float(result.loss.detach().cpu()), int(result.iterations)

    __call__ = refine_pose


__all__ = ["BlockSparseRefinementState", "BlockSparseRaycastRefinerCfg", "BlockSparseRaycastPoseRefiner"]
