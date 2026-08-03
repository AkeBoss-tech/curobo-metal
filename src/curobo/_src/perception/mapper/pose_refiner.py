"""Portable high-level depth/TSDF camera-pose refinement.

The pinned implementation owns a Warp sparse-block raycaster.  This facade
instead delegates to the actual dense PyTorch mapper renderer and returns the
same high-level result shape: ``(Pose, alignment_error, n_iterations)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
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
    environment_indices: Optional[torch.Tensor] = None
    map_generation: Optional[torch.Tensor] = None

    def clone(self):
        def clone(value):
            return value.clone() if hasattr(value, "clone") else value
        return type(self)(clone(self.pose), clone(self.loss), self.iterations, clone(self.depth),
                          clone(self.intrinsics), clone(self.best_n_valid), clone(self.lambda_damping),
                          clone(self.environment_indices), clone(self.map_generation))

    def copy_(self, other):
        for name in (
            "pose", "loss", "depth", "intrinsics", "best_n_valid", "lambda_damping",
            "environment_indices", "map_generation",
        ):
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
            if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
                raise ValueError("iterations must be a nonnegative integer")
            self.max_iterations = self.iterations
        for name in (
            "n_points", "minimum_valid_depth_pixels", "max_iterations", "inner_iterations",
            "n_samples_per_ray", "tile_block_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n_points < 1 or self.minimum_valid_depth_pixels < 1 or self.max_iterations < 0 or self.inner_iterations < 1:
            raise ValueError("n_points/minimum_valid_depth_pixels must be positive, max_iterations nonnegative, and inner_iterations positive")
        if self.n_samples_per_ray < 1 or self.tile_block_dim < 1:
            raise ValueError("n_samples_per_ray and tile_block_dim must be positive")
        for name in (
            "distance_threshold", "min_valid_ratio", "depth_minimum_distance",
            "depth_maximum_distance", "minimum_tsdf_weight", "lambda_initial", "lambda_factor",
            "lambda_min", "lambda_max", "rho_min", "learning_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.distance_threshold <= 0 or not 0.0 <= self.min_valid_ratio <= 1.0:
            raise ValueError("distance_threshold must be positive and min_valid_ratio in [0, 1]")
        if self.depth_minimum_distance < 0 or self.depth_minimum_distance >= self.depth_maximum_distance:
            raise ValueError("depth distance bounds must satisfy 0 <= minimum < maximum")
        if self.minimum_tsdf_weight < 0:
            raise ValueError("minimum_tsdf_weight must be nonnegative")
        if self.lambda_initial <= 0 or self.lambda_factor <= 0 or self.lambda_min <= 0 or self.lambda_max < self.lambda_min:
            raise ValueError("lambda parameters must be positive with lambda_max >= lambda_min")
        if not 0.0 <= self.rho_min <= 1.0:
            raise ValueError("rho_min must be in [0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.max_iterations < 0 or self.inner_iterations < 1:
            raise ValueError("max_iterations must be nonnegative and inner_iterations positive")

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
        self._last_state: BlockSparseRefinementState | None = None

    @property
    def device(self) -> torch.device:
        return self._native_mapper(self.integrator).state.tsdf.device

    @property
    def last_state(self) -> BlockSparseRefinementState | None:
        return self._last_state

    def reset(self) -> None:
        """Drop portable cached refinement results without mutating the map."""
        self._last_state = None

    def reset_cuda_graph(self) -> None:
        raise NotImplementedError("CUDA graph refinement is unavailable on CPU/MPS")

    @staticmethod
    def _native_mapper(value):
        for attribute in ("mapper", "_mapper"):
            if hasattr(value, attribute):
                value = getattr(value, attribute)
        if not hasattr(value, "refine_pose"):
            raise TypeError("integrator must expose a portable dense mapper")
        return value

    @staticmethod
    def _matrices(value, *, device, dtype) -> torch.Tensor:
        matrix = value.get_matrix() if hasattr(value, "get_matrix") else torch.as_tensor(value)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.ndim != 3 or matrix.shape[-2:] != (4, 4):
            raise ValueError("estimated_pose must be a Pose or [..., 4, 4] transform")
        matrix = matrix.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("estimated_pose must be finite")
        expected = matrix.new_tensor((0.0, 0.0, 0.0, 1.0))
        if not torch.allclose(matrix[..., 3, :], expected.expand_as(matrix[..., 3, :]), atol=1e-5, rtol=0):
            raise ValueError("estimated_pose bottom row must be [0,0,0,1]")
        rotation = matrix[..., :3, :3]
        identity = torch.eye(3, device=device, dtype=dtype)
        if not torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-4, rtol=0):
            raise ValueError("estimated_pose rotation must be orthonormal")
        return matrix

    def _reject_raw_options(self) -> None:
        """Avoid silently treating sparse-raycast controls as dense settings."""
        defaults = BlockSparseRaycastRefinerCfg()
        for name in ("n_points", "n_samples_per_ray", "tile_block_dim", "minimum_tsdf_weight",
                     "lambda_initial", "lambda_factor", "lambda_min", "lambda_max", "rho_min"):
            if getattr(self.config, name) != getattr(defaults, name):
                raise NotImplementedError(
                    f"{name} controls raw CUDA/Warp sparse-raycast refinement and is unavailable on CPU/MPS"
                )

    @staticmethod
    def _batch_inputs(depth, intrinsics, matrices, *, environments: int, env_indices: torch.Tensor | None):
        if not isinstance(depth, torch.Tensor) or depth.ndim not in (2, 3):
            raise ValueError("depth must be [H, W] or [batch, H, W]")
        if not isinstance(intrinsics, torch.Tensor) or intrinsics.ndim not in (2, 3) or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must be [3, 3] or [batch, 3, 3]")
        batched = depth.ndim == 3
        values = depth.unsqueeze(0) if depth.ndim == 2 else depth
        batch = values.shape[0]
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape[0] not in (1, batch):
            raise ValueError("intrinsics batch must be one or match depth batch")
        if intrinsics.shape[0] == 1 and batch > 1:
            intrinsics = intrinsics.expand(batch, -1, -1)
        if matrices.shape[0] not in (1, batch):
            raise ValueError("estimated_pose batch must be one or match depth batch")
        if matrices.shape[0] == 1 and batch > 1:
            matrices = matrices.expand(batch, -1, -1)
        if env_indices is None:
            if batch > environments:
                raise ValueError("depth batch exceeds mapper environments; provide a compatible env_indices selection")
            indices = torch.arange(batch, device=values.device, dtype=torch.int64)
        else:
            if not isinstance(env_indices, torch.Tensor) or env_indices.dtype != torch.int64 or env_indices.ndim != 1:
                raise ValueError("env_indices must be an int64 vector")
            indices = env_indices.to(values.device)
            if indices.shape != (batch,) or len(torch.unique(indices)) != batch or bool(((indices < 0) | (indices >= environments)).any()):
                raise ValueError("env_indices must be unique, in range, and parallel to depth batch")
        return values, intrinsics, matrices, indices, batched

    def refine_pose(self, depth, intrinsics, estimated_pose, *, env_indices: torch.Tensor | None = None):
        """Refine one or many camera poses against selected dense-map environments.

        Rank-two depth returns the pinned ``(Pose, float, int)`` tuple.  Rank-
        three depth returns ``(Pose(batch), loss[batch], iterations[batch])``.
        """
        self._reject_raw_options()
        native = self._native_mapper(self.integrator)
        target_device = native.state.tsdf.device
        target_dtype = native.state.tsdf.dtype
        matrices = self._matrices(estimated_pose, device=target_device, dtype=target_dtype)
        depths, intrinsics, matrices, indices, batched = self._batch_inputs(
            depth, intrinsics, matrices, environments=native.config.environments, env_indices=env_indices
        )
        depths = depths.to(target_device, dtype=target_dtype)
        intrinsics = intrinsics.to(target_device, dtype=target_dtype)
        indices = indices.to(target_device)
        valid = torch.isfinite(depths) & (depths >= self.config.depth_minimum_distance) & (depths <= self.config.depth_maximum_distance)
        valid_count = valid.reshape(len(depths), -1).sum(-1)
        if bool((valid_count < self.config.minimum_valid_depth_pixels).any()):
            raise ValueError("depth has fewer valid pixels than minimum_valid_depth_pixels")
        outputs, losses, iterations = [], [], []
        for item in range(len(depths)):
            result = native.refine_pose(
                CameraObservation(depths[item], intrinsics[item], matrices[item]),
                environment=int(indices[item].item()), iterations=self.config.max_iterations,
                learning_rate=self.config.learning_rate,
            )
            outputs.append(result.camera_to_world)
            losses.append(result.loss)
            iterations.append(result.iterations)
        matrix = torch.stack(outputs)
        pose = Pose.from_matrix(matrix)
        loss = torch.stack(losses)
        iteration = torch.tensor(iterations, device=target_device, dtype=torch.int64)
        self._last_state = BlockSparseRefinementState(
            pose=pose.detach(), loss=loss.detach(), iterations=int(iteration.max().item()),
            depth=depths.detach().clone(), intrinsics=intrinsics.detach().clone(),
            best_n_valid=valid_count.detach().clone(),
            lambda_damping=loss.new_full((len(loss),), self.config.lambda_initial),
            environment_indices=indices.detach().clone(), map_generation=native.state.generation.detach().clone(),
        )
        if not batched:
            return pose, float(loss[0].detach().cpu()), int(iteration[0].item())
        return pose, loss, iteration

    __call__ = refine_pose


__all__ = ["BlockSparseRefinementState", "BlockSparseRaycastRefinerCfg", "BlockSparseRaycastPoseRefiner"]
