"""Portable pinned cuRobo V2 self-collision trajectory cost.

The upstream class launches a Warp/CUDA reduction over configured sphere
pairs.  This implementation retains the useful public lifecycle (persistent
buffers, pair diagnostics, batch reset, and the squared-overlap objective)
using regular PyTorch operations and the production CPU/MPS sphere-pair
operator.  It deliberately does not expose the CUDA kernel ABI or CUDA graph
capture objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from curobo._src.curobolib.cuda_ops.geometry import SelfCollisionDistance
from curobo._src.util.logging import log_and_raise
from curobo_metal.ops.collision import sphere_sphere_signed_distance

from .portable import BaseCost

if TYPE_CHECKING:
    from .cost_self_collision_cfg import SelfCollisionCostCfg


class _SelfCollisionCostPortableMixin:
    def get_gradient_buffer(self) -> Optional[torch.Tensor]:
        return self._out_grad


class SelfCollisionCost(_SelfCollisionCostPortableMixin, BaseCost):
    """Evaluate the maximum configured sphere-pair squared overlap.

    For a pair ``(i, j)`` the V2 CUDA kernel evaluates
    ``(r_i + p_i + r_j + p_j)^2 - ||x_i - x_j||^2`` and returns half the
    configured weight times the largest positive value.  Pair order resolves
    exact ties, which makes diagnostics deterministic on CPU and Metal.
    """

    def __init__(self, config: SelfCollisionCostCfg):
        if getattr(config, "self_collision_kin_config", None) is None:
            raise ValueError("SelfCollisionCostCfg must contain self_collision_kin_config")
        super().__init__(config)
        config.class_type = type(self)
        self._out_distance: Optional[torch.Tensor] = None
        self._out_grad: Optional[torch.Tensor] = None
        self._sparse_sphere_idx: Optional[torch.Tensor] = None
        self._pair_distance: Optional[torch.Tensor] = None
        self._block_batch_max_index: Optional[torch.Tensor] = None
        self._block_batch_max_value: Optional[torch.Tensor] = None

    @property
    def _kinematics_config(self) -> Any:
        return self.config.self_collision_kin_config

    def _pairs(self, *, device: torch.device) -> torch.Tensor:
        pairs = getattr(self._kinematics_config, "collision_pairs", None)
        if pairs is None:
            count = int(getattr(self._kinematics_config, "num_spheres", 0))
            return torch.triu_indices(count, count, offset=1, device=device).transpose(0, 1)
        result = torch.as_tensor(pairs, device=device, dtype=torch.int64)
        if result.ndim != 2 or result.shape[-1] != 2:
            raise ValueError("collision_pairs must have shape [pairs,2]")
        return result

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        if isinstance(batch_size, bool) or isinstance(horizon, bool):
            raise TypeError("batch_size and horizon must be integers")
        if not isinstance(batch_size, int) or not isinstance(horizon, int):
            raise TypeError("batch_size and horizon must be integers")
        if batch_size < 0 or horizon < 0:
            raise ValueError("batch_size and horizon must be non-negative")
        count = int(getattr(self._kinematics_config, "num_spheres", 0))
        if count < 0:
            raise ValueError("self_collision_kin_config.num_spheres must be non-negative")
        pair_count = int(self._pairs(device=self.device_cfg.device).shape[0])
        same_shape = (
            self._batch_size == batch_size
            and self._horizon == horizon
            and self._out_distance is not None
            and self._out_grad is not None
            and self._out_grad.shape[-2] == count
            and self._pair_distance is not None
            and (
                (self.config.store_pair_distance and self._pair_distance.shape == (batch_size, horizon, pair_count))
                or (not self.config.store_pair_distance and self._pair_distance.shape == (1,))
            )
        )
        if same_shape:
            return True
        super().setup_batch_tensors(batch_size, horizon)
        spec = self.device_cfg.as_torch_dict()
        self._out_distance = torch.zeros((batch_size, horizon, 1), **spec)
        self._out_grad = torch.zeros((batch_size, horizon, count, 4), **spec)
        self._sparse_sphere_idx = torch.zeros(
            (batch_size, horizon, count), device=self.device_cfg.device, dtype=torch.uint8
        )
        self._pair_distance = torch.zeros(
            (batch_size, horizon, pair_count) if self.config.store_pair_distance else (1,), **spec
        )
        block_count = int(getattr(self._kinematics_config, "num_blocks_per_batch", 0))
        self._block_batch_max_value = torch.zeros((batch_size, horizon, block_count), **spec)
        self._block_batch_max_index = torch.zeros(
            (batch_size, horizon, block_count, 2),
            device=self.device_cfg.device,
            dtype=torch.int16,
        )
        return True

    def validate_input(self, robot_spheres: torch.Tensor):
        if not isinstance(robot_spheres, torch.Tensor) or robot_spheres.ndim != 4:
            raise ValueError("robot_spheres must have shape [batch,horizon,spheres,4]")
        if robot_spheres.shape[-1] != 4:
            raise ValueError("robot_spheres must contain xyzw-radius values")
        if not self.device_cfg.is_same_torch_device(robot_spheres.device):
            raise ValueError("robot_spheres must be on the configured device")
        if robot_spheres.dtype != self.device_cfg.dtype:
            raise TypeError("robot_spheres dtype must match device_cfg.dtype")
        if robot_spheres.device.type == "mps" and robot_spheres.dtype != torch.float32:
            raise TypeError("MPS self collision supports float32 only")
        expected = int(getattr(self._kinematics_config, "num_spheres", 0))
        if robot_spheres.shape[-2] != expected:
            raise ValueError("sphere count does not match self collision kinematics config")
        if not bool(torch.isfinite(robot_spheres).all().item()):
            raise ValueError("robot_spheres must contain only finite values")
        if bool((robot_spheres[..., 3] < 0).any().item()):
            raise ValueError("robot sphere radii must be non-negative")
        if self._batch_size >= 0 and robot_spheres.shape[:2] != (self._batch_size, self._horizon):
            raise ValueError("robot_spheres batch and horizon must match setup_batch_tensors")
        pairs = self._pairs(device=robot_spheres.device)
        if bool(((pairs < 0) | (pairs >= expected)).any().item()):
            raise ValueError("collision_pairs contains an out-of-range sphere")
        if bool((pairs[:, 0] == pairs[:, 1]).any().item()):
            raise ValueError("collision_pairs cannot contain a sphere paired with itself")
        return True

    def _padding(self, spheres: torch.Tensor) -> torch.Tensor:
        value = getattr(self._kinematics_config, "sphere_padding", None)
        if value is None:
            return spheres.new_zeros((spheres.shape[-2],))
        padding = torch.as_tensor(value, device=spheres.device, dtype=spheres.dtype).reshape(-1)
        if padding.numel() == 1:
            padding = padding.expand(spheres.shape[-2])
        if padding.numel() != spheres.shape[-2]:
            raise ValueError("sphere_padding must be scalar or contain one value per sphere")
        if not bool(torch.isfinite(padding).all().item()) or bool((padding < 0).any().item()):
            raise ValueError("sphere_padding must be finite and non-negative")
        return padding

    def _copy_diagnostics(
        self,
        overlap: torch.Tensor,
        winners: torch.Tensor,
        pairs: torch.Tensor,
        spheres: torch.Tensor,
        effective_radius: torch.Tensor,
    ) -> None:
        assert self._out_distance is not None and self._out_grad is not None
        assert self._sparse_sphere_idx is not None and self._pair_distance is not None
        positive = overlap.clamp_min(0)
        if overlap.shape[-1] == 0:
            reduced = overlap.new_zeros(overlap.shape[:-1])
        else:
            reduced = positive.gather(-1, winners.unsqueeze(-1)).squeeze(-1)
        self._out_distance.copy_(
            (0.5 * self._weight.reshape(-1)[0].to(spheres) * reduced).unsqueeze(-1).detach()
        )
        self._out_grad.zero_()
        self._sparse_sphere_idx.zero_()
        if self.config.store_pair_distance:
            self._pair_distance.copy_(overlap.detach())
        if pairs.numel() == 0:
            return
        selected = pairs[winners]
        first, second = selected[..., 0], selected[..., 1]
        delta = spheres[..., :3].gather(
            -2, first[..., None, None].expand(*first.shape, 1, 3)
        ).squeeze(-2) - spheres[..., :3].gather(
            -2, second[..., None, None].expand(*second.shape, 1, 3)
        ).squeeze(-2)
        weight = self._weight.reshape(-1)[0].to(spheres)
        chosen_radius = effective_radius.gather(-1, winners.unsqueeze(-1)).squeeze(-1)
        active = (reduced > 0).unsqueeze(-1)
        grad_first = torch.cat((-weight * delta, weight * chosen_radius.unsqueeze(-1)), dim=-1)
        grad_second = torch.cat((weight * delta, weight * chosen_radius.unsqueeze(-1)), dim=-1)
        grad_first = torch.where(active, grad_first, torch.zeros_like(grad_first))
        grad_second = torch.where(active, grad_second, torch.zeros_like(grad_second))
        flat_grad = self._out_grad.reshape(-1, self._out_grad.shape[-2], 4)
        flat_first, flat_second = first.reshape(-1), second.reshape(-1)
        flat_grad[torch.arange(flat_grad.shape[0], device=spheres.device), flat_first] = grad_first.reshape(-1, 4)
        flat_grad[torch.arange(flat_grad.shape[0], device=spheres.device), flat_second] = grad_second.reshape(-1, 4)
        flat_sparse = self._sparse_sphere_idx.reshape(-1, self._sparse_sphere_idx.shape[-1])
        active_flat = (reduced > 0).reshape(-1).to(torch.uint8)
        flat_sparse[torch.arange(flat_sparse.shape[0], device=spheres.device), flat_first] = active_flat
        flat_sparse[torch.arange(flat_sparse.shape[0], device=spheres.device), flat_second] = active_flat

    def forward(self, robot_spheres: torch.Tensor):
        # Upstream requires explicit setup.  Lazy setup preserves that layout
        # while keeping the Python façade ergonomic for standalone users.
        if self._batch_size < 0:
            self.setup_batch_tensors(robot_spheres.shape[0], robot_spheres.shape[1])
        self.validate_input(robot_spheres)
        pairs = self._pairs(device=robot_spheres.device)
        batch, horizon = robot_spheres.shape[:2]
        if pairs.numel() == 0:
            empty = robot_spheres.new_zeros((batch, horizon, 0))
            winners = torch.empty((batch, horizon), device=robot_spheres.device, dtype=torch.long)
            self._copy_diagnostics(empty, winners, pairs, robot_spheres, empty)
            # Preserve a zero VJP for an empty topology.  A detached
            # ``new_zeros`` result makes a perfectly valid loss impossible to
            # backpropagate through when a robot has no enabled self pairs.
            return robot_spheres[..., 0].sum(dim=-1, keepdim=True) * 0

        flat = robot_spheres.reshape(batch * horizon, robot_spheres.shape[-2], 4)
        # This call supplies the shared input and pair validation, first-tie
        # convention, and fused MPS autograd path.  We reconstruct the exact
        # V2 squared-overlap objective from its signed clearance.
        signed = sphere_sphere_signed_distance(flat, pairs).distances
        radius_sum = flat[:, pairs[:, 0], 3] + flat[:, pairs[:, 1], 3]
        length = signed + radius_sum
        padding = self._padding(robot_spheres)
        effective_radius = radius_sum + padding[pairs[:, 0]] + padding[pairs[:, 1]]
        overlap = effective_radius.square() - length.square()
        overlap = overlap.reshape(batch, horizon, -1)
        effective_radius = effective_radius.reshape(batch, horizon, -1)
        positive = overlap.clamp_min(0)
        winners = positive.argmax(dim=-1)
        reduced = positive.gather(-1, winners.unsqueeze(-1))
        self._copy_diagnostics(overlap, winners, pairs, robot_spheres, effective_radius)
        result = 0.5 * self._weight.reshape(-1)[0].to(robot_spheres) * reduced
        if self.config.convert_to_binary:
            result = torch.where(result > 0, result.clamp_max(1) + 1, result)
        return result if self.enabled else result * 0

    __call__ = forward

    def reset(self, reset_problem_ids: Optional[torch.Tensor] = None, **kwargs):
        del kwargs
        buffers = (
            self._out_distance,
            self._out_grad,
            self._sparse_sphere_idx,
            self._pair_distance,
            self._block_batch_max_index,
            self._block_batch_max_value,
        )
        if reset_problem_ids is None:
            for buffer in buffers:
                if buffer is not None:
                    buffer.zero_()
            return None
        if self._out_distance is None:
            return None
        indices = torch.as_tensor(reset_problem_ids, device=self._out_distance.device)
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("reset_problem_ids must be a rank-1 integer tensor")
        if bool(((indices < 0) | (indices >= self._out_distance.shape[0])).any().item()):
            raise ValueError("reset_problem_ids contains an out-of-range batch index")
        for buffer in buffers:
            if buffer is not None and buffer.ndim >= 3:
                buffer[indices] = 0
        return None


def __getattr__(name: str):
    """Retain the early portable module's convenient config re-export.

    Pinned V2 imports the configuration from ``cost_self_collision_cfg``.
    Resolving the old local convenience lazily avoids that module's import
    cycle while keeping existing portable callers source-compatible.
    """
    if name == "SelfCollisionCostCfg":
        from .cost_self_collision_cfg import SelfCollisionCostCfg

        return SelfCollisionCostCfg
    raise AttributeError(name)


__all__ = ["BaseCost", "SelfCollisionCost", "SelfCollisionCostCfg", "SelfCollisionDistance"]
