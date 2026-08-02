"""Portable joint-space target-distance cost.

cuRobo V2 evaluates this term with a Warp kernel and keeps two persistent
``[batch, horizon, dof]`` work buffers.  The portable implementation uses
ordinary PyTorch operations instead, preserving the allocation, goal-index,
weighting, and first-order autograd contracts on CPU and Apple Metal.  It
does not expose Warp launch or CUDA graph ABI objects.

Historically the early Metal shim returned a per-DOF value for every direct
call.  That remains available for unallocated convenience calls so existing
portable code is not broken.  Once :meth:`setup_batch_tensors` has established
an actual V2 rollout shape, :meth:`forward` returns the upstream
``[batch, horizon]`` reduced cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .portable import BaseCost
from .wp_torch_cspace_dist import L2DistFunction

if TYPE_CHECKING:
    from .portable import CSpaceDistCostCfg


class CSpaceDistCost(BaseCost):
    """Squared distance from a batched trajectory to indexed joint goals.

    ``goal_vec[idxs_goal[b]]`` is broadcast over the horizon of batch item
    ``b``.  The last horizon element uses ``terminal_dof_weight``; preceding
    elements use ``non_terminal_dof_weight``.  All computations stay on the
    caller's device and use standard PyTorch autograd.
    """

    def __init__(self, config: "CSpaceDistCostCfg"):
        configured_dof = getattr(config, "dof", 0)
        if isinstance(configured_dof, bool) or not isinstance(configured_dof, int) or configured_dof < 0:
            raise ValueError("dof must be a non-negative integer")
        super().__init__(config)
        # A config can be re-used by a cost manager after direct construction.
        # Keep it pointing at this facade rather than the early portable shim.
        config.class_type = type(self)
        self._out_cv_buffer: Optional[torch.Tensor] = None
        self._out_g_buffer: Optional[torch.Tensor] = None

    def setup_batch_tensors(self, batch: int, horizon: int) -> bool:
        if isinstance(batch, bool) or isinstance(horizon, bool) or not isinstance(batch, int) or not isinstance(horizon, int):
            raise TypeError("batch and horizon must be integers")
        if batch < 0 or horizon < 0:
            raise ValueError("batch and horizon must be non-negative")
        super().setup_batch_tensors(batch, horizon)
        dof = int(getattr(self.config, "dof", 0))
        spec = self.device_cfg.as_torch_dict()
        self._out_cv_buffer = torch.zeros((batch, horizon, dof), **spec)
        self._out_g_buffer = torch.zeros((batch, horizon, dof), **spec)
        return True

    def reset(self, reset_problem_ids=None, **kwargs) -> None:
        """Clear persistent diagnostics for all or selected batch problems."""
        del kwargs
        if self._out_cv_buffer is None or self._out_g_buffer is None:
            return None
        if reset_problem_ids is None:
            self._out_cv_buffer.zero_()
            self._out_g_buffer.zero_()
            return None
        ids = torch.as_tensor(reset_problem_ids, device=self._out_cv_buffer.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("reset_problem_ids must be a rank-1 integer tensor")
        if bool(((ids < 0) | (ids >= self._out_cv_buffer.shape[0])).any().item()):
            raise ValueError("reset_problem_ids contains an out-of-range batch index")
        self._out_cv_buffer[ids] = 0
        self._out_g_buffer[ids] = 0
        return None

    def _check_device_and_dtype(self, *values: torch.Tensor) -> None:
        first = values[0]
        if not self.device_cfg.is_same_torch_device(first.device):
            raise ValueError(
                f"current_vec device does not match device_cfg: {first.device} != {self.device_cfg.device}"
            )
        if not torch.is_floating_point(first):
            raise TypeError("current_vec must be floating point")
        if first.device.type == "mps" and first.dtype != torch.float32:
            raise TypeError("MPS CSpaceDistCost supports float32 only")
        for index, value in enumerate(values[1:], start=1):
            if value.device != first.device:
                raise ValueError("current_vec, goal_vec, and idxs_goal must share a device")
            if value.dtype not in (torch.int32, torch.int64) and value.dtype != first.dtype:
                raise TypeError("current_vec and goal_vec must have matching dtypes")

    def validate_input(
        self,
        current_vec: torch.Tensor,
        goal_vec: torch.Tensor,
        idxs_goal: Optional[torch.Tensor] = None,
    ) -> bool:
        """Validate the portable V2 tensor contract without hidden copies."""
        if not isinstance(current_vec, torch.Tensor) or not isinstance(goal_vec, torch.Tensor):
            raise TypeError("current_vec and goal_vec must be tensors")
        if current_vec.ndim not in (2, 3):
            raise ValueError("current_vec must have shape [batch,dof] or [batch,horizon,dof]")
        if goal_vec.ndim not in (2, 3):
            raise ValueError("goal_vec must have shape [goals,dof] or a broadcastable trajectory shape")
        if current_vec.shape[-1] != goal_vec.shape[-1]:
            raise ValueError("current_vec and goal_vec must have matching dof")
        dof = int(getattr(self.config, "dof", 0))
        if dof and current_vec.shape[-1] != dof:
            raise ValueError("current_vec dof does not match configured dof")
        batch = current_vec.shape[0]
        if self._batch_size >= 0 and batch != self._batch_size:
            raise ValueError("current_vec batch size does not match setup_batch_tensors")
        if current_vec.ndim == 3 and self._horizon >= 0 and current_vec.shape[1] != self._horizon:
            raise ValueError("current_vec horizon does not match setup_batch_tensors")
        if idxs_goal is not None:
            if not isinstance(idxs_goal, torch.Tensor) or idxs_goal.ndim != 1 or idxs_goal.shape[0] != batch:
                raise ValueError("idxs_goal must have shape [batch]")
            if idxs_goal.dtype not in (torch.int32, torch.int64):
                raise TypeError("idxs_goal must be an int32 or int64 tensor")
            self._check_device_and_dtype(current_vec, goal_vec, idxs_goal)
            if goal_vec.ndim != 2:
                raise ValueError("indexed goals must have shape [goals,dof]")
            if bool(((idxs_goal < 0) | (idxs_goal >= goal_vec.shape[0])).any().item()):
                raise ValueError("idxs_goal contains an out-of-range goal index")
        else:
            self._check_device_and_dtype(current_vec, goal_vec)
            if goal_vec.ndim == 2 and goal_vec.shape[0] not in (1, batch):
                raise ValueError("unindexed goal_vec must have one goal or one goal per batch")
            if goal_vec.ndim == 3 and not (
                goal_vec.shape[0] in (1, batch)
                and goal_vec.shape[1] in (1, current_vec.shape[1] if current_vec.ndim == 3 else 1)
            ):
                raise ValueError("goal_vec is not broadcastable to current_vec")
        return True

    @staticmethod
    def _indexed_goal(current_vec: torch.Tensor, goal_vec: torch.Tensor, idxs_goal: Optional[torch.Tensor]) -> torch.Tensor:
        if idxs_goal is not None:
            selected = goal_vec.index_select(0, idxs_goal.to(device=goal_vec.device, dtype=torch.long))
            return selected.unsqueeze(1) if current_vec.ndim == 3 else selected
        if current_vec.ndim == 3 and goal_vec.ndim == 2:
            return goal_vec.unsqueeze(1)
        return goal_vec

    def _dof_weight(self, current_vec: torch.Tensor) -> torch.Tensor:
        dof = current_vec.shape[-1]
        terminal = getattr(self.config, "terminal_dof_weight", None)
        if terminal is None:
            terminal = torch.ones((dof,), device=current_vec.device, dtype=current_vec.dtype)
        else:
            terminal = torch.as_tensor(terminal, device=current_vec.device, dtype=current_vec.dtype).reshape(-1)
        if terminal.numel() != dof:
            raise ValueError("terminal_dof_weight must contain one value per dof")
        running = getattr(self.config, "non_terminal_dof_weight", None)
        if running is None:
            running = torch.zeros_like(terminal) if bool(getattr(self.config, "only_terminal_cost", False)) else terminal
        else:
            running = torch.as_tensor(running, device=current_vec.device, dtype=current_vec.dtype).reshape(-1)
        if running.numel() != dof:
            raise ValueError("non_terminal_dof_weight must contain one value per dof")
        if current_vec.ndim == 2:
            return terminal
        result = terminal.reshape(1, 1, dof).expand_as(current_vec).clone()
        if current_vec.shape[1] > 1:
            result[:, :-1, :] = running
        return result

    def _scalar_weight(self, current_vec: torch.Tensor) -> torch.Tensor:
        weight = torch.as_tensor(self._weight, device=current_vec.device, dtype=current_vec.dtype).reshape(-1)
        if weight.numel() != 1:
            raise ValueError("CSpaceDistCost weight must be scalar")
        return weight[0]

    def forward(
        self,
        current_vec: torch.Tensor,
        goal_vec: torch.Tensor,
        idxs_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.validate_input(current_vec, goal_vec, idxs_goal)
        goal = self._indexed_goal(current_vec, goal_vec, idxs_goal)
        dof_weight = self._dof_weight(current_vec)
        component_cost = (current_vec - goal).square() * dof_weight * self._scalar_weight(current_vec)
        component_grad = 2.0 * (current_vec - goal) * dof_weight * self._scalar_weight(current_vec)

        # V2's normal allocation lifecycle writes persistent diagnostics then
        # reduces DOFs.  Preserve legacy component values only for direct
        # convenience calls made before a rollout shape has been allocated.
        exact_rollout = current_vec.ndim == 3 and self._batch_size >= 0
        if exact_rollout:
            # The portable config permits a direct, initially unspecified
            # dof. Resolve it from this first allocated rollout instead of
            # retaining a malformed zero-width diagnostic buffer.
            if int(getattr(self.config, "dof", 0)) == 0:
                self.config.dof = int(current_vec.shape[-1])
                self.setup_batch_tensors(self._batch_size, self._horizon)
            assert self._out_cv_buffer is not None and self._out_g_buffer is not None
            self._out_cv_buffer.copy_(component_cost.detach())
            self._out_g_buffer.copy_(component_grad.detach())
            return component_cost.sum(dim=-1)
        return component_cost

    __call__ = forward

    def forward_out_distance(
        self,
        current_vec: torch.Tensor,
        goal_vec: torch.Tensor,
        idxs_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cost = self.forward(current_vec, goal_vec, idxs_goal)
        return cost, self.jit_squared_cost_to_l2(cost, self._scalar_weight(current_vec), torch.ones_like(cost))

    @staticmethod
    def jit_squared_cost_to_l2(cost, weight=None, run_weight_vec=None) -> torch.Tensor:
        """Convert a weighted squared cost to an L2 distance safely."""
        cost = torch.as_tensor(cost)
        weight_value = torch.ones((), device=cost.device, dtype=cost.dtype) if weight is None else torch.as_tensor(
            weight, device=cost.device, dtype=cost.dtype
        )
        run_value = torch.ones_like(cost) if run_weight_vec is None else torch.as_tensor(
            run_weight_vec, device=cost.device, dtype=cost.dtype
        )
        denominator = weight_value * run_value
        inverse = torch.where(denominator != 0, denominator.reciprocal(), torch.zeros_like(denominator))
        return torch.sqrt((cost * inverse).clamp_min(0))


__all__ = ["BaseCost", "CSpaceDistCost", "L2DistFunction"]
