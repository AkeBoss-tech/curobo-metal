"""Portable result lifecycle for the pinned trajectory optimiser API.

The CUDA implementation owns packed rollout buffers and CUDA-graph state.  A
result object should not expose either implementation detail: callers need a
stable, regular PyTorch value object that can select seeds, copy a successful
candidate into an existing result, and retain the timing/state metadata on CPU
or MPS.  This module deliberately implements that public lifecycle with normal
tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import (
    copy_joint_state_at_batch_seed_indices,
    gather_joint_state_by_seed,
)


def _clone_value(value: Any) -> Any:
    """Clone value-like result data without aliasing tensor/state payloads."""
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


@dataclass
class TrajOptSolverResult(BaseSolverResult):
    """Trajectory optimisation output with portable batch/seed operations.

    Tensor data follows cuRobo's public ``[batch, seed, ...]`` convention.  The
    methods below preserve that convention on both CPU and MPS and leave raw
    CUDA graph, stream, and packed result-buffer APIs deliberately outside the
    Metal compatibility surface.
    """

    solution: Optional[torch.Tensor] = None
    js_solution: Optional[JointState] = None
    interpolated_trajectory: Optional[JointState] = None
    interpolated_last_tstep: Optional[torch.Tensor] = None
    interpolated_metrics: Optional[object] = None
    maximum_trajectory_dt: Optional[torch.Tensor] = None
    minimum_trajectory_dt: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        """Device owning the result's primary tensor payload."""
        for value in (self.solution, self.success, self.seed_cost):
            if isinstance(value, torch.Tensor):
                return value.device
        if self.js_solution is not None:
            return self.js_solution.position.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for value in (self.solution, self.seed_cost, self.position_error, self.success):
            if isinstance(value, torch.Tensor):
                return value.dtype
        return torch.get_default_dtype()

    def motion_time(self) -> torch.Tensor:
        """Return trajectory duration for each returned batch/seed plan.

        The pinned public API spells this as a method.  ``dt`` can be scalar,
        per batch, or per batch/seed; ordinary broadcasting keeps all forms
        device-resident and differentiable.
        """
        if self.js_solution is None:
            raise ValueError("js_solution is not set")
        # The V2 result derives elapsed time from the materialised rollout
        # state.  ``maximum_trajectory_dt`` is a constraint/diagnostic, not
        # necessarily the dt selected by an optimiser.  It is still a useful
        # fallback for lightweight callers that only materialise a solution.
        dt = self.js_solution.dt
        if dt is None:
            dt = self.maximum_trajectory_dt
        if dt is None:
            raise ValueError("trajectory dt is not set")
        horizon = self.js_solution.position.shape[-2]
        return dt * (horizon - 1)

    def clone(self) -> "TrajOptSolverResult":
        """Deep-clone materialised portable result state on its current device."""
        return type(self)(**{item.name: _clone_value(getattr(self, item.name)) for item in fields(self)})

    def get_interpolated_plan(self) -> Optional[JointState]:
        """Return an interpolated plan trimmed to its recorded final timestep."""
        trajectory = self.interpolated_trajectory
        if trajectory is None or self.interpolated_last_tstep is None:
            return trajectory
        final_step = self.interpolated_last_tstep
        # A single returned plan is the upstream supported trimming form.  For
        # a batch with equal lengths, the same trim is well-defined and useful
        # to portable callers; ragged batches remain explicit rather than
        # silently dropping different final states.
        flat = final_step.reshape(-1)
        if flat.numel() == 0:
            return trajectory
        if not torch.equal(flat, flat[0].expand_as(flat)):
            raise ValueError("ragged interpolated trajectories cannot be represented as one JointState")
        end = int(flat[0].item())
        trimmed = trajectory.clone()
        # ``dt`` is commonly [batch, seed], not a time-series tensor, and so
        # must remain intact while the kinematic channels are shortened.
        for name in ("position", "velocity", "acceleration", "jerk", "knot"):
            value = getattr(trimmed, name, None)
            if value is not None and value.ndim >= 2:
                setattr(trimmed, name, value[..., :end, :])
        return trimmed

    @staticmethod
    def _copy_tensor_at_mask(target: Optional[torch.Tensor], source: Optional[torch.Tensor], mask: torch.Tensor) -> None:
        if target is None and source is None:
            return
        if target is None or source is None:
            raise ValueError("both trajectory result fields must be set or both must be None")
        if target.device != source.device:
            raise ValueError("result tensors must share a device")
        if target.shape != source.shape:
            raise ValueError(f"result tensor shapes differ: {tuple(target.shape)} != {tuple(source.shape)}")
        target[mask] = source[mask]

    def copy_at_batch_indices(self, other: "TrajOptSolverResult", mask: torch.Tensor) -> None:
        """Copy complete results for selected batch entries.

        ``mask`` indexes the leading batch dimension; seed dimensions are
        retained.  This is the composition operation used by graph/trajectory
        retry logic, distinct from :meth:`copy_successful_solutions` which
        selects individual seed candidates.
        """
        if not isinstance(other, TrajOptSolverResult):
            raise TypeError("other must be a TrajOptSolverResult")
        # BaseSolverResult owns every inherited materialised field, including
        # the normal rollout JointState.  Do not duplicate that logic here:
        # doing so previously made subclass behaviour silently diverge from
        # IK/MPC whenever optional fields were asymmetric.
        super().copy_at_batch_indices(other, mask)
        self._copy_interpolated_at_batch_indices(other, mask)

    def _copy_interpolated_at_batch_indices(
        self, other: "TrajOptSolverResult", mask: torch.Tensor
    ) -> None:
        """Copy TrajOpt-only materialised payloads at complete batch rows."""
        self._copy_tensor_at_mask(
            self.interpolated_last_tstep, other.interpolated_last_tstep, mask
        )
        left, right = self.interpolated_trajectory, other.interpolated_trajectory
        if left is None and right is None:
            pass
        elif left is None or right is None:
            raise ValueError("both interpolated trajectories must be set or both must be None")
        else:
            for field_name in left._tensor_fields():
                self._copy_tensor_at_mask(
                    getattr(left, field_name), getattr(right, field_name), mask
                )
        target_metrics, source_metrics = self.interpolated_metrics, other.interpolated_metrics
        if target_metrics is None and source_metrics is None:
            return
        if target_metrics is None or source_metrics is None:
            raise ValueError("both interpolated_metrics fields must be set or both must be None")
        copy = getattr(target_metrics, "copy_only_index", None)
        if not callable(copy):
            raise NotImplementedError(
                "interpolated metric values must provide copy_only_index for portable batch merging"
            )
        copy(source_metrics, mask)

    @staticmethod
    def _gather_seed_tensor(value: Optional[torch.Tensor], indices: torch.Tensor, seed_shape: tuple[int, int]) -> Optional[torch.Tensor]:
        if value is None or value.ndim < 2 or tuple(value.shape[:2]) != seed_shape:
            return value
        expanded = indices.reshape(indices.shape + (1,) * (value.ndim - 2))
        return value.gather(1, expanded.expand(indices.shape + value.shape[2:]))

    def _seed_indices(self, topk: int) -> torch.Tensor:
        costs = self.total_cost_reshaped if self.total_cost_reshaped is not None else self.seed_cost
        if costs is None or costs.ndim != 2:
            raise ValueError("seed_cost or total_cost_reshaped [batch, seed] is required")
        if topk < 1 or topk > costs.shape[1]:
            raise ValueError("topk must be between 1 and the available seed count")
        if self.seed_rank is not None:
            if self.seed_rank.shape != costs.shape:
                raise ValueError("seed_rank must have the same [batch, seed] shape as seed cost")
            return self.seed_rank[:, :topk]
        # Stable sorting gives deterministic first-seed tie resolution across
        # CPU and MPS, unlike a backend-specific topk tie order.
        return torch.argsort(costs, dim=1, stable=True)[:, :topk]

    def get_topk_seeds(self, topk: int) -> "TrajOptSolverResult":
        """Select the best ``topk`` seeds while retaining all result channels."""
        indices = self._seed_indices(topk)
        # ``num_seeds`` can be omitted by callers constructing a bare result;
        # derive it from the cost tensor in that case.
        original_count = self.seed_cost.shape[1] if self.seed_cost is not None else self.total_cost_reshaped.shape[1]
        # Preserve V2's identity fast path.  Besides avoiding needless copies,
        # this retains metric views when callers request every available seed.
        if topk == original_count:
            return self
        seed_shape = (indices.shape[0], original_count)
        result = self.clone()
        for item in fields(result):
            value = getattr(result, item.name)
            if isinstance(value, torch.Tensor):
                setattr(result, item.name, self._gather_seed_tensor(value, indices, seed_shape))
        for name in ("js_solution", "interpolated_trajectory"):
            value = getattr(self, name)
            if value is not None and value.position.ndim >= 2 and tuple(value.position.shape[:2]) == seed_shape:
                setattr(result, name, gather_joint_state_by_seed(value, indices))
        # Metrics describe the original flattened rollout buffers and cannot
        # safely be selected without their CUDA/rollout allocator.  Seed-level
        # tensors above remain complete, so clear these stale views explicitly.
        result.metrics = None
        result.interpolated_metrics = None
        result.batch_size = indices.shape[0]
        result.num_seeds = topk
        return result

    @staticmethod
    def _jit_compute_rank(
        trajectory_dt: torch.Tensor,
        acceleration: Optional[torch.Tensor],
        jerk: Optional[torch.Tensor],
        seed_rollout_cost: torch.Tensor,
        success: torch.Tensor,
        batch_size: int,
        num_seeds: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank costs with portable smoothness and feasibility penalties."""
        if seed_rollout_cost.numel() != batch_size * num_seeds:
            raise ValueError("seed_rollout_cost does not match batch_size * num_seeds")
        cost = seed_rollout_cost.reshape(batch_size, num_seeds).clone()
        dt_cost = trajectory_dt.reshape(batch_size, num_seeds).to(cost.dtype)
        smooth = dt_cost * 1000.0
        for weight, value in ((0.01, acceleration), (0.001, jerk)):
            if value is None:
                continue
            # Average over the usable interior.  Short horizons use every
            # timestep rather than creating an empty reduction/NaN.
            horizon = value.shape[-2]
            interior = value[..., 8:-8, :] if horizon > 16 else value
            smooth = smooth + weight * interior.abs().mean(dim=(-1, -2)).reshape(batch_size, num_seeds)
        cost = cost + smooth
        cost = torch.where(success.reshape(batch_size, num_seeds), cost, cost + 1.0e16)
        return cost, torch.argsort(cost, dim=1, stable=True)

    def process_metrics_and_rank_seeds(self) -> None:
        """Apply feasibility and deterministically rank available seed costs."""
        if self.feasible is not None:
            self.success = self.success & self.feasible.to(dtype=torch.bool)
        if self.seed_cost is None:
            return
        if self.seed_cost.ndim != 2:
            raise ValueError("seed_cost must have shape [batch, seed]")
        self.batch_size, self.num_seeds = self.seed_cost.shape
        ranked_cost = self.seed_cost.clone()
        # A failed seed must never beat a successful candidate just because
        # its raw rollout error happened to be smaller.  Some portable solve
        # paths already return only their selected seeds; preserve those
        # mismatched payloads rather than inventing a mapping to all seeds.
        if self.success.shape == ranked_cost.shape:
            ranked_cost = torch.where(self.success, ranked_cost, ranked_cost + 1.0e16)
        self.seed_rank = torch.argsort(ranked_cost, dim=1, stable=True)
        self.total_cost_reshaped = ranked_cost

    def copy_successful_solutions(self, other: "TrajOptSolverResult") -> None:
        """Copy successful individual batch/seed candidates from ``other``."""
        if not isinstance(other, TrajOptSolverResult):
            raise TypeError("other must be a TrajOptSolverResult")
        if self.success.ndim != 2 or other.success.ndim != 2:
            # A non-seeded TrajOpt result has the same batch-level contract as
            # BaseSolverResult; leave its strict validation and merge there.
            super().copy_successful_solutions(other)
            self._copy_interpolated_at_batch_indices(other, other.success)
            return
        super().copy_successful_solutions(other)
        if self.success.shape != other.success.shape:
            # ``super`` normally reports this before we reach here, but retain
            # a direct guarantee for static type users calling this override.
            raise ValueError("success tensors must share shape")
        batch_idx, seed_idx = other.success.nonzero(as_tuple=True)
        left, right = self.interpolated_trajectory, other.interpolated_trajectory
        if left is None and right is None:
            pass
        elif left is None or right is None:
            raise ValueError("both interpolated trajectories must be set or both must be None")
        else:
            copy_joint_state_at_batch_seed_indices(left, right, batch_idx, seed_idx)
        target, source = self.interpolated_last_tstep, other.interpolated_last_tstep
        if target is None and source is None:
            pass
        elif target is None or source is None:
            raise ValueError("both interpolated_last_tstep fields must be set or both must be None")
        else:
            if target.shape != source.shape or target.device != source.device:
                raise ValueError("interpolated_last_tstep tensors must share shape and device")
            target[batch_idx, seed_idx] = source[batch_idx, seed_idx]
        target_metrics, source_metrics = self.interpolated_metrics, other.interpolated_metrics
        if target_metrics is None and source_metrics is None:
            return
        if target_metrics is None or source_metrics is None:
            raise ValueError("both interpolated_metrics fields must be set or both must be None")
        copy = getattr(target_metrics, "copy_at_batch_seed_indices", None)
        if not callable(copy):
            raise NotImplementedError(
                "interpolated metric values must provide copy_at_batch_seed_indices for portable seed merging"
            )
        copy(source_metrics, batch_idx, seed_idx)


__all__ = ["TrajOptSolverResult"]
