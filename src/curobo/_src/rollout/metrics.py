"""Portable rollout cost aggregation and result value objects.

The CUDA implementation stores these values in optimizer-owned buffers.  The
Metal backend keeps the same public, batch/seed-aware value semantics with
ordinary PyTorch tensors, so the objects remain differentiable on both CPU and
MPS without exposing CUDA graph or packed-buffer ABI details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


TensorOrBool = Union[torch.Tensor, bool]


class CostCollectionSum(torch.autograd.Function):
    """Compatibility VJP helper used by a few optimizer internals.

    The first half of the arguments are values and the second half are their
    already-computed VJPs.  Normal portable aggregation uses normal PyTorch
    autograd; this narrow helper deliberately retains V2's explicit-gradient
    contract without any CUDA extension.
    """

    @staticmethod
    def forward(ctx, *values: torch.Tensor) -> torch.Tensor:
        if not values or len(values) % 2:
            raise ValueError("CostCollectionSum expects value/VJP tensor pairs")
        count = len(values) // 2
        ctx.gradients = tuple(values[count:])
        return _sum(list(values[:count]), sum_horizon=True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # V2's caller supplies the exact VJPs, so grad_output is intentionally
        # not multiplied into them.  The trailing VJP inputs are constants.
        return tuple(ctx.gradients) + tuple(None for _ in ctx.gradients)


def _sum(values: Sequence[torch.Tensor], sum_horizon: bool) -> Optional[torch.Tensor]:
    """Sum component axes and, optionally, the final horizon axis.

    Values normally have ``[..., horizon, component]`` shape.  The leading
    dimensions may be either ``[batch]`` or ``[batch, seed]``; reducing the
    *last* dimension after component reduction therefore preserves both
    layouts.  Rank-one and rank-two values are accepted for scalar/legacy
    terms by treating their final axis as a single component.
    """
    if not values:
        return None
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("cost collection values must be torch tensors")
    if any(value.ndim == 0 for value in values):
        raise ValueError("cost collection values must include a batch dimension")
    normalized = [value if value.ndim >= 3 else value.unsqueeze(-1) for value in values]
    result = torch.cat(normalized, dim=-1).sum(dim=-1)
    return result.sum(dim=-1, keepdim=True) if sum_horizon else result


def _clone(value: Any) -> Any:
    """Clone common rollout payloads without retaining mutable debug state."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _to(value: Any, *args: Any, **kwargs: Any) -> Any:
    """Move nested portable rollout payloads while retaining boolean metadata.

    The V2 call sites commonly carry a :class:`DeviceCfg`; ordinary tensors,
    however, should not be cast from bool/int to the configuration's floating
    dtype.  State objects own their own device configuration and receive it
    directly.  This keeps a metrics container internally device-consistent on
    CPU and MPS without recreating CUDA's packed buffer implementation.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _to(item, *args, **kwargs) for key, item in value.items()}
    if isinstance(value, list):
        return [_to(item, *args, **kwargs) for item in value]
    if isinstance(value, tuple):
        return tuple(_to(item, *args, **kwargs) for item in value)
    if isinstance(value, torch.Tensor):
        if args and isinstance(args[0], DeviceCfg):
            if len(args) != 1:
                raise TypeError("DeviceCfg cannot be combined with positional tensor.to arguments")
            cfg = args[0]
            tensor_kwargs = dict(kwargs)
            tensor_kwargs.setdefault("device", cfg.device)
            if value.is_floating_point() or value.is_complex():
                tensor_kwargs.setdefault("dtype", cfg.dtype)
            return value.to(**tensor_kwargs)
        return value.to(*args, **kwargs)
    move = getattr(value, "to", None)
    return move(*args, **kwargs) if callable(move) else value


def _detach(value: Any) -> Any:
    """Detach nested payloads into a new portable value object."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach()
    detach = getattr(value, "detach", None)
    # The compatibility JointState's detach is intentionally in-place.  A
    # container-level detach follows PyTorch value semantics instead.
    return _clone(value).detach() if callable(detach) and hasattr(value, "clone") else value


def _index(value: Any, index: Any) -> Any:
    if value is None:
        return None
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        # Debug metadata and scalar values are intentionally shared exactly as
        # the CUDA value objects do; they are not rollout buffers.
        return value


def _copy_indexed(target: Any, source: Any, batch_idx: Any, seed_idx: Any) -> None:
    """Copy a pair of indexed batch/seed slices through state-aware helpers."""
    if target is None or source is None:
        return
    copy = getattr(target, "copy_at_batch_seed_indices", None)
    if callable(copy):
        copy(source, batch_idx, seed_idx)
    else:
        target[batch_idx, seed_idx] = source[batch_idx, seed_idx]


def _copy_only_index(target: Any, source: Any, index: Any) -> None:
    if target is None or source is None:
        return
    copy = getattr(target, "copy_only_index", None)
    if callable(copy):
        copy(source, index)
    else:
        target[index] = source[index]


@dataclass
class CostCollection:
    values: List[torch.Tensor] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    # Lists are aligned with ``values`` for terms added through ``add``.  The
    # optional element retains compatibility with older manually-built lists.
    weights: List[Optional[torch.Tensor]] = field(default_factory=list)
    sq_weights: List[Optional[torch.Tensor]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.values) != len(self.names):
            raise ValueError("CostCollection values and names must have equal length")
        if len(self.weights) > len(self.values) or len(self.sq_weights) > len(self.values):
            raise ValueError("CostCollection metadata cannot exceed the value count")

    def add(
        self,
        value: torch.Tensor,
        name: str,
        weight: Optional[torch.Tensor] = None,
        sq_weight: Optional[torch.Tensor] = None,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be a torch.Tensor")
        if not isinstance(name, str) or not name:
            raise ValueError("cost name must be a non-empty string")
        # Normalize older manually-built collections before extending them.
        self.weights.extend([None] * (len(self.values) - len(self.weights)))
        self.sq_weights.extend([None] * (len(self.values) - len(self.sq_weights)))
        self.values.append(value)
        self.names.append(name)
        self.weights.append(weight)
        self.sq_weights.append(sq_weight)

    def get_sum(self, sum_horizon: bool = True) -> Optional[torch.Tensor]:
        return _sum(self.values, sum_horizon)

    def is_empty(self) -> bool:
        return not self.values

    def clone(self) -> "CostCollection":
        return type(self)(
            [_clone(value) for value in self.values],
            self.names.copy(),
            [_clone(weight) for weight in self.weights],
            [_clone(weight) for weight in self.sq_weights],
        )

    def to(self, *args: Any, **kwargs: Any) -> "CostCollection":
        """Return a device/dtype converted collection without sharing buffers."""
        return type(self)(
            [_to(value, *args, **kwargs) for value in self.values],
            self.names.copy(),
            [_to(weight, *args, **kwargs) for weight in self.weights],
            [_to(weight, *args, **kwargs) for weight in self.sq_weights],
        )

    def detach(self) -> "CostCollection":
        return type(self)(
            [_detach(value) for value in self.values], self.names.copy(),
            [_detach(weight) for weight in self.weights],
            [_detach(weight) for weight in self.sq_weights],
        )

    def __getitem__(self, index: Any) -> "CostCollection":
        return type(self)(
            [_index(value, index) for value in self.values],
            self.names.copy(),
            [_index(weight, index) for weight in self.weights],
            [_index(weight, index) for weight in self.sq_weights],
        )

    def get_only_batch_seed_indices(self, batch_idx: Any, seed_idx: Any) -> "CostCollection":
        return type(self)(
            [_index(value, (batch_idx, seed_idx)) for value in self.values],
            self.names.copy(),
            [_index(weight, (batch_idx, seed_idx)) for weight in self.weights],
            [_index(weight, batch_idx) for weight in self.sq_weights],
        )

    def merge(self, other: "CostCollection") -> "CostCollection":
        if not isinstance(other, CostCollection):
            raise TypeError("can only merge another CostCollection")
        self.weights.extend([None] * (len(self.values) - len(self.weights)))
        self.sq_weights.extend([None] * (len(self.values) - len(self.sq_weights)))
        other_weights = other.weights + [None] * (len(other.values) - len(other.weights))
        other_sq_weights = other.sq_weights + [None] * (len(other.values) - len(other.sq_weights))
        self.values.extend(other.values)
        self.names.extend(other.names)
        self.weights.extend(other_weights)
        self.sq_weights.extend(other_sq_weights)
        return self

    def copy_at_batch_seed_indices(self, other: "CostCollection", batch_idx: Any, seed_idx: Any):
        if len(self.values) != len(other.values):
            raise ValueError("cannot copy cost collections with different term counts")
        for target, source in zip(self.values, other.values):
            target[batch_idx, seed_idx] = source[batch_idx, seed_idx]
        for target, source in zip(self.weights, other.weights):
            _copy_indexed(target, source, batch_idx, seed_idx)
        # Squared weights are normally seed-independent [batch, ...].
        for target, source in zip(self.sq_weights, other.sq_weights):
            _copy_only_index(target, source, batch_idx)
        return self

    def copy_only_index(self, other: "CostCollection", index: Any):
        if len(self.values) != len(other.values):
            raise ValueError("cannot copy cost collections with different term counts")
        for target, source in zip(self.values, other.values):
            target[index] = source[index]
        for target, source in zip(self.weights, other.weights):
            _copy_only_index(target, source, index)
        for target, source in zip(self.sq_weights, other.sq_weights):
            _copy_only_index(target, source, index)
        return self


@dataclass
class CostsAndConstraints:
    costs: CostCollection = field(default_factory=CostCollection)
    constraints: CostCollection = field(default_factory=CostCollection)
    hybrid_costs_constraints: CostCollection = field(default_factory=CostCollection)
    _grad_out_values: List[torch.Tensor] = field(default_factory=list)

    @staticmethod
    def _selected(collection: CostCollection, include_all: bool, names: Sequence[str]):
        if include_all:
            return collection.values
        return [collection.values[collection.names.index(name)] for name in names if name in collection.names]

    def get_sum_cost(
        self, sum_horizon: bool = False, include_all_hybrid: bool = True,
        include_from_hybrid: Sequence[str] = (),
    ) -> Optional[torch.Tensor]:
        values = self.costs.values + self._selected(
            self.hybrid_costs_constraints, include_all_hybrid, include_from_hybrid
        )
        return _sum(values, sum_horizon)

    def get_sum_constraint(
        self, sum_horizon: bool = False, include_all_hybrid: bool = True,
        include_from_hybrid: Sequence[str] = (),
    ) -> Optional[torch.Tensor]:
        values = self.constraints.values + self._selected(
            self.hybrid_costs_constraints, include_all_hybrid, include_from_hybrid
        )
        return _sum(values, sum_horizon)

    def get_sum_cost_and_constraint(
        self, sum_horizon: bool = False, include_all_hybrid: bool = True,
    ) -> Optional[torch.Tensor]:
        values = self.costs.values + self.constraints.values
        if include_all_hybrid:
            values += self.hybrid_costs_constraints.values
        return _sum(values, sum_horizon)

    def get_list_costs_and_constraints(self) -> List[torch.Tensor]:
        return self.costs.values + self.constraints.values + self.hybrid_costs_constraints.values

    def get_feasible(
        self, sum_horizon: bool = False, include_all_hybrid: bool = True,
        include_from_hybrid: Sequence[str] = (),
    ) -> TensorOrBool:
        value = self.get_sum_constraint(sum_horizon, include_all_hybrid, include_from_hybrid)
        return True if value is None else value <= 0

    def clone(self) -> "CostsAndConstraints":
        return type(self)(
            self.costs.clone(), self.constraints.clone(), self.hybrid_costs_constraints.clone(),
            [_clone(value) for value in self._grad_out_values],
        )

    def to(self, *args: Any, **kwargs: Any) -> "CostsAndConstraints":
        return type(self)(
            self.costs.to(*args, **kwargs),
            self.constraints.to(*args, **kwargs),
            self.hybrid_costs_constraints.to(*args, **kwargs),
            [_to(value, *args, **kwargs) for value in self._grad_out_values],
        )

    def detach(self) -> "CostsAndConstraints":
        return type(self)(
            self.costs.detach(), self.constraints.detach(), self.hybrid_costs_constraints.detach(),
            [_detach(value) for value in self._grad_out_values],
        )

    def __getitem__(self, index: Any) -> "CostsAndConstraints":
        return type(self)(
            self.costs[index], self.constraints[index], self.hybrid_costs_constraints[index],
            [_index(value, index) for value in self._grad_out_values],
        )

    def get_only_batch_seed_indices(self, batch_idx: Any, seed_idx: Any) -> "CostsAndConstraints":
        return type(self)(
            self.costs.get_only_batch_seed_indices(batch_idx, seed_idx),
            self.constraints.get_only_batch_seed_indices(batch_idx, seed_idx),
            self.hybrid_costs_constraints.get_only_batch_seed_indices(batch_idx, seed_idx),
            [_index(value, (batch_idx, seed_idx)) for value in self._grad_out_values],
        )

    def get_constraint_weights(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        weights: List[torch.Tensor] = []
        sq_weights: List[torch.Tensor] = []
        for collection in (self.constraints, self.hybrid_costs_constraints):
            for index, weight in enumerate(collection.weights):
                if weight is not None:
                    weights.append(weight)
                    sq_weight = collection.sq_weights[index] if index < len(collection.sq_weights) else None
                    sq_weights.append(torch.ones_like(weight) if sq_weight is None else sq_weight)
        return weights, sq_weights

    def copy_at_batch_seed_indices(self, other: "CostsAndConstraints", batch_idx: Any, seed_idx: Any):
        for name in ("costs", "constraints", "hybrid_costs_constraints"):
            getattr(self, name).copy_at_batch_seed_indices(getattr(other, name), batch_idx, seed_idx)
        return self

    def copy_only_index(self, other: "CostsAndConstraints", index: Any):
        for name in ("costs", "constraints", "hybrid_costs_constraints"):
            getattr(self, name).copy_only_index(getattr(other, name), index)
        return self


@dataclass
class RolloutResult(Sequence):
    actions: Optional[torch.Tensor] = None
    costs_and_constraints: Optional[CostsAndConstraints] = None
    state: Optional[JointState] = None
    debug: Optional[Any] = None

    def __getitem__(self, index: Any):
        return type(self)(
            _index(self.actions, index), _index(self.costs_and_constraints, index),
            _index(self.state, index), _index(self.debug, index),
        )

    def __len__(self) -> int:
        # Python's ``len`` protocol cannot represent the upstream sentinel
        # ``-1``.  An unpopulated result is a real, empty portable container.
        return 0 if self.actions is None else len(self.actions)

    def clone(self):
        return type(self)(
            _clone(self.actions), _clone(self.costs_and_constraints), _clone(self.state), _clone(self.debug)
        )

    def to(self, *args: Any, **kwargs: Any):
        return type(self)(
            _to(self.actions, *args, **kwargs), _to(self.costs_and_constraints, *args, **kwargs),
            _to(self.state, *args, **kwargs), _to(self.debug, *args, **kwargs),
        )

    def detach(self):
        return type(self)(
            _detach(self.actions), _detach(self.costs_and_constraints), _detach(self.state),
            _detach(self.debug),
        )


@dataclass
class RolloutMetrics(RolloutResult):
    feasible: Optional[TensorOrBool] = None
    convergence: Optional[CostCollection] = field(default_factory=CostCollection)

    def clone(self, **kwargs):
        return type(self)(
            _clone(self.actions), _clone(self.costs_and_constraints), _clone(self.state), _clone(self.debug),
            _clone(self.feasible), _clone(self.convergence),
        )

    def to(self, *args: Any, **kwargs: Any):
        return type(self)(
            _to(self.actions, *args, **kwargs), _to(self.costs_and_constraints, *args, **kwargs),
            _to(self.state, *args, **kwargs), _to(self.debug, *args, **kwargs),
            _to(self.feasible, *args, **kwargs), _to(self.convergence, *args, **kwargs),
        )

    def detach(self):
        return type(self)(
            _detach(self.actions), _detach(self.costs_and_constraints), _detach(self.state),
            _detach(self.debug), _detach(self.feasible), _detach(self.convergence),
        )

    def __getitem__(self, index: Any):
        return type(self)(
            _index(self.actions, index), _index(self.costs_and_constraints, index),
            _index(self.state, index), _index(self.debug, index), _index(self.feasible, index),
            _index(self.convergence, index),
        )

    def get_only_batch_seed_indices(self, batch_idx: Any, seed_idx: Any):
        cc = None if self.costs_and_constraints is None else self.costs_and_constraints.get_only_batch_seed_indices(batch_idx, seed_idx)
        convergence = None if self.convergence is None else self.convergence.get_only_batch_seed_indices(batch_idx, seed_idx)
        return type(self)(
            _index(self.actions, (batch_idx, seed_idx)), cc, _index(self.state, (batch_idx, seed_idx)),
            self.debug, _index(self.feasible, (batch_idx, seed_idx)), convergence,
        )

    def copy_at_batch_seed_indices(self, other: "RolloutMetrics", batch_idx: Any, seed_idx: Any):
        _copy_indexed(self.actions, other.actions, batch_idx, seed_idx)
        _copy_indexed(self.state, other.state, batch_idx, seed_idx)
        _copy_indexed(self.feasible, other.feasible, batch_idx, seed_idx)
        if self.convergence is not None and other.convergence is not None:
            self.convergence.copy_at_batch_seed_indices(other.convergence, batch_idx, seed_idx)
        if self.costs_and_constraints is not None and other.costs_and_constraints is not None:
            self.costs_and_constraints.copy_at_batch_seed_indices(other.costs_and_constraints, batch_idx, seed_idx)
        return self

    def copy_only_index(self, other: "RolloutMetrics", index: Any):
        _copy_only_index(self.actions, other.actions, index)
        _copy_only_index(self.state, other.state, index)
        _copy_only_index(self.feasible, other.feasible, index)
        if self.convergence is not None and other.convergence is not None:
            self.convergence.copy_only_index(other.convergence, index)
        if self.costs_and_constraints is not None and other.costs_and_constraints is not None:
            self.costs_and_constraints.copy_only_index(other.costs_and_constraints, index)
        return self


__all__ = ["CostCollectionSum", "CostCollection", "CostsAndConstraints", "RolloutResult", "RolloutMetrics"]
