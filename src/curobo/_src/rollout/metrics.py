"""Portable cost aggregation and rollout result models."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence
import torch
from curobo._src.state.state_joint import JointState


def _sum(values, sum_horizon):
    if not values:
        return None
    normalized = [v if v.ndim >= 3 else v.unsqueeze(-1) for v in values]
    result = torch.cat(normalized, dim=-1).sum(-1)
    return result.sum(1) if sum_horizon and result.ndim > 1 else result


@dataclass
class CostCollection:
    values: List[torch.Tensor] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    weights: List[torch.Tensor] = field(default_factory=list)
    sq_weights: List[torch.Tensor] = field(default_factory=list)

    def add(self, value, name, weight=None, sq_weight=None):
        self.values.append(value); self.names.append(name)
        if weight is not None: self.weights.append(weight)
        if sq_weight is not None: self.sq_weights.append(sq_weight)

    def get_sum(self, sum_horizon=True):
        return _sum(self.values, sum_horizon)

    def is_empty(self): return not self.values
    def clone(self):
        return type(self)([x.clone() for x in self.values], self.names.copy(),
                          [x.clone() for x in self.weights],
                          [x.clone() for x in self.sq_weights])
    def merge(self, other):
        self.values.extend(other.values); self.names.extend(other.names)
        self.weights.extend(other.weights); self.sq_weights.extend(other.sq_weights)
    def copy_at_batch_seed_indices(self, other, batch_idx, seed_idx):
        for a, b in zip(self.values, other.values): a[batch_idx, seed_idx] = b[batch_idx, seed_idx]
        return self
    def copy_only_index(self, other, index):
        for a, b in zip(self.values, other.values): a[index] = b[index]
        return self


@dataclass
class CostsAndConstraints:
    costs: CostCollection = field(default_factory=CostCollection)
    constraints: CostCollection = field(default_factory=CostCollection)
    hybrid_costs_constraints: CostCollection = field(default_factory=CostCollection)
    _grad_out_values: List[torch.Tensor] = field(default_factory=list)

    def _selected(self, collection, include_all, names):
        if include_all: return collection.values
        return [collection.values[collection.names.index(n)] for n in names if n in collection.names]
    def get_sum_cost(self, sum_horizon=False, include_all_hybrid=True, include_from_hybrid=[]):
        return _sum(self.costs.values + self._selected(self.hybrid_costs_constraints,
                    include_all_hybrid, include_from_hybrid), sum_horizon)
    def get_sum_constraint(self, sum_horizon=False, include_all_hybrid=True, include_from_hybrid=[]):
        return _sum(self.constraints.values + self._selected(self.hybrid_costs_constraints,
                    include_all_hybrid, include_from_hybrid), sum_horizon)
    def get_sum_cost_and_constraint(self, sum_horizon=False, include_all_hybrid=True):
        values = self.costs.values + self.constraints.values
        if include_all_hybrid: values += self.hybrid_costs_constraints.values
        return _sum(values, sum_horizon)
    def get_list_costs_and_constraints(self):
        return self.costs.values + self.constraints.values + self.hybrid_costs_constraints.values
    def get_feasible(self, sum_horizon=False, include_all_hybrid=True, include_from_hybrid=[]):
        value = self.get_sum_constraint(sum_horizon, include_all_hybrid, include_from_hybrid)
        return None if value is None else value <= 0
    def clone(self):
        return type(self)(self.costs.clone(), self.constraints.clone(),
                          self.hybrid_costs_constraints.clone())
    def get_constraint_weights(self): return self.constraints.weights
    def copy_at_batch_seed_indices(self, other, batch_idx, seed_idx):
        for name in ("costs", "constraints", "hybrid_costs_constraints"):
            getattr(self, name).copy_at_batch_seed_indices(getattr(other, name), batch_idx, seed_idx)
        return self
    def copy_only_index(self, other, index):
        for name in ("costs", "constraints", "hybrid_costs_constraints"):
            getattr(self, name).copy_only_index(getattr(other, name), index)
        return self


@dataclass
class RolloutResult(Sequence):
    actions: Optional[torch.Tensor] = None
    costs_and_constraints: Optional[CostsAndConstraints] = None
    state: Optional[JointState] = None
    debug: Optional[Any] = None
    def __getitem__(self, idx):
        return type(self)(None if self.actions is None else self.actions[idx],
                          self.costs_and_constraints,
                          None if self.state is None else self.state[idx], self.debug)
    def __len__(self): return 0 if self.actions is None else len(self.actions)
    def clone(self):
        return type(self)(None if self.actions is None else self.actions.clone(),
                          None if self.costs_and_constraints is None else self.costs_and_constraints.clone(),
                          None if self.state is None else self.state.clone(), self.debug)


@dataclass
class RolloutMetrics(RolloutResult):
    feasible: Optional[torch.Tensor | bool] = None
    convergence: CostCollection = field(default_factory=CostCollection)

    def clone(self):
        return type(self)(
            None if self.actions is None else self.actions.clone(),
            None if self.costs_and_constraints is None else self.costs_and_constraints.clone(),
            None if self.state is None else self.state.clone(), self.debug,
            self.feasible.clone() if hasattr(self.feasible, "clone") else self.feasible,
            self.convergence.clone(),
        )

    def get_only_batch_seed_indices(self, batch_idx, seed_idx):
        result = self.clone()
        if result.actions is not None:
            result.actions = result.actions[batch_idx, seed_idx]
        if result.state is not None:
            result.state = result.state[batch_idx, seed_idx]
        if hasattr(result.feasible, "__getitem__"):
            result.feasible = result.feasible[batch_idx, seed_idx]
        return result

    def copy_at_batch_seed_indices(self, other, batch_idx, seed_idx):
        if self.actions is not None and other.actions is not None:
            self.actions[batch_idx, seed_idx] = other.actions[batch_idx, seed_idx]
        if self.state is not None and other.state is not None:
            self.state.position[batch_idx, seed_idx] = other.state.position[batch_idx, seed_idx]
        if hasattr(self.feasible, "__setitem__") and hasattr(other.feasible, "__getitem__"):
            self.feasible[batch_idx, seed_idx] = other.feasible[batch_idx, seed_idx]
        if self.costs_and_constraints is not None and other.costs_and_constraints is not None:
            self.costs_and_constraints.copy_at_batch_seed_indices(other.costs_and_constraints, batch_idx, seed_idx)
        return self

    def copy_only_index(self, other, index):
        if self.actions is not None and other.actions is not None:
            self.actions[index] = other.actions[index]
        if self.state is not None and other.state is not None:
            self.state.position[index] = other.state.position[index]
        if hasattr(self.feasible, "__setitem__") and hasattr(other.feasible, "__getitem__"):
            self.feasible[index] = other.feasible[index]
        return self


__all__ = ["CostCollection", "CostsAndConstraints", "RolloutResult", "RolloutMetrics"]
