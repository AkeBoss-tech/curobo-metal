"""Differentiable CPU/MPS Rosenbrock rollout with the cuRobo V2 interface."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg

@dataclass
class RosenbrockCfg:
    device_cfg: DeviceCfg
    a: float = 1.0
    b: float = 100.0
    dimensions: int = 2
    time_horizon: int = 1
    time_action_horizon: int = 1
    sum_horizon: bool = False
    sampler_seed: int = 1312
    @classmethod
    def create(cls, config_dict: Dict, device_cfg: DeviceCfg = DeviceCfg()):
        return cls(device_cfg=device_cfg, **{
            k: config_dict.get(k, f.default)
            for k, f in cls.__dataclass_fields__.items() if k != "device_cfg"
        })

class RosenbrockRollout:
    def __init__(self, config: Optional[RosenbrockCfg] = None, use_cuda_graph: bool = False):
        config = config or RosenbrockCfg(DeviceCfg())
        self.config = config
        self.a, self.b, self.dimensions = config.a, config.b, config.dimensions
        self.time_horizon, self.time_action_horizon = config.time_horizon, config.time_action_horizon
        self.device_cfg, self.sum_horizon, self.sampler_seed = config.device_cfg, config.sum_horizon, config.sampler_seed
        self._batch_size = 1
        self._generator = torch.Generator(device="cpu").manual_seed(config.sampler_seed)
        self._action_bound_lows = torch.full((self.action_horizon,), -1.5, **self.device_cfg.as_torch_dict())
        self._action_bound_highs = torch.full((self.action_horizon,), 2.0, **self.device_cfg.as_torch_dict())
    action_dim = property(lambda self: self.dimensions)
    action_bound_lows = property(lambda self: self._action_bound_lows)
    action_bound_highs = property(lambda self: self._action_bound_highs)
    action_bounds = property(lambda self: torch.stack((self._action_bound_lows, self._action_bound_highs)))
    horizon = property(lambda self: self.time_horizon)
    action_horizon = property(lambda self: self.time_action_horizon)
    state_bounds = property(lambda self: {"position": [-2.048, 2.048]})
    batch_size = property(lambda self: self._batch_size, lambda self, v: setattr(self, "_batch_size", v))
    dt = property(lambda self: 1.0)
    def _cost(self, value):
        return ((self.a - value[..., :-1]).square()
                + self.b * (value[..., 1:] - value[..., :-1].square()).square()).sum(-1, keepdim=True)
    def _compute_state_from_action_impl(self, act_seq):
        return JointState(position=act_seq, device_cfg=self.device_cfg)
    def _collections(self, state):
        costs = CostCollection(); costs.add(self._cost(state.position), "rosenbrock")
        return CostsAndConstraints(costs=costs)
    def evaluate_action(self, act_seq, **kwargs):
        self.update_batch_size(act_seq.shape[0])
        state = self._compute_state_from_action_impl(act_seq)
        return RolloutResult(act_seq, self._collections(state), state)
    def compute_metrics_from_state(self, state, **kwargs):
        cc = self._collections(state)
        return RolloutMetrics(state=state, costs_and_constraints=cc,
                              feasible=torch.ones(state.position.shape[:-1], device=state.position.device, dtype=torch.bool))
    def compute_metrics_from_action(self, act_seq, **kwargs):
        state = self._compute_state_from_action_impl(act_seq)
        result = self.compute_metrics_from_state(state, **kwargs); result.actions = act_seq
        return result
    def update_params(self, a=None, b=None, **kwargs):
        if a is not None: self.a = a
        if b is not None: self.b = b
        return True
    def update_batch_size(self, batch_size): self._batch_size = batch_size
    def update_dt(self, dt, **kwargs): return True
    def reset(self, reset_problem_ids=None, **kwargs): return True
    def reset_shape(self): return True
    def reset_cuda_graph(self): return True
    def reset_seed(self): self._generator.manual_seed(self.sampler_seed)
    def sample_random_actions(self, n=0, bounded=True):
        n = n or self.batch_size
        cpu = torch.rand((n, self.action_horizon, self.action_dim), generator=self._generator)
        value = cpu.to(**self.device_cfg.as_torch_dict())
        return self.action_bound_lows.view(1, -1, 1) + value * (
            self.action_bound_highs - self.action_bound_lows).view(1, -1, 1)
    def get_initial_action(self, use_random=True, use_zero=False, **kwargs):
        if use_zero: return torch.zeros((self.batch_size, self.action_horizon, self.action_dim), **self.device_cfg.as_torch_dict())
        return self.sample_random_actions(self.batch_size) if use_random else None
    def get_all_cost_components(self): return ["rosenbrock"]

__all__ = ["RosenbrockCfg", "RosenbrockRollout"]
