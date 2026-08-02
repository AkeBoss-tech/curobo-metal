"""Differentiable CPU/MPS Rosenbrock rollout with the cuRobo V2 interface.

This is a real portable rollout rather than a CUDA shim: costs, gradients,
seeded sampling, and direct graph-executor lifecycle all run with regular
PyTorch on CPU and MPS.  ``use_cuda_graph`` preserves the high-level
shape-stable executor API, but it does not create a raw CUDA graph.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer

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

    def __post_init__(self) -> None:
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool) or self.dimensions < 2:
            raise ValueError("dimensions must be an integer of at least 2")
        for name in ("time_horizon", "time_action_horizon"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.sampler_seed, int) or isinstance(self.sampler_seed, bool):
            raise ValueError("sampler_seed must be an integer")

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
        self._use_cuda_graph = bool(use_cuda_graph)
        self._compute_metrics_from_state_executor: Optional[GraphExecutor] = None
        self._compute_metrics_from_action_executor: Optional[GraphExecutor] = None
        self.start_state = None
        self.rollout_instance_name = None
        self._batch_size = 1
        self._action_bound_lows = torch.full((self.action_horizon,), -1.5, **self.device_cfg.as_torch_dict())
        self._action_bound_highs = torch.full((self.action_horizon,), 2.0, **self.device_cfg.as_torch_dict())
        # V2's sample-buffer backend is seeded and device resident.  Bounds
        # here are action-*dimension* bounds; action-horizon bounds remain the
        # public V2 property above and can have a different length.
        self.act_sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            up_bounds=torch.full((self.action_dim,), 2.0, **self.device_cfg.as_torch_dict()),
            low_bounds=torch.full((self.action_dim,), -1.5, **self.device_cfg.as_torch_dict()),
            seed=self.sampler_seed,
            device_cfg=self.device_cfg,
        )
    action_dim = property(lambda self: self.dimensions)
    action_bound_lows = property(lambda self: self._action_bound_lows)
    action_bound_highs = property(lambda self: self._action_bound_highs)
    action_bounds = property(lambda self: torch.stack((self._action_bound_lows, self._action_bound_highs)))
    horizon = property(lambda self: self.time_horizon)
    action_horizon = property(lambda self: self.time_action_horizon)
    state_bounds = property(lambda self: {"position": [-2.048, 2.048]})
    batch_size = property(lambda self: self._batch_size, lambda self, v: setattr(self, "_batch_size", v))
    dt = property(lambda self: 1.0)
    def _validate_actions(self, act_seq: torch.Tensor) -> None:
        if not isinstance(act_seq, torch.Tensor):
            raise TypeError("act_seq must be a torch.Tensor")
        if act_seq.ndim != 3:
            raise ValueError("act_seq must have shape [batch, horizon, action_dim]")
        if act_seq.shape[-1] != self.action_dim:
            raise ValueError(f"act_seq action dimension must be {self.action_dim}")
        if not self.device_cfg.is_same_torch_device(act_seq.device):
            raise ValueError(
                f"act_seq is on {act_seq.device}, expected {self.device_cfg.device}; "
                "move inputs explicitly rather than relying on a host copy"
            )

    def _cost(self, value: torch.Tensor) -> torch.Tensor:
        return ((self.a - value[..., :-1]).square()
                + self.b * (value[..., 1:] - value[..., :-1].square()).square()).sum(-1, keepdim=True)

    def _compute_state_from_action_impl(self, act_seq):
        self._validate_actions(act_seq)
        return JointState(position=act_seq, device_cfg=self.device_cfg)

    def _compute_state_from_action_metrics_impl(self, act_seq):
        return self._compute_state_from_action_impl(act_seq)

    def _collections(self, state):
        costs = CostCollection(); costs.add(self._cost(state.position), "rosenbrock")
        return CostsAndConstraints(costs=costs)

    def _compute_costs_and_constraints_impl(self, state, **kwargs):
        if not isinstance(state, JointState):
            raise TypeError("state must be a JointState")
        self._validate_actions(state.position)
        return self._collections(state)

    def _compute_costs_and_constraints_metrics_impl(self, state, **kwargs):
        return self._compute_costs_and_constraints_impl(state, **kwargs)

    def evaluate_action(self, act_seq, **kwargs):
        self._validate_actions(act_seq)
        self.update_batch_size(act_seq.shape[0])
        state = self._compute_state_from_action_impl(act_seq)
        return RolloutResult(actions=act_seq, costs_and_constraints=self._collections(state), state=state)

    def _compute_metrics_from_state_impl(self, state, **kwargs):
        cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        convergence = cc.get_sum_cost(sum_horizon=False)
        return RolloutMetrics(
            state=state, costs_and_constraints=cc,
            feasible=cc.get_feasible(), convergence=convergence,
        )

    def compute_metrics_from_state(self, state, **kwargs):
        if self._use_cuda_graph:
            if self._compute_metrics_from_state_executor is None:
                self._compute_metrics_from_state_executor = create_graph_executor(
                    capture_fn=self._compute_metrics_from_state_impl,
                    device=self.device_cfg.device,
                )
            return self._compute_metrics_from_state_executor(state)
        return self._compute_metrics_from_state_impl(state, **kwargs)

    def _compute_metrics_from_action_impl(self, act_seq, **kwargs):
        state = self._compute_state_from_action_impl(act_seq)
        result = self._compute_metrics_from_state_impl(state, **kwargs)
        result.actions = act_seq
        return result

    def compute_metrics_from_action(self, act_seq, **kwargs):
        self._validate_actions(act_seq)
        self.update_batch_size(act_seq.shape[0])
        if self._use_cuda_graph:
            if self._compute_metrics_from_action_executor is None:
                self._compute_metrics_from_action_executor = create_graph_executor(
                    capture_fn=self._compute_metrics_from_action_impl,
                    device=self.device_cfg.device,
                )
            return self._compute_metrics_from_action_executor(act_seq)
        return self._compute_metrics_from_action_impl(act_seq, **kwargs)
    def update_params(self, a=None, b=None, **kwargs):
        if a is not None: self.a = a
        if b is not None: self.b = b
        return True
    def update_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        self._batch_size = batch_size
    def update_dt(self, dt, **kwargs): return True
    def reset(self, reset_problem_ids=None, **kwargs): return True
    def reset_shape(self): return True
    def reset_cuda_graph(self):
        if self._compute_metrics_from_state_executor is not None:
            self._compute_metrics_from_state_executor.reset()
        if self._compute_metrics_from_action_executor is not None:
            self._compute_metrics_from_action_executor.reset()
        return self.reset_shape()

    def reset_seed(self):
        self.act_sample_gen.reset()

    def sample_random_actions(self, n=0, bounded=True):
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("n must be a non-negative integer")
        return self.act_sample_gen.get_samples(n, bounded=bounded)

    def get_initial_action(self, use_random=True, use_zero=False, **kwargs):
        if use_random:
            return self.sample_random_actions(
                self.batch_size * self.action_horizon, bounded=True
            ).view(self.batch_size, self.action_horizon, self.action_dim)
        if use_zero:
            return torch.zeros(
                (self.batch_size, self.action_horizon, self.action_dim),
                **self.device_cfg.as_torch_dict(),
            )
        log_and_raise("get_init_action_seq is not implemented")

    def get_all_cost_components(self):
        return {}

__all__ = ["RosenbrockCfg", "RosenbrockRollout"]
