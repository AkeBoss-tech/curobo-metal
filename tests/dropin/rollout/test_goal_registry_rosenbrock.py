"""Direct portable behavior coverage for V2 goal and Rosenbrock rollouts."""

import pytest
import torch

from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_rosenbrock import RosenbrockCfg, RosenbrockRollout
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def test_goal_registry_seed_indices_copy_and_kernel_contract():
    device_cfg = DeviceCfg()
    seed = JointState.from_position(torch.zeros(2, 2, 3))
    registry = GoalRegistry.create_idx(2, True, 3, device_cfg, seed_goal_state=seed)
    assert registry.idxs_link_pose.squeeze(-1).tolist() == [0, 0, 0, 1, 1, 1]
    assert registry.idxs_env.squeeze(-1).tolist() == [0, 0, 0, 1, 1, 1]
    assert registry.idxs_seed_goal_js.shape == (4, 1)
    target = GoalRegistry()
    target.copy_(registry, allow_clone=False)
    assert target.goal_js is None
    kernel = torch.eye(registry.get_index_size())
    transformed = registry.apply_kernel(kernel)
    assert torch.equal(transformed.idxs_goal_js, registry.idxs_goal_js)
    with pytest.raises(ValueError, match="width"):
        registry.apply_kernel(torch.eye(2))


def test_rosenbrock_generalized_cost_autograd_and_graph_executor():
    rollout = RosenbrockRollout(RosenbrockCfg(DeviceCfg(), dimensions=3, time_horizon=2, time_action_horizon=2), use_cuda_graph=True)
    action = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], requires_grad=True)
    metrics = rollout.compute_metrics_from_action(action)
    assert metrics.costs_and_constraints.costs.names == ["rosenbrock"]
    assert torch.allclose(metrics.convergence, torch.tensor([[2.0, 0.0]]))
    metrics.convergence.sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert rollout._compute_metrics_from_action_executor is not None
    rollout.reset_cuda_graph()
    assert not rollout._compute_metrics_from_action_executor.is_initialized


def test_rosenbrock_sampling_is_seeded_for_any_horizon_and_validates_shapes():
    cfg = RosenbrockCfg(DeviceCfg(), dimensions=2, time_horizon=3, time_action_horizon=3, sampler_seed=19)
    rollout = RosenbrockRollout(cfg)
    first = rollout.get_initial_action()
    assert first.shape == (1, 3, 2)
    assert torch.all((first >= -1.5) & (first <= 2.0))
    rollout.reset_seed()
    assert torch.equal(first, rollout.get_initial_action())
    # Evaluation intentionally accepts a runtime horizon different from the
    # configured sampling horizon, as V2 optimizers can evaluate a candidate
    # rollout with a shortened horizon.
    assert rollout.evaluate_action(torch.zeros(1, 2, 2)).state.position.shape == (1, 2, 2)
    with pytest.raises(ValueError, match="shape"):
        rollout.evaluate_action(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="at least 2"):
        RosenbrockCfg(DeviceCfg(), dimensions=1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_rosenbrock_mps_fallback_disabled_is_differentiable(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    rollout = RosenbrockRollout(RosenbrockCfg(DeviceCfg(torch.device("mps")), time_horizon=2, time_action_horizon=2))
    action = rollout.get_initial_action().requires_grad_(True)
    result = rollout.evaluate_action(action)
    value = result.costs_and_constraints.get_sum_cost().sum()
    value.backward()
    assert result.state.position.device.type == "mps"
    assert action.grad is not None and action.grad.device.type == "mps"
