"""Portable RobotRollout lifecycle tests independent of CUDA graph internals."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


class _Transition:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = 2
        self.action_horizon = 3
        self.horizon = 3
        self.action_bound_lows = torch.tensor([-2.0, -1.0])
        self.action_bound_highs = torch.tensor([2.0, 3.0])
        self.default_joint_position = torch.tensor([0.25, -0.25])
        self._dt = torch.tensor([0.1])
        self.updated_batches = []

    def update_batch_size(self, batch_size):
        self.updated_batches.append(batch_size)

    def update_traj_dt(self, dt):
        self._dt = torch.as_tensor(dt).reshape(-1)

    def forward(self, start, action, *args, **kwargs):
        del start, args, kwargs
        return RobotState(JointState.from_position(action))

    def filter_robot_state(self, state):
        return state

    def get_robot_command(self, current, action, shift_steps=1, **kwargs):
        del kwargs
        return JointState.from_position(current.position + action[:, shift_steps - 1])


def _config():
    device = DeviceCfg()
    transition = SimpleNamespace(class_type=_Transition)
    empty = RobotCostManagerCfg()
    return RobotRolloutCfg(
        device_cfg=device,
        sampler_seed=17,
        transition_model_cfg=transition,
        cost_cfg=empty,
        constraint_cfg=empty,
        hybrid_cost_constraint_cfg=empty,
        convergence_cfg=empty,
    )


def test_rollout_constructs_transition_and_separated_manager_lifecycle():
    rollout = RobotRollout(_config())
    assert rollout.action_dim == 2
    assert rollout.action_bounds.shape == (2, 2)
    assert len(rollout._cost_manager_list) == 7
    assert rollout.default_joint_position.tolist() == [0.25, -0.25]

    action = torch.randn(2, 3, 2, requires_grad=True)
    result = rollout.evaluate_action(action)
    metrics = rollout.compute_metrics_from_action(action)
    assert result.state.joint_state.position is action
    assert metrics.actions is action
    assert metrics.feasible is True
    assert rollout.transition_model.updated_batches[-1] == 2


def test_rollout_goal_expansion_dt_sampling_and_command_are_deterministic():
    rollout = RobotRollout(_config())
    current = JointState.from_position(torch.zeros(2, 2))
    seed = JointState.from_position(torch.zeros(2, 1, 2))
    seed.dt = torch.full((2, 1), 0.1)
    goal = GoalRegistry(current_js=current, seed_goal_js=seed)
    rollout.update_params(goal, num_particles=3)
    assert rollout.batch_size == 6
    assert rollout.start_state.position.shape == (2, 2)

    samples_a = rollout.sample_random_actions(2)
    rollout.reset_seed()
    samples_b = rollout.sample_random_actions(2)
    torch.testing.assert_close(samples_a, samples_b)
    assert bool((samples_a >= torch.tensor([-2.0, -1.0])).all())
    assert bool((samples_a <= torch.tensor([2.0, 3.0])).all())

    replacement = goal.clone()
    replacement.seed_goal_js.dt.fill_(0.25)
    rollout.update_goal_dt(replacement)
    assert float(rollout._metrics_goal.seed_goal_js.dt.flatten()[0]) == pytest.approx(0.25)
    assert rollout.update_goal_dt(0.2) is True
    assert rollout.dt == pytest.approx(0.2)

    command = rollout.get_robot_command(current, torch.ones(2, 3, 2), shift_steps=2)
    torch.testing.assert_close(command.position, torch.ones(2, 2))


def test_rollout_cuda_graph_and_invalid_inputs_are_explicit_boundaries():
    rollout = RobotRollout(_config(), use_cuda_graph=True)
    assert rollout.valid_compute_metrics_from_action_cuda_graph() is False
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        rollout.reset_cuda_graph()
    with pytest.raises(ValueError, match="shape"):
        rollout.evaluate_action(torch.zeros(2, 2))
