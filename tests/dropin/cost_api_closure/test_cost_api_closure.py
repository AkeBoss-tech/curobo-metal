"""Behavioral coverage for the portable pinned-cuRoboV2 cost façade."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_state import StateCSpaceCost
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.cost.cost_support_polygon import CostSupportPolygon
from curobo._src.cost.cost_support_polygon_cfg import CostSupportPolygonCfg
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.wp_cspace_position import PositionCSpaceFunction
from curobo._src.cost.wp_cspace_state import StateCSpaceFunction
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


def _limits() -> JointLimits:
    pair = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    return JointLimits(["a", "b"], pair, pair * 2, pair * 3, pair * 4, pair * 5, device_cfg=DeviceCfg("cpu"))


def test_cost_enable_disable_preserves_reusable_config() -> None:
    config = BaseCostCfg(weight=[2.0], device_cfg=DeviceCfg("cpu"))
    from curobo._src.cost.cost_base import BaseCost

    cost = BaseCost(config)
    cost.disable_cost()
    assert not cost.enabled and config.weight.item() == 2.0
    cost.enable_cost()
    assert cost.enabled and cost.weight.item() == 2.0
    assert cost.setup_batch_tensors(2, 3)
    assert cost.forward().shape == (2, 3, 1)


def test_derived_cost_config_clone_keeps_type_and_independent_tensor_state() -> None:
    cfg = CSpaceCostCfg(
        weight=[1.0, 2.0], dof=2, cost_type=CSpaceCostType.POSITION,
        activation_distance=[0.0, 0.1], joint_limits=_limits(),
        cspace_target_weight=3.0, cspace_target_dof_weight=[4.0, 5.0],
        device_cfg=DeviceCfg("cpu"),
    )
    copied = cfg.clone()
    assert isinstance(copied, CSpaceCostCfg)
    assert copied.joint_limits is not cfg.joint_limits
    copied.weight[0] = 99.0
    copied.cspace_target_dof_weight[0] = 88.0
    assert cfg.weight[0].item() == 1.0
    assert cfg.cspace_target_dof_weight[0].item() == 4.0


def test_raw_warp_cost_bridges_fail_precisely() -> None:
    with pytest.raises(RuntimeError, match="raw Warp/CUDA ABI"):
        PositionCSpaceFunction.apply(torch.zeros(1))
    with pytest.raises(RuntimeError, match="raw Warp/CUDA ABI"):
        StateCSpaceFunction.apply(torch.zeros(1))


def test_cspace_position_and_state_bounds_targets_are_differentiable() -> None:
    q = torch.tensor([[[1.2, 0.0], [0.6, -0.2]]], requires_grad=True)
    state = JointState.from_position(q)
    target = JointState.from_position(torch.zeros((1, 2)))
    cfg = CSpaceCostCfg(
        weight=[2.0, 0.0], dof=2, cost_type=CSpaceCostType.POSITION,
        activation_distance=[0.1, 0.0], joint_limits=_limits(),
        cspace_target_weight=1.0, cspace_target_dof_weight=[1.0, 2.0],
        cspace_non_terminal_weight_factor=0.25,
        device_cfg=DeviceCfg("cpu"),
    )
    value = PositionCSpaceCost(cfg)(state, target_joint_state=target)
    assert value.shape == (1, 2, 2)
    value.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad[0, 0, 0] > 0

    q2 = q.detach().clone().requires_grad_()
    state2 = JointState(q2, torch.ones_like(q2) * 3, torch.zeros_like(q2), jerk=torch.zeros_like(q2), device_cfg=DeviceCfg("cpu"))
    state_cfg = CSpaceCostCfg(
        weight=[1.0] * 5, dof=2, cost_type="state", activation_distance=[0.0] * 5,
        joint_limits=_limits(), squared_l2_regularization_weight=[.1] * 5,
        device_cfg=DeviceCfg("cpu"),
    )
    state_value = StateCSpaceCost(state_cfg)(state2, joint_torque=torch.ones_like(q2) * 7)
    state_value.sum().backward()
    assert state_value.shape == (1, 2, 2)
    assert torch.isfinite(q2.grad).all()


def test_cspace_distance_goal_indices_terminal_weight_and_distance_output() -> None:
    q = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [4.0, 3.0]]], requires_grad=True)
    goal = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    cfg = CSpaceDistCostCfg(weight=2.0, dof=2, only_terminal_cost=True, device_cfg=DeviceCfg("cpu"))
    cfg.update_terminal_dof_weight([2.0, 3.0])
    cost = CSpaceDistCost(cfg)
    result, distance = cost.forward_out_distance(q, goal, torch.tensor([0, 1]))
    assert torch.equal(result[:, 0], torch.zeros_like(result[:, 0]))
    torch.testing.assert_close(distance.square(), result / 2.0)
    result.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_tool_pose_goalset_criteria_and_gradient() -> None:
    position = torch.tensor([[[[.2, 0.0, 0.0]], [[.5, 0.0, 0.0]]]], requires_grad=True)
    quat = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    current = ToolPose(["tool"], position, quat)
    goals = GoalToolPose(
        ["tool"],
        torch.tensor([[[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]]),
        torch.tensor([[[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]]]),
    )
    cfg = ToolPoseCostCfg(weight=1.0, tool_frames=["tool"], device_cfg=DeviceCfg("cpu"))
    output, linear, angular, idx = ToolPoseCost(cfg)(current, goals)
    assert output.shape == (1, 2, 2)
    assert linear.shape == angular.shape == idx.shape == (1, 2, 1)
    assert idx[0, 0, 0].item() == 0 and idx[0, 1, 0].item() == 0
    output.sum().backward()
    assert torch.isfinite(position.grad).all() and torch.isfinite(quat.grad).all()


def test_tool_pose_criteria_partial_updates_tolerance_and_goal_frame_projection() -> None:
    position = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], requires_grad=True)
    quat = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]], requires_grad=True)
    current = ToolPose(["a", "b"], position, quat)
    # The goal frame for a is rotated 90 degrees around z.  Tracking only its
    # goal-frame y axis distinguishes projected from world-frame translation.
    half = 2**-0.5
    goals = GoalToolPose(
        ["a", "b"],
        torch.zeros((1, 1, 2, 1, 3)),
        torch.tensor([[[[[half, 0.0, 0.0, half]], [[1.0, 0.0, 0.0, 0.0]]]]]),
    )
    cfg = ToolPoseCostCfg(weight=1.0, tool_frames=["a", "b"], device_cfg=DeviceCfg("cpu"))
    cfg.tool_pose_criteria["a"] = ToolPoseCriteria(
        terminal_pose_axes_weight_factor=[0, 1, 0, 0, 0, 0],
        non_terminal_pose_axes_weight_factor=[0, 1, 0, 0, 0, 0],
        project_distance_to_goal=True,
        device_cfg=DeviceCfg("cpu"),
    )
    cost = ToolPoseCost(cfg)
    initial, _, _, _ = cost(current, goals)
    assert initial[0, 0, 0] > 0  # x-world is y in the rotated goal frame.
    before_b = cost._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor[1].clone()
    cost.update_tool_pose_criteria({"a": ToolPoseCriteria([0.] * 6, [0.] * 6, device_cfg=DeviceCfg("cpu"))})
    assert torch.equal(cost._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor[1], before_b)
    disabled, _, _, _ = cost(current, goals)
    assert disabled[0, 0, 0].item() == 0.0

    tolerance_cfg = ToolPoseCostCfg(weight=1.0, tool_frames=["a"], device_cfg=DeviceCfg("cpu"))
    tolerance_cfg.tool_pose_criteria["a"] = ToolPoseCriteria(
        terminal_pose_convergence_tolerance=[1.0, 0.0],
        device_cfg=DeviceCfg("cpu"),
    )
    tolerance_cost = ToolPoseCost(tolerance_cfg)
    single = ToolPose(["a"], position[:, :, :1], quat[:, :, :1])
    single_goal = GoalToolPose(
        ["a"], goals.position[:, :, :1], torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]]),
    )
    output, *_ = tolerance_cost(single, single_goal)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert torch.isfinite(position.grad).all() and torch.isfinite(quat.grad).all()


class _Checker:
    def get_sphere_distance(self, spheres, env_query_idx=None):
        del env_query_idx
        return torch.tensor([[[.3, -.2, .3]]], device=spheres.device, dtype=spheres.dtype)


class _SweepChecker:
    def __init__(self):
        self.speed_metric = None

    def get_swept_sphere_distance(self, state, buffer, weight, *, activation_distance,
                                  trajectory_dt, enable_speed_metric, env_query_idx, return_loss):
        del buffer, weight, activation_distance, trajectory_dt, env_query_idx, return_loss
        self.speed_metric = enable_speed_metric
        return state.robot_spheres.new_zeros(state.robot_spheres.shape[:2])


@dataclass
class _SelfConfig:
    num_spheres: int = 3
    sphere_padding: float = 0.0
    collision_pairs: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 2]]))


def test_scene_and_self_collision_reductions_and_support_polygon() -> None:
    spheres = torch.tensor([[[[0., 0., 0., .2], [.6, 0., 0., .2], [.1, .7, 0., .2]]]], requires_grad=True)
    scene = SceneCollisionCost(SceneCollisionCostCfg(
        weight=2.0,
        activation_distance=.1,
        sum_distance=False,
        _scene_collision_checker=_Checker(),
        device_cfg=DeviceCfg("cpu"),
    ))
    scene_value = scene(spheres)
    assert scene_value.shape == (1, 1, 3) and bool((scene_value > 0).any())

    self_cost = SelfCollisionCost(SelfCollisionCostCfg(weight=1.0, self_collision_kin_config=_SelfConfig(), store_pair_distance=True, device_cfg=DeviceCfg("cpu")))
    self_value = self_cost(spheres)
    assert self_value.shape == (1, 1, 1) and self_cost._pair_distance.shape == (1, 1, 1)
    (scene_value.sum() + self_value.sum()).backward()
    assert torch.isfinite(spheres.grad).all()

    support = CostSupportPolygon(CostSupportPolygonCfg(weight=1.0, foot_sphere_indices=torch.tensor([0, 1, 2]), device_cfg=DeviceCfg("cpu")))
    com = torch.tensor([[[2.0, 0.0, 0.0]]])
    support_value = support(com, spheres.detach())
    assert support_value.shape == (1, 1) and support_value.item() > 0


def test_scene_swept_cost_forwards_speed_metric_and_validates_lifecycle() -> None:
    spheres = torch.zeros((1, 3, 1, 4))
    state = type("State", (), {"robot_spheres": spheres})()
    checker = _SweepChecker()
    cost = SceneCollisionCost(SceneCollisionCostCfg(
        weight=1.0, num_spheres=1, use_sweep=True, use_speed_metric=True,
        _scene_collision_checker=checker,
        device_cfg=DeviceCfg("cpu"),
    ))
    cost.setup_batch_tensors(1, 3)
    result = cost(state, trajectory_dt=torch.tensor([0.1]))
    assert result.shape == (1, 3) and checker.speed_metric is True
    with pytest.raises(ValueError, match="idxs_env_query"):
        cost(state, idxs_env_query=torch.zeros(2, dtype=torch.long), trajectory_dt=torch.tensor([0.1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_cost_closure_mps_stays_resident_without_fallback() -> None:
    device = DeviceCfg(torch.device("mps"))
    q = torch.tensor([[[.3, -.4]]], device="mps", requires_grad=True)
    cfg = CSpaceDistCostCfg(weight=1.0, dof=2, device_cfg=device, only_terminal_cost=False)
    value = CSpaceDistCost(cfg)(q, torch.zeros((1, 2), device="mps"), torch.tensor([0], device="mps"))
    value.sum().backward()
    assert value.device.type == q.grad.device.type == "mps"
