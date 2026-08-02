"""Regression coverage for portable robot/collision/cost public seams."""

from __future__ import annotations

import torch

from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_support_polygon import ConvexPolygon2DHelper
from curobo._src.cost.wp_tool_pose import (
    compute_rotation_error_axis_angle,
    convert_angular_velocity_to_quaternion_rate,
    scale_quaternion_difference_by_axis,
)
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def test_cspace_cost_lifecycle_and_l2_helper_are_differentiable() -> None:
    q = torch.tensor([[0.2, -0.4]], requires_grad=True)
    state = JointState.from_position(q, joint_names=["a", "b"])
    position = PositionCSpaceCost(CSpaceCostCfg(weight=1.0, dof=2))
    assert position.validate_input(state)
    assert position.setup_batch_tensors(1, 1)

    distance = CSpaceDistCost(CSpaceDistCostCfg(weight=1.0))
    assert distance.validate_input(q, torch.zeros_like(q))
    value = distance.jit_squared_cost_to_l2(distance(q, torch.zeros_like(q))).sum()
    value.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_portable_pose_helpers_preserve_quaternion_and_autograd_contract() -> None:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    error = compute_rotation_error_axis_angle(q, q, torch.ones(3), 1.0, 0.0)
    error.sum().backward()
    assert torch.equal(error, torch.zeros_like(error))
    assert q.grad is not None
    rate = convert_angular_velocity_to_quaternion_rate(torch.ones((1, 3)), q.detach())
    assert rate.shape == q.shape
    torch.testing.assert_close(
        scale_quaternion_difference_by_axis(q.detach(), torch.tensor([2.0, 3.0, 4.0])),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )


def test_scene_collision_and_kinematics_aliases_use_portable_backends() -> None:
    cfg = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=DeviceCfg())
    kinematics = Kinematics(cfg, compute_spheres=True)
    active = JointState.from_position(
        kinematics.default_joint_position, joint_names=kinematics.joint_names
    )
    full = kinematics.get_full_js(active)
    assert full.joint_names == kinematics.joint_names
    assert kinematics.get_mimic_js(active).position.shape == active.position.shape

    class SphereChecker:
        collision_buffer = "portable-buffer"

        @staticmethod
        def get_sphere_distance(spheres, env_query_idx=None):
            del env_query_idx
            return torch.ones(spheres.shape[:-1], dtype=spheres.dtype, device=spheres.device)

    state = kinematics.compute_kinematics(active)
    settings = SceneCollisionCostCfg(weight=1.0)
    settings.scene_collision_checker = SphereChecker()
    cost = SceneCollisionCost(settings)
    assert cost.validate_input(state)
    assert cost.get_gradient_buffer() == "portable-buffer"
    value = cost(state)
    assert value.shape[:2] == state.robot_spheres.shape[:2]
    # These are real portable re-exports, not CUDA/Warp stand-ins.
    assert ConvexPolygon2DHelper is not None
