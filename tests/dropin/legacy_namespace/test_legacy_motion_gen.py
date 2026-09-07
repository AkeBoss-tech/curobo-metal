"""Zero-edit coverage for the established cuRobo application namespace."""

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig,
)


def test_established_motion_gen_application_runs_unchanged_on_portable_backend():
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml", None,
        tensor_args=TensorDeviceType(device="cpu"),
        num_ik_seeds=2,
        num_trajopt_seeds=1,
    )
    motion_gen = MotionGen(config)
    start_state = JointState.from_position(
        motion_gen.get_retract_config().view(1, -1),
        joint_names=motion_gen.ik_solver.kinematics.joint_names,
    )
    kinematics = motion_gen.ik_solver.kinematics.compute_kinematics(start_state)
    goal_pose = Pose(
        kinematics.tool_poses.position[:, 0, 0],
        kinematics.tool_poses.quaternion[:, 0, 0],
    )
    result = motion_gen.plan_single(
        start_state, goal_pose,
        MotionGenPlanConfig(max_attempts=1, enable_graph=False),
    )
    assert result is not None
    assert bool(result.success.all().item())
