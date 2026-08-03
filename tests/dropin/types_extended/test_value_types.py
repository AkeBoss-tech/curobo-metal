import inspect

import pytest
import torch

from curobo.types import (
    CameraObservation, ContentPath, GoalToolPose, JointState, LidarObservation,
    Pose, RobotState, ToolPose, ToolPoseCriteria,
)


def test_public_exports_and_pinned_signatures():
    import curobo.types as types
    assert types.__all__ == [
        "JointState", "RobotState", "Pose", "ToolPose", "GoalToolPose",
        "ToolPoseCriteria", "CameraObservation", "LidarObservation", "ContentPath", "DeviceCfg",
    ]
    assert list(inspect.signature(CameraObservation).parameters) == [
        "name", "rgb_image", "depth_image", "image_segmentation", "projection_matrix",
        "projection_rays", "resolution", "pose", "intrinsics", "timestamp",
        "depth_to_meter", "feature_grid",
    ]
    assert list(inspect.signature(RobotState).parameters) == [
        "joint_state", "joint_torque", "cuda_robot_model_state",
    ]
    assert list(inspect.signature(Pose.multiply).parameters) == [
        "self", "other_pose", "out_position", "out_quaternion",
    ]


def test_content_path_resolves_and_rejects_ambiguous_paths(tmp_path):
    value = ContentPath(robot_config_root_path=str(tmp_path), robot_config_file="robot.yml")
    assert value.get_robot_configuration_path() == str(tmp_path / "robot.yml")
    with pytest.raises(ValueError, match="cannot be provided together"):
        ContentPath(robot_config_file="a.yml", robot_config_absolute_path="/a.yml")
    with pytest.raises(ValueError, match="No Robot"):
        ContentPath().get_robot_configuration_path()


def test_tool_pose_and_goal_pose_shapes_selection_and_mutation():
    position = torch.arange(12.0).view(1, 2, 2, 3)
    quaternion = torch.zeros(1, 2, 2, 4); quaternion[..., 0] = 1
    value = ToolPose(["a", "b"], position, quaternion)
    assert value["b"].position.shape == (2, 3)
    goal = value.as_goal(["b"])
    assert goal.shape == (1, 2, 1, 1, 3)
    assert GoalToolPose.from_poses({"a": value["a"]}).shape == (2, 1, 1, 1, 3)
    clone = value.clone()
    clone.copy_(value)
    with pytest.raises(ValueError, match="4D"):
        ToolPose(["a"], torch.zeros(1, 1, 3), torch.zeros(1, 1, 4))


def test_observation_clone_copy_device_projection_and_errors():
    depth = torch.ones(2, 2)
    intrinsics = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    camera = CameraObservation(depth_image=depth, intrinsics=intrinsics)
    camera.update_projection_rays()
    assert camera.get_pointcloud().shape == (1, 2, 2, 3)
    with pytest.raises(ValueError, match="rgb_image"):
        _ = camera.shape
    lidar = LidarObservation(range_image=torch.ones(1, 2, 3))
    copied = lidar.clone().to(torch.device("cpu"))
    assert copied.shape == (1, 2, 3)
    assert copied.copy_(lidar) is copied


def test_criteria_robot_state_and_high_use_pose_joint_methods():
    criteria = ToolPoseCriteria.track_position_and_orientation()
    assert criteria.terminal_pose_axes_weight_factor.device.type == "cpu"
    assert criteria.project_distance_to_goal.dtype == torch.uint8
    with pytest.raises(ValueError, match="6 floats"):
        ToolPoseCriteria(terminal_pose_axes_weight_factor=[1.])

    pose = Pose.from_list([1, 0, 0, 1, 0, 0, 0])
    output_position = torch.empty_like(pose.position)
    result = pose.multiply(Pose.from_list([0, 2, 0, 1, 0, 0, 0]), output_position)
    assert result.position.data_ptr() == output_position.data_ptr()
    assert torch.allclose(result.linear_distance(pose), torch.tensor([2.]))
    points = torch.zeros(1, 2, 3)
    assert pose.batch_transform_points(points).shape == points.shape

    joints = JointState.from_position(torch.zeros(2, 3), ["a", "b", "c"])
    robot = RobotState(joints)
    assert len(robot) == 2 and robot.tool_frames == []
    with pytest.raises(ValueError, match="Link poses"):
        robot.get_link_pose("missing")
    assert joints.stack(joints).shape == (4, 3)
    assert joints.cat(joints, 0).shape == (4, 3)
