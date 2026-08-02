from __future__ import annotations

import inspect

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util import config_io
from curobo._src.util.tensor_util import copy_or_clone
from curobo._src.util.trajectory import (
    TrajInterpolationType,
    calculate_dt_no_clamp,
    get_batch_interpolated_trajectory,
    get_bspline_interpolation,
    get_cuda_linear_interpolation,
)
from curobo._src.util.trajectory_execution_manager import TrajectoryExecutionManager
from curobo._src.util.xrdf_util import ContentPath, UrdfRobotParser, convert_curobo_to_xrdf
from curobo._src.util_file import Loader, Path, yaml


def test_legacy_configuration_and_xrdf_reexports_are_live():
    assert config_io.Path is Path
    assert config_io.Loader is Loader
    assert config_io.yaml is yaml
    assert ContentPath.__name__ == "ContentPath"
    assert UrdfRobotParser.__name__ == "UrdfRobotParser"

    xrdf = convert_curobo_to_xrdf(
        {
            "robot_cfg": {
                "kinematics": {
                    "tool_frames": ["tool"],
                    "collision_spheres": {},
                    "cspace": {
                        "joint_names": ["joint"],
                        "default_joint_position": [0.0],
                        "max_acceleration": [1.0],
                        "max_jerk": [1.0],
                    },
                    "lock_joints": {},
                }
            }
        }
    )
    assert xrdf["format"] == "xrdf"
    assert xrdf["cspace"]["joint_names"] == ["joint"]


def test_tensor_buffer_contract_and_trajectory_aliases_are_portable():
    source = torch.tensor([1.0, 2.0])
    assert copy_or_clone(None, None) is None
    target = copy_or_clone(source, None)
    assert target is not source
    torch.testing.assert_close(target, source)
    with pytest.raises(ValueError, match="ref_tensor is None"):
        copy_or_clone(source, None, allow_clone=False)

    raw = JointState.from_position(torch.tensor([[[0.0], [1.0], [2.0]]]))
    raw.dt = torch.tensor([0.1])
    output, steps = get_batch_interpolated_trajectory(
        raw, torch.tensor(0.05), TrajInterpolationType.LINEAR_CUDA, device_cfg=DeviceCfg()
    )
    direct = get_cuda_linear_interpolation(raw, steps, output.clone())
    torch.testing.assert_close(direct.position, output.position)
    with pytest.raises(NotImplementedError, match="CUDA spline kernel"):
        get_bspline_interpolation(raw)


def test_execution_manager_and_limit_timestep_shapes_match_pinned_surface():
    assert list(inspect.signature(TrajectoryExecutionManager).parameters) == [
        "interpolation_steps", "command_start_idx", "command_end_idx"
    ]
    state = JointState.from_position(torch.arange(12.0).reshape(1, 6, 2))
    manager = TrajectoryExecutionManager(2, command_start_idx=1)
    manager.update_state_action_buffers(state, state.position.clone())
    torch.testing.assert_close(manager.get_next_command().position, state.position[:, 1])
    torch.testing.assert_close(manager.get_next_command().position, state.position[:, 2])
    assert not manager.has_valid_next_command()

    values = torch.ones(2, 4, 3)
    dt = calculate_dt_no_clamp(
        values, values * 4, values * 8,
        torch.ones(3), torch.full((3,), 4.0), torch.full((3,), 8.0),
    )
    torch.testing.assert_close(dt, torch.full((2,), 1.00001))
