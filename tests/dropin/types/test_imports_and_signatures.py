import importlib
import inspect

import torch


def test_public_types_are_a_module_at_the_pinned_layout() -> None:
    import curobo

    types = importlib.import_module("curobo.types")

    assert curobo.__version__ == "1.0.0"
    assert not hasattr(curobo, "Pose")
    assert types.__file__.endswith("curobo/types.py")
    assert types.__all__ == [
        "JointState",
        "RobotState",
        "Pose",
        "ToolPose",
        "GoalToolPose",
        "ToolPoseCriteria",
        "CameraObservation",
        "LidarObservation",
        "ContentPath",
        "DeviceCfg",
    ]
    legacy_math = importlib.import_module("curobo.types.math")
    legacy_base = importlib.import_module("curobo.types.base")
    assert legacy_math.Pose is types.Pose
    assert legacy_base.DeviceCfg is types.DeviceCfg


def test_internal_pinned_import_paths() -> None:
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.control_space import ControlSpace
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.math import Pose as DeprecatedPose
    from curobo._src.types.pose import Pose
    from curobo._src.types.robot import RobotCfg

    assert DeprecatedPose is Pose
    assert ControlSpace.BSPLINE_5.value == 5
    assert inspect.signature(DeviceCfg.from_basic).parameters["dev_id"].default is inspect.Parameter.empty
    assert list(inspect.signature(RobotCfg).parameters) == ["kinematics", "dynamics", "device_cfg"]
    assert list(inspect.signature(JointState.from_state_tensor).parameters) == [
        "state_tensor", "joint_names", "dof"
    ]
    assert list(inspect.signature(Pose.to).parameters) == ["self", "device_cfg", "device"]


def test_import_does_not_initialize_or_require_cuda() -> None:
    from curobo.types import DeviceCfg

    assert DeviceCfg().device == torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    assert not torch.cuda.is_initialized()
