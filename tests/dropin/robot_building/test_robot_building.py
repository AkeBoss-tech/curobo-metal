import importlib

import pytest
import torch

from curobo.content import get_assets_path
from curobo.kinematics import KinematicsCfg
from curobo.robot_builder import RobotBuilder
from curobo.robot_parser import UrdfRobotParser
from curobo._src.robot.dynamics import Dynamics, DynamicsCfg
from curobo._src.robot.kinematics.kinematics_reducer import KinematicsReducer
from curobo._src.robot.loader import KinematicsLoader
from curobo._src.robot.types import CSpaceParams, JointLimits, JointType
from curobo._src.state.state_joint import JointState


URDF = get_assets_path() / "robot/franka_description/franka_panda.urdf"


def test_owned_pinned_modules_import():
    names = [
        "curobo.robot_builder", "curobo.robot_parser",
        "curobo._src.robot.builder.builder_robot",
        "curobo._src.robot.builder.debugger_robot",
        "curobo._src.robot.dynamics.dynamics",
        "curobo._src.robot.dynamics.dynamics_cfg",
        "curobo._src.robot.loader.kinematics_loader",
        "curobo._src.robot.loader.kinematics_loader_cfg",
        "curobo._src.robot.loader.util",
        "curobo._src.robot.parser.parser_base",
        "curobo._src.robot.parser.parser_urdf",
        "curobo._src.robot.types.collision_geometry",
        "curobo._src.robot.types.cspace_params",
        "curobo._src.robot.types.joint_limits",
        "curobo._src.robot.types.joint_types",
        "curobo._src.robot.types.link_params",
        "curobo._src.robot.types.self_collision_params",
        "curobo._src.robot.kinematics.kinematics_reducer",
    ]
    for name in names:
        importlib.import_module(name)


def test_urdf_parser_preserves_tree_mimic_and_inertials():
    parser = UrdfRobotParser(str(URDF))
    assert parser.root_link == "base_link"
    assert parser.get_chain("base_link", "panda_hand")[-1] == "panda_hand"
    assert "panda_joint1" in parser.get_actuated_joint_names()
    assert parser.get_mimic_joint_map() == {}
    params = parser.get_link_parameters("panda_link1")
    assert params.joint_type is JointType.Z_ROT
    assert params.fixed_transform.shape == (3, 4)
    assert params.link_mass > 0
    assert "robot" in parser.get_urdf_string()
    with pytest.raises(NotImplementedError, match="mesh"):
        parser.get_link_mesh("panda_link1")


def test_builder_and_loader_compile_full_urdf():
    builder = RobotBuilder(str(URDF), tool_frames=["panda_hand"])
    config = builder.build()
    loader = KinematicsLoader(config)
    assert loader.kinematics_config.num_dof == 9
    assert loader.kinematics_parser.root_link == "base_link"
    limits = loader.get_joint_limits()
    assert limits.position.shape == (2, 9)
    assert builder._create_neighbor_ignore_matrix()["panda_link1"] == [
        "panda_link0", "panda_link2"
    ]
    with pytest.raises(NotImplementedError, match="sphere fitting"):
        builder.fit_collision_spheres()


def test_joint_limits_and_cspace_reindex_scale_clone():
    cfg = CSpaceParams(
        ["a", "b"], [0.0, 1.0], [1.0, 2.0], [1.0, 1.0],
        velocity_scale=[0.5, 0.25],
    )
    limits = JointLimits.from_data_dict({
        "joint_names": ["a", "b"],
        "position": [[-1, -2], [1, 2]],
        "velocity": [[-4, -8], [4, 8]],
        "acceleration": [[-2, -2], [2, 2]],
        "jerk": [[-10, -10], [10, 10]],
    })
    scaled = cfg.scale_joint_limits(limits)
    torch.testing.assert_close(scaled.velocity[1], torch.tensor([2.0, 2.0]))
    cfg.inplace_reindex(["b", "a"])
    assert cfg.joint_names == ["b", "a"]
    torch.testing.assert_close(cfg.default_joint_position, torch.tensor([1.0, 0.0]))


def test_franka_dynamics_batches_and_autograd():
    cfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
    dynamics = Dynamics(DynamicsCfg(cfg.kinematics_config, cfg.device_cfg))
    q = torch.zeros((2, 3, cfg.dof), requires_grad=True)
    state = JointState(
        q, torch.zeros_like(q), torch.zeros_like(q),
        joint_names=cfg.kinematics_config.joint_names,
    )
    torque = dynamics.compute_inverse_dynamics(state)
    assert torque.shape == q.shape and torch.isfinite(torque).all()
    torch.autograd.grad(torque.sum(), q)
    dynamics.update_link_mass("panda_link1", 3.0)
    with pytest.raises(NotImplementedError, match="external"):
        dynamics.compute_inverse_dynamics(
            state, f_ext=torch.ones((2, 3, 1, 6))
        )


def test_kinematics_reducer_and_state_reconstruction():
    cfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
    reduced = KinematicsReducer.reduce_dof(
        cfg.kinematics_config, ["panda_link4"], remove_collision_spheres=True
    )
    assert reduced.tool_frames == ["panda_link4"]
    assert reduced.num_dof == 4
    state = JointState.from_position(
        torch.arange(4.0), joint_names=reduced.joint_names
    )
    locked = JointState.from_position(torch.tensor([9.0]), joint_names=["extra"])
    rebuilt = KinematicsReducer.reconstruct_joint_state(
        state, locked, reduced.joint_names + ["extra"]
    )
    assert rebuilt.joint_names[-1] == "extra"
    assert rebuilt.position[-1].item() == 9.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_dynamics_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg
    cfg = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"))
    )
    dynamics = Dynamics(DynamicsCfg(cfg.kinematics_config, cfg.device_cfg))
    q = torch.zeros((1, cfg.dof), device="mps")
    result = dynamics.compute_inverse_dynamics(
        JointState(
            q, torch.zeros_like(q), torch.zeros_like(q),
            joint_names=cfg.kinematics_config.joint_names,
        )
    )
    assert result.device.type == "mps"
